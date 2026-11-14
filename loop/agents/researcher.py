"""loop.agents.researcher — one LLM agent per candidate (DESIGN §9): worktree-confined tools, then the
harness (diff -> surface check -> smoke -> commit -> init-compat) builds a typed ResearchResult.
The worktree is always reaped."""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from loop.adapters import flatten
from loop import infra as INFRA
from loop import databank as DB
from loop import project as P
from loop import novelty as NV
from loop.agents import briefs
from loop.agents import ideas as IDEAS
from loop.core import factorial as FAC
from loop.core.data import AXES, OFFICIAL_WEIGHTS
from loop.contracts import (PANEL, ROOT_NODE_ID, AxisPrediction, Candidate, CandidateRequest,
                            HandoffNote, HypothesisKind, HypothesisRecord, LeverTags, Operator,
                            ResearchResult, Strategy, Worktree)
from loop.spec import LoopContext
from loop.select import intake
from loop.select.embed import HashingEmbedder
from loop.select.intake import path_matches

BANNED_COMMANDS = (
    (r"\bgit\s+(reset|checkout|switch|stash|clean|rebase|commit|merge|push|pull|fetch|worktree|"
     r"restore|am|cherry-pick|revert|tag|update-ref)\b|\bgit\s+branch\s+-[dDmM]|"
     # ... and the same through python's subprocess (["git", "commit", ...]): 11-08, python now runs here
     r"[\"']git[\"']\s*,\s*[\"'](reset|checkout|switch|stash|clean|rebase|commit|merge|push|pull|fetch|worktree|"
     r"restore|am|cherry-pick|revert|tag|update-ref|apply|branch)[\"']", "git state change"),
    (r"\b(sbatch|srun|scancel|salloc|scontrol)\b", "SLURM"),
    (r"\b(curl|wget|ssh|scp|rsync)\b", "network"),
    (r"\bsudo\b", "sudo"),
    # the worktree's .venv is a symlink to the SHARED base venv: `uv run` / `uv sync` / pip would re-sync it under
    # running jobs. Run python directly: `.venv/bin/python ...`
    (r"(?<![\w./-])uvx?\s+\S|(?<![\w./-])uvx?\s*$|(?<![\w./-])(pip3?|pipx|conda|mamba|poetry)\s+\S|"
     r"\bpython3?\s+-m\s+(pip|ensurepip|venv)\b|\.venv/bin/(pip3?|uv)\b",
     "package manager / venv mutation (use .venv/bin/python directly)"),
)
#: env for every researcher shell command: uv must never re-sync / re-lock the shared venv, pip never installs
SAFE_ENV = {"UV_NO_SYNC": "1", "UV_FROZEN": "1", "PIP_NO_INPUT": "1", "PIP_REQUIRE_VIRTUALENV": "1"}
_BANNED = [(re.compile(p), why) for p, why in BANNED_COMMANDS]
_DENY = re.compile(r"ledger|kgraph|(?<![\w.])/data\b|nanopath-old|overfitX|injected_ideas|IDEAS\.md|labless", re.I)
_SUBST = re.compile(r"`|\$\(")                 # command substitution: its output could name any path
_VAR = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")   # a $VAR / ${VAR} reference: only shell-local ones are allowed
_ASSIGN = re.compile(r"(?:^|[\s;&|(])([A-Za-z_][A-Za-z0-9_]*)=")
_FOR_VAR = re.compile(r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\b")
_QUOTED_HEREDOC = re.compile(r"<<-?\s*(['\"])(\w+)\1.*?\n\s*\2\b", re.S)
_SINGLE = re.compile(r"'[^']*'")


def _expanded(command: str) -> str:
    """The parts of ``command`` the shell expands: single-quoted strings (awk / sed programs, python code) and the
    bodies of quoted heredocs (``<<'EOF'``) taken out -- a ``$NF`` there is no shell variable."""
    return _SINGLE.sub("''", _QUOTED_HEREDOC.sub("<<EOF", command))


#: 11-08: python code a session runs (inline or a script of its worktree) may not reach the network
_NET_PY = re.compile(r"\b(import\s+(socket|urllib|requests|httpx|aiohttp|ftplib|smtplib|paramiko|http\.client)|"
                     r"from\s+(socket|urllib|requests|httpx|aiohttp|http)\b|urlopen|create_connection|"
                     r"huggingface_hub|hf_hub_download|snapshot_download|torch\.hub|wandb\.init)")
_PY_CMD = re.compile(r"(?:^|[\s;&|(/])(python3?(?:\.\d+)?)(?=\s|$)")
#: env every researcher command gets on top of SAFE_ENV: libraries that respect proxies / offline flags stay offline
OFFLINE_ENV = {"http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9", "HTTP_PROXY":
               "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "ALL_PROXY": "http://127.0.0.1:9",
               "all_proxy": "http://127.0.0.1:9", "NO_PROXY": "", "no_proxy": "", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled",
               "PIP_NO_INDEX": "1", "PYTHONDONTWRITEBYTECODE": "1"}
_OUT_CAP = 12000
# bash commands that WRITE (review M1): a redirect / tee target, or an in-place / copy / move / patch tool -- refused
# when they name a locked path (best effort; the post-command check below restores whatever slips through)
_REDIRECT = re.compile(r"(?:>>?|>\|)\s*([^\s;&|<>]+)")
_TEE = re.compile(r"\btee\b((?:\s+[^\s;&|<>]+)+)")
_WRITER = re.compile(r"\bsed\b[^|;&]*\s-[a-zA-Z]*i|\bperl\b[^|;&]*\s-[a-zA-Z]*i|"
                     r"\b(cp|mv|ln|rm|rmdir|install|truncate|dd|patch|unlink|touch|chmod)\b|\bgit\s+(apply|mv|rm|am)\b")


def _tool(name: str, desc: str, required: tuple[str, ...] = (), **props: str) -> dict[str, Any]:
    return {"name": name, "description": desc, "input_schema": {
        "type": "object", "required": list(required), "properties": {k: {"type": v} for k, v in props.items()}}}


TOOL_DEFS = (
    _tool("read_file", "Read a file in the worktree (1-based line window).", ("path",),
          path="string", offset="integer", limit="integer"),
    _tool("list_dir", "List a directory in the worktree.", path="string"),
    _tool("edit_file", "Replace one exact occurrence of `old` with `new`.", ("path", "old", "new"),
          path="string", old="string", new="string"),
    _tool("write_file", "Create or overwrite an editable file.", ("path", "content"),
          path="string", content="string"),
    _tool("bash", "Run a shell command in the worktree (120 s cap). The project's python is on PATH (`python` / "
                  "`.venv/bin/python`): import-check or unit-test your edit before the smoke, e.g. `cd src && python -c "
                  "'import nanobrain.model'`. `cd` within the worktree and shell variables you set in the same command "
                  "work. Refused: paths outside the worktree, writes outside it, $(...) / backticks / environment "
                  "variables, git state changes, SLURM, network and uv / pip.", ("command",), command="string"),
    {"name": "set_config", "description": "Set one existing config key (dotted, e.g. dino.lr) to a "
                                          "YAML value; keeps formatting and comments.",
     "input_schema": {"type": "object", "required": ["key", "value"], "properties": {
         "key": {"type": "string"}, "value": {"description": "new value, e.g. 0.0001, true, [0.3, 1.0]"}}}},
    _tool("check", "Run the intake rules on your current diff: 'OK ...' or 'REJECT <rule>: <fix>'. Pass the "
                   "family, mechanism_tag and claim you will declare: call it once you know your mechanism (even "
                   "before editing) -- a dead / in-flight mechanism is reported with the nodes to name.",
          family="string", mechanism_tag="string", claim="string", falsification="string"),
    _tool("check_idea", "Cheap pre-edit check of a MECHANISM (no diff needed): runs intake's dead_mechanism / "
                        "near_dup_mechanism / repeat / in-flight / saturated-cluster rules on the family, mechanism_tag "
                        "and claim you would declare. Call it BEFORE editing: 'OK ...' or 'REJECT <rule>: <fix>'.",
          ("mechanism_tag",), family="string", mechanism_tag="string", claim="string"),
    _tool("smoke", "Short run on a data subset; call after your last edit."),
    _tool("revert", "Undo your edits in YOUR research worktree: `path` restores one file to the session's starting "
                    "point (a file you created is removed); no path restores every file. Use it to drop a change "
                    "before building another, instead of finishing with a diff you do not want.", path="string"),
)
from loop.agents import diagnose as DX  # noqa: E402
#: v3 sessions also get ``diagnose`` (loop.agents.diagnose): per-dataset / per-head classification, the incumbent's
#: feature diagnostic job, the data bank's manifests
from loop.agents import history as HX  # noqa: E402
#: v3-c9 C1: + ``history`` (loop.agents.history): the read-only trajectory dashboard, every brief kind (bold included)
V3_TOOL_DEFS = (DX.TOOL_DEF, HX.TOOL_DEF)
from loop.core.bold import SLATE_TOOL as BOLD_SLATE_TOOL  # noqa: E402  v3-c8b: diverge-then-judge's propose_slate
MAX_FINISH_BOUNCES = 2
#: a smoke job that never got a slot is retried this many more times in the researcher's thread; still none:
#: ``backend_unavailable`` with this error prefix (the driver re-issues the request; no node, no reward)
SMOKE_UNAVAILABLE_RETRIES = 1
#: a smoke the ENVIRONMENT killed (loop.infra: the gate is tripped) waits out the hold and tries again this many times
#: (each wait capped at INFRA_WAIT_MAX_S) before the session ends ``backend_unavailable`` and is re-issued
INFRA_SMOKE_WAITS = 3
INFRA_WAIT_MAX_S = 900.0
INFRA_CANARY_WAIT_S = 1800.0                    # at most this long polling behind another session's canary smoke
SMOKE_UNAVAILABLE = "smoke_unavailable"
#: sessions that get WebSearch / WebFetch: literature-grounded ideation (never a DEBUG fix or a tuning variant)
from loop.components import WEB_BRIEFS, WEB_STRATEGIES, web_eligible  # noqa: E402,F401
from loop.agents import notes as NT  # noqa: E402
WEB_TEXT = ("## Background research (web)\n"
            "WebSearch and WebFetch are available in this session. BEFORE you edit, do quick background research: "
            "2-5 searches (recent papers on the mechanism, method details, the hyperparameters they used and at what "
            "scale). Stop once you have what you need; the edit and the smoke are the job. Cite your sources (arXiv id "
            "or URL) in hypothesis.claim, and say in the handoff learnings what the literature suggested. State in "
            "hypothesis.prior_work what prior work it builds on and why that is not already in the recipe.\n")


SLATE_TEXT = ("## Plan before code: a slate of {lo}-{hi} DISTINCT ideas\n"
              "Research first, then -- BEFORE editing anything -- call `propose_ideas` with {lo}-{hi} ideas that differ "
              "in MECHANISM or COMPONENT (not doses of one idea: doses are variants). Each is a plan: title, "
              "mechanism_tag, family, files (or file::Class.method spans), config_keys (new ones included), init "
              "('unchanged' / 'zero_init' / 'changes_shapes' / 'new_init'), dose_knob, the first-principles record, predicted_sigmas, "
              "citations, variants and plan. The loop runs the intake rules on every plan (dead / cold / duplicate "
              "mechanism, family mapping, multi-family, protected keys, locked code, init-compat, the theme share "
              "guard), ranks the passing ones by its allocation signal, under-exploration and novelty, and names the "
              "ONE to implement now; the next {top} go to the idea bank as priority and the rest after them, with "
              "their plans -- your research is not lost. A rejected plan comes back with the rule and its fix: revise "
              "and propose again.\n")


def wants_web(request: CandidateRequest, ctx: Any = None) -> bool:
    """Literature-grounded ideation (``components.web_eligible``); with ``ctx`` also a v3 brief of a kind in
    ``ideation.web_kinds`` (repair / crossover: 10-13 09Z MA, web research had dropped out of them), bounded."""
    if "web" in request.ablated:
        return False
    return web_eligible(request.strategy, request.operator, request.brief_kind) or bounded_web(request, ctx)


def bounded_web(request: CandidateRequest, ctx: Any) -> bool:
    """A v3 repair / crossover session (``ideation.web_kinds``): web research capped at ``ideation.web_max_searches``."""
    if ctx is None or not is_v3(request) or request.operator is Operator.DEBUG:
        return False
    from loop.core.levers import opt
    return request.brief_kind in tuple(opt(ctx, "web_kinds") or ()) and not web_eligible(
        request.strategy, request.operator, request.brief_kind)


WEB_BOUNDED = ("## Background research (web, bounded)\n"
               "WebSearch and WebFetch are available, for at most {n} searches: look up the specific mechanism or the "
               "interaction you are resolving (method details, the dose used at what scale) only where the code and "
               "the brief do not answer it. The edit and the smoke are the job. Cite what you used (arXiv id or URL) "
               "in hypothesis.claim.\n")


_EVIDENCE_RX = re.compile(r"https?://\S+|\barxiv[:.\s]*\d{4}\.\d{4,5}|\b\d{4}\.\d{4,5}(v\d)?\b|doi\.org/|\b10\.\d{4,}/\S+"
                          r"|\brun_sub_[0-9a-f]{6,}\b", re.I)
_BOARD_ID = re.compile(r"\brun_sub_[0-9a-f]{6,}\b")
EVIDENCE_PROPS = {
    "evidence": {"type": "array", "items": {"type": "string"}, "description": (
        "REQUIRED for a structural change: what supports it -- a paper (URL / arXiv id / doi you actually read), a "
        "competition board run id (run_sub_...) from the board digest, or a reference-model fact from the digest "
        "('Walnut: ...'). One item per source, each with the one-line fact it gives")},
    "readout_path": {"type": "string", "description": (
        "REQUIRED for a structural change: how the change alters the FROZEN features the evaluation reads (which "
        "tasks / axis, through what property of the representation)")}}


def evidence_problem(hy: Mapping[str, Any], ctx: Any = None, store: Any = None) -> str:
    """'' when ``hypothesis.evidence`` holds at least one checkable source and ``readout_path`` ties the mechanism to
    the evaluation; else what is missing. A board run id must be one the board ingest stored; a reference-model fact
    must name a reference point of the digest."""
    ev = [str(x).strip() for x in (hy.get("evidence") or []) if str(x).strip()] \
        if isinstance(hy.get("evidence"), (list, tuple)) else ([str(hy["evidence"])] if hy.get("evidence") else [])
    labels: list[str] = []
    known: set[str] | None = None
    if ctx is not None and store is not None:
        from loop import board as BD
        if BD.enabled(ctx):
            d = BD.digest(store)
            labels = [str(r.get("label") or "") for r in d.get("refs") or [] if r.get("label")]
            known = {str(r.get("run_id")) for r in BD.stored_runs(store)}
    ok = []
    for e in ev:
        ids = _BOARD_ID.findall(e)
        if ids and known is not None and not all(i in known for i in ids):
            return f"hypothesis.evidence names board run(s) the board does not have: {[i for i in ids if i not in known]}"
        if _EVIDENCE_RX.search(e) or any(lab and lab.lower() in e.lower() for lab in labels):
            ok.append(e)
    miss = []
    if not ok:
        miss.append("hypothesis.evidence (no item cites a paper URL / arXiv id / doi, a board run id or a reference "
                    "model of the board digest" + (f": {', '.join(labels)}" if labels else "") + ")")
    if len(str(hy.get("readout_path") or "").split()) < 10:
        miss.append("hypothesis.readout_path (>= 10 words: how the change moves what the frozen-feature evaluation reads)")
    return "; ".join(miss)


def wants_research(request: CandidateRequest) -> bool:
    """A web session whose ideation must go through the RESEARCH STAGE (loop.agents.notes): moonshots and literature
    briefs."""
    return wants_web(request) and (request.strategy is Strategy.MOONSHOT or request.brief_kind in ("literature", "moonshot", "base"))
V3_SIZING = ("Size the change by the brief's per-axis budget (\"Effect size\"): mechanism_prediction.budget lists, per "
             "axis you move, its expected level gain and the bracketed row id it draws on; the panel claim is computed "
             "from it.")
DEFAULT_SIZING = ("Use \"What we've learned so far\": avoid mechanisms whose effect is confidently negative, and "
                  "scale\nyour predicted sizes by the stated prediction bias.")
MOONSHOT_SIZING = ("Use \"What we've learned so far\" as evidence for your mechanism, but IGNORE its advice to shrink "
                   "predicted\nsizes: this is a moonshot, so predict the effect you actually believe, per axis.")
BOLD_SIZING = ("Predict the ONE overall effect you actually believe (the chosen slate candidate's distribution and "
               "predictions.panel); there is no per-axis budget and no size floor, and the advice to shrink toward the "
               "past mean does not apply: an honest bold prediction that is falsified is useful evidence.")
BOLD_FP = ("## Think from first principles (required: `finish` bounces without this record)\n"
           "(a) MECHANISTIC MODEL -- what the incumbent's recipe does with the data at this fixed budget, and what it "
           "structurally never uses or cannot represent.\n"
           "(b) BOTTLENECK -- the ONE structural limit your change removes, and why it binds more than the others.\n"
           "(c) DERIVED CHANGE -- the structural change that follows (one taxonomy category).\n"
           "(d) MECHANISM PREDICTION -- `hypothesis.mechanism_prediction` = {{diagnostic, direction, axes}}: ONE "
           "measured quantity of the run diagnostics ({quantities}), its direction vs the parent on the same seed, and "
           "the axes you expect to move (e.g. {axes_example}). After the first finished seed the loop "
           "checks whether it moved.\n"
           "(e) PRIOR ATTEMPTS -- what in this loop or the literature came closest, and why this is different.\n"
           "Put them in `hypothesis.mechanistic_model`, `hypothesis.bottleneck`, `hypothesis.derivation`, "
           "`hypothesis.mechanism_prediction` and `hypothesis.prior_attempts`.\n\n")
OPERATOR_TEXT = {
    Operator.DRAFT: "DRAFT a new idea from the brief, starting from the parent's code.",
    Operator.IMPROVE: "IMPROVE the parent: keep its mechanism and make it work better (placement, a missing "
                      "piece, a structural fix; a dose change only as a substantiated MAJOR move), guided by its "
                      "handoff and posterior.",
    Operator.DEBUG: "DEBUG the parent: it crashed or voided. Read the failure below, fix the "
                    "implementation, keep the SAME mechanism and hypothesis. Debug depth {depth}/{cap}.",
    Operator.CROSSOVER: "CROSSOVER: the second parent ({other}) has been merged into this worktree "
                        "by git. Resolve it SEMANTICALLY so BOTH mechanisms work together; fix "
                        "conflicts and interactions, do not just pick one side. The diff is judged "
                        "against the parents' common ancestor, so it carries BOTH parents' changes; "
                        "register the levers of both.\nMerge result: {merge}",
}
#: a dose ladder's KNOB EXPOSURE (loop.tune): a pure refactor of the parent, never a new mechanism
EXPOSE_TEXT = ("EXPOSE the parent's hard-coded mechanism strength as ONE new config key (default = its current "
               "value), read it in the code, and declare it as hypothesis.dose_knob. Change NOTHING else: at the "
               "default the run must be numerically identical to the parent's. Smoke it; set "
               "hypothesis.identical_at_default true only when you verified that. Details in the brief.")
#: the LEG FACTORIAL's branch (loop.legs): every listed leg switchable by config, the carrier unchanged at the defaults
LEGS_TEXT = ("BUILD THE LEG BRANCH on the parent (the carrier): graft every listed leg behind its own 0/1 config flag "
             "(default 0) and expose the listed carrier legs' hard-coded strengths as multiplier keys (default 1), "
             "every key named ...leg_<node id>. Change NOTHING else: at the defaults the run must be numerically "
             "identical to the parent's. Smoke it with every flag on; set hypothesis.identical_at_default true only "
             "when you verified the defaults. Details in the brief.")


def diff_hash(diff_text: str) -> str:
    """sha256 of the diff without volatile header lines and trailing space."""
    keep = [ln.rstrip() for ln in diff_text.splitlines()
            if not ln.startswith(("index ", "diff --git", "@@"))]
    return hashlib.sha256("\n".join(l for l in keep if l).encode()).hexdigest()


def config_file(ctx: LoopContext, adapter: Any) -> str:
    """The task's config file (relative): the adapter's, else the first writable YAML."""
    rel = getattr(adapter, "config_rel", "") or getattr(adapter, "CONFIG_FILE", "")
    return rel or next((w for w in ctx.spec.surface.writable if w.endswith((".yaml", ".yml"))), "")


def _yaml_value(raw: Any, old: Any) -> Any:
    """The model's value as YAML data; '1e-4'-style numbers (a string in YAML 1.1) are floats."""
    v = yaml.safe_load(raw) if isinstance(raw, str) else raw
    if isinstance(v, str) and isinstance(old, (int, float)) and not isinstance(old, bool):
        try:
            v = float(v)
        except ValueError:
            pass
    return v


def set_yaml_value(text: str, key: str, value: Any) -> str:
    """``text`` with the scalar/flow value of dotted ``key`` replaced in place (comments and
    formatting kept). ValueError (one actionable line) when the key is not a one-line leaf."""
    lines, start, indent, hit = text.splitlines(keepends=True), 0, -1, -1
    end = len(lines)
    for part in key.split("."):
        rx, hit, block = re.compile(rf"^(\s*){re.escape(part)}\s*:(\s|$)"), -1, None
        for i in range(start, end):
            ln = lines[i]
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            ind = len(ln) - len(ln.lstrip())
            if ind <= indent:
                end = i
                break
            block = ind if block is None else block
            if ind == block and rx.match(ln):
                hit = i
                break
        if hit < 0:
            raise ValueError(f"key {key!r} not found in the config")
        start, indent = hit + 1, block
        end = next((j for j in range(start, end) if lines[j].strip() and not lines[j].lstrip().startswith("#")
                    and len(lines[j]) - len(lines[j].lstrip()) <= indent), end)
    m = re.match(r"^(\s*[^:#]+:[ \t]*)([^#\n]*?)([ \t]+#[^\n]*)?(\r?\n)?$", lines[hit])
    if m is None or not m.group(2).strip():
        raise ValueError(f"{key!r} is a block (mapping or multi-line list); set a leaf key or use edit_file")
    new = yaml.safe_dump(value, default_flow_style=True, width=1 << 16).strip()
    new = new[:-4].rstrip() if new.endswith("\n...") else new.removesuffix("...").rstrip()
    lines[hit] = f"{m.group(1)}{new}{m.group(3) or ''}{m.group(4) or ''}"
    return "".join(lines)


def claim_text(hy: Mapping[str, Any]) -> str:
    """The claim intake sees: + prior work, + ``differs_from`` (loop.core.ideation's anti-repeat retry)."""
    return str(hy.get("claim") or "") + "".join(f" [{label}: {x}]" for label, k in (
        ("prior work", "prior_work"), ("differs from", "differs_from"),
        ("differs from cluster", "differs_from_cluster")) if (x := str(hy.get(k) or "").strip()))


def _ids(xs: Any) -> tuple[str, ...]:
    """Node ids named in a handoff list (``a node``, also inside prose)."""
    return tuple(dict.fromkeys(m for x in (xs or ()) for m in re.findall(r"\bn\d{3,}\b", str(x))))


#: v3 (loop.core.ideation): the declared mechanism -- a diagnostics quantity, its direction, the axes it should move and
#: the expected composite size (logged for calibration only); engagement is checked after the first done seed
#: v3-c4i (09Z MA: claims shrank to .0025-.003 after the gap-sized deploy, anchored on a per-family EI table): the size
#: is a per-axis BUDGET -- each entry an axis, its expected level gain and the brief row (deficit / headroom) it draws
#: on; the panel claim (``size``) is computed from it with the official weights (:func:`budget_total`)
def budget_item() -> dict[str, Any]:
    """Evaluated at CALL time: the axis enum follows the installed project (loop.project), not the import."""
    return {"type": "object", "required": ["axis", "row", "gain"], "properties": {
        "axis": {"type": "string", "enum": list(AXES)}, "row": {"type": "string", "minLength": 1},
        "gain": {"type": "number"}}}


def mechanism_prediction() -> dict[str, Any]:
    return {"type": "object", "required": ["diagnostic", "direction", "axes", "budget"], "properties": {
        "diagnostic": {"type": "string", "minLength": 1}, "direction": {"type": "integer", "enum": [-1, 1]},
        "axes": {"type": "object", "additionalProperties": {"type": "integer", "enum": [-1, 0, 1]}},
        "budget": {"type": "array", "minItems": 1, "items": budget_item()},
        "size": {"type": "number"}}}


def axes_example() -> str:
    """The example axes object of the bold first-principles record: the target axis moves, a guard stays put."""
    other = P.guard() or next((a for a in AXES if a != P.target()), P.target())
    return json.dumps({P.target(): 1, other: 0})
#: a PROGRAM's staged plan (loop.core.program): its components, each a mechanism of one family with its doses, the
#: brief row it targets and its predicted share of the gain; ``off`` = the config values that switch it off (for the
#: leave-one-out), default the parent's values of its knobs
COMPONENT = {"type": "object", "required": ["family", "mechanism", "knobs", "target", "share"], "properties": {
    "family": {"type": "string", "minLength": 1}, "mechanism": {"type": "string", "minLength": 1},
    "knobs": {"type": "array", "minItems": 1, "items": FAC.KNOB_SCHEMA}, "target": {"type": "string", "minLength": 1},
    "share": {"type": "number", "exclusiveMinimum": 0}, "off": {"type": "object"}}}


from loop.core.calibration import FINISH_PROPS as CAL_FINISH  # noqa: E402
#: v3-c6i: a kuhn conjecture's analogy from a distant literature (2608.16951): the field, the cited source, the mapping
ANALOGY = {"type": "object", "properties": {"field": {"type": "string"}, "source": {"type": "string"},
                                            "mapping": {"type": "string"}}}


def analogy_problem(hy: Mapping[str, Any]) -> str:
    """'' when hypothesis.analogy names a field, a source (paper id / URL) and the mapping onto this setting."""
    a = hy.get("analogy") if isinstance(hy.get("analogy"), Mapping) else {}
    miss = [k for k in ("field", "source", "mapping") if len(str(a.get(k) or "").strip()) < 4]
    return (f"a paradigm-shift conjecture needs hypothesis.analogy from a DISTANT literature (not "
            f"{P.rule('analogy_field', 'pathology / SSL')}): "
            f"missing {', '.join(miss)} -- the field, the source you read (arXiv id or URL) and how its mechanism maps "
            "onto this recipe") if miss else ""


def budget_total(budget: Any) -> float | None:
    """The panel claim of a per-axis budget: sum of official weight x gain (None when malformed)."""
    if not isinstance(budget, (list, tuple)) or not budget:
        return None
    try:
        return sum(OFFICIAL_WEIGHTS[str(b["axis"])] * float(b["gain"]) for b in budget)
    except (KeyError, TypeError, ValueError):
        return None


def budget_problem(budget: Any, rows: tuple[str, ...]) -> str:
    """'' when every budget entry cites a row of the brief (and an axis row its own axis), else the fix. A brief
    without row ids (v3-c5: 10-13, a node cited 'classification-external-UNI-2-h-+0.0819', a node node ids, a node
    'guard_robustness_n2646' -- all unchecked) still needs a well-formed row id: the per-axis ``D-<axis>`` rows."""
    from loop.core.ideation import ROW_ID
    bad = []
    for b in budget or ():
        if not isinstance(b, Mapping):
            bad.append(f"{b!r} is not an {{axis, row, gain}} entry")
            continue
        row, axis = str(b.get("row") or "").strip("[] "), str(b.get("axis") or "")
        pre, _, rest = row.partition("-")
        if not rows and not (ROW_ID.fullmatch(f"[{row}]") and (pre != "D" or rest in AXES)):
            bad.append(f"{row!r} is not a row id")
        elif rows and row not in rows:
            bad.append(f"{row!r} is not a row of the brief")
        elif pre in ("D", "H", "X") and rest in AXES and rest != axis:
            bad.append(f"{row} is the {rest} row, not {axis}")
        elif pre == "XD" and axis != P.target():
            bad.append(f"{row} is a classification dataset row, not {axis}")
    return "; ".join(bad)


def _bank_block(ctx: LoopContext, store: Any) -> str:
    """The v3 context pack's data bank: every registered dataset's manifest line (loop.databank), and how to use one
    or ask for another ('' when the bank is off or empty)."""
    lines = DB.bank_lines(ctx, store) if hasattr(store, "get_state") else []
    if not lines:
        return ""
    return (P.rule("databank_pack_header", "\n### Data bank (data_curation: set data.dataset_dir to one; TCGA tile "
                   "datasets in the training Arrow format, read from their manifests)\n") + "\n".join(f"- {x}" for x in lines)
            + "\nA new one: declare dataset_spec in the finish (the loop builds it; your node waits until it is "
              "ready).")


def is_program_arm(request: CandidateRequest) -> bool:
    """A per-arm operator-program session (``program_arm:<program>:<arm>``): run on purpose, so the similarity dedups
    (near_dup, signature, repeat_of_negative, mechanism, reserved_inflight) never apply to it."""
    return str(getattr(request, "arm", "") or "").startswith("program_arm:")


def is_v3(request: CandidateRequest) -> bool:
    """A brief from loop.core.ideation (its request ids carry ``REQUEST_PREFIX``; brief kinds overlap the old ones)."""
    from loop.core.ideation import REQUEST_PREFIX
    return request.id.startswith(REQUEST_PREFIX)


def unknown_diagnostic(hy: Mapping[str, Any]) -> str:
    """'' when hypothesis.mechanism_prediction names a diagnostics quantity (or a raw feature key), else the fix."""
    from loop import diagnostics as DG
    name = str((hy.get("mechanism_prediction") or {}).get("diagnostic") or "").strip()
    if name in DG.quantities() or re.fullmatch(r"b\d+_[a-z0-9_]+", name):
        return ""
    return (f"REJECT mechanism_prediction: diagnostic {name!r} is not a measured quantity; name one of "
            f"{', '.join(sorted(DG.quantities()))} (the loop checks after the first seed whether it moved)")


#: v3-c8b: a bold session's mechanism prediction -- ONE overall distribution (the slate's), no per-axis budget rows
def bold_prediction() -> dict[str, Any]:
    return {**mechanism_prediction(), "required": ["diagnostic", "direction", "axes"]}


def __getattr__(name: str) -> Any:          # the old import-time constants, now read at call time
    if name == "BUDGET_ITEM":
        return budget_item()
    if name == "MECHANISM_PREDICTION":
        return mechanism_prediction()
    if name == "BOLD_PREDICTION":
        return bold_prediction()
    raise AttributeError(name)


def finish_schema(ctx: LoopContext, v3: bool = False, bold: bool = False) -> dict[str, Any]:
    text = {"type": "string", "minLength": 1}
    pred = {"type": "object", "required": ["sign", "size"], "properties": {
        "sign": {"type": "integer", "enum": [-1, 1]}, "size": {"type": "number", "exclusiveMinimum": 0}}}
    return {"type": "object", "required": ["hypothesis", "levers", "handoff"], "properties": {
        "hypothesis": {"type": "object", "required": ["claim", "mechanism_tag", "family", "predictions", "falsification",
                                                      *(["mechanism_prediction"] if v3 else [])],
                       "properties": {
                           "claim": text, "mechanism_tag": text, "falsification": text,
                           "mechanism_type": {"type": "string", "enum": list(NV.TYPES)}, "prior_work": {"type": "string"},
                           "builds_on": {"type": "string"},
                           "dose_knob": {"type": "string"}, "identical_at_default": {"type": "boolean"},
                           **{f: {"type": "string"} for f in IDEAS.FP_FIELDS},
                           "mechanism_prediction": bold_prediction() if bold else mechanism_prediction() if v3
                           else IDEAS.PREDICTION_SCHEMA,
                           "differs_from": {"type": "string"},
                           "differs_from_cluster": {"type": "string", "description": (
                               "REQUIRED when check_idea says negative_family: '<node>: how it differs and why that "
                               "failure does not apply; <node>: ...' naming >= 2 measured nodes of the negative cluster")},
                           **({"analogy": ANALOGY} if v3 else {}),
                           **(EVIDENCE_PROPS if v3 and getattr(getattr(ctx.cfg, "ideation", None), "require_evidence",
                                                               False) else {}),
                           "idea_id": {"type": "string"}, "component": {"type": "string"},
                           "family": {"type": "string", "enum": sorted(ctx.cfg.families)},
                           "predictions": {"type": "object", "required": [PANEL],
                                           "additionalProperties": False,
                                           "properties": {t: pred for t in (PANEL, *ctx.cfg.axes)}}}},
        "levers": {"type": "object", "additionalProperties": False,
                   "properties": {l: {"type": "integer", "enum": [-1, 0, 1]}
                                  for l in ctx.cfg.levers.vocabulary}},
        "handoff": {"type": "object", "required": ["actions", "learnings", "concerns"],
                    "properties": {**{k: {"type": "string"} for k in ("actions", "learnings", "concerns")},
                                   **{k: {"type": "array", "items": {"type": "string"}}
                                      for k in ("stacks_with", "conflicts_with")}}},
        "commit_message": {"type": "string"},
        "variants": {"type": "array", "items": IDEAS.VARIANT_SCHEMA},
        "plan": {"type": "array", "items": IDEAS.PLAN_SCHEMA},
        "knobs": {"type": "array", "items": FAC.KNOB_SCHEMA},     # loop.core.factorial: a wave over them
        **({"components": {"type": "array", "items": COMPONENT}} if v3 else {}),   # a program's staged plan
        **(CAL_FINISH if v3 else {}),            # v3-c6i: the verbalized-sampling slate (loop.core.calibration)
        "campaign": IDEAS.CAMPAIGN_SCHEMA, "search_space": IDEAS.SPACE_SCHEMA,
        **({"dataset_spec": DB.SPEC_SCHEMA} if getattr(getattr(ctx.cfg, "databank", None), "enabled", False) else {})}}


class WorktreeTools:
    """The agent's executor ``tools(name, args) -> str``, scoped to ``worktree.path``."""

    def __init__(self, adapter: Any, worktree: Worktree, *, writable: tuple[str, ...],
                 locked: tuple[str, ...], readonly_roots: tuple[str, ...] = (),
                 bash_timeout_s: float = 120.0, config_rel: str = "",
                 protected: tuple[str, ...] = (), checker: Callable[..., str] | None = None,
                 python: str = "") -> None:
        self.adapter, self.wt, self.root = adapter, worktree, Path(worktree.path).resolve()
        # 11-08: the project interpreter (loop.project.python; the worktree's .venv links to its venv) may run here
        self.venv = self._venv_dir(python)
        self.pythons = self._pythons(python)
        self.writable, self.locked = tuple(writable), tuple(locked)
        self.readonly_roots = tuple(os.path.realpath(r) for r in readonly_roots)
        self.bash_timeout_s, self.dirty, self.last_smoke = bash_timeout_s, False, None
        self.config_rel, self.protected, self.checker = config_rel, tuple(protected), checker
        self.bounces = 0
        self.need_prior_work = False            # a web session must name its prior work (loop.novelty)
        self.need_stacks: tuple[str, ...] = ()  # the incumbent's legs: say which it stacks / conflicts with
        self.need_fp = False                    # an ideation session: the first-principles record (loop.agents.ideas)
        self.planner: Callable[[list], tuple[int | None, str]] | None = None   # a slate session's propose_ideas
        self.slate: list[dict] = []             # the proposed ideas ...
        self.chosen: int | None = None          # ... and the one the loop picked to implement
        self.research: NT.ResearchStage | None = None   # moonshot / literature: the research stage's ledger
        self.v3 = False                         # a loop.core.ideation brief: v3 mechanism prediction, differs_from
        self.diagnoser: Callable[..., str] | None = None   # v3: the diagnose tool (loop.agents.diagnose)
        self.historian: Callable[..., str] | None = None   # v3-c9: the history tool (loop.agents.history)
        self.claim_floor, self.rows = 0.0, ()   # v3 target mode: the budget floor, the brief's row ids
        self.program = False                    # a program brief: the finish declares its components
        self.size_bounces = 0                   # budget-below-floor bounces so far (ideation.claim_bounces)
        self.size_cap = 1
        self.min_components = 2
        self.below_floor = False                # the final budget stayed below the floor (logged at intake)
        self.final_reject = ""                  # the bounce limit was reached on a hard REJECT: the session ends rejected
        # v3-c5: the budget's weighted sum vs the panel claim (ideation.claim_budget_tol / claim_mismatch_bounces)
        self.mismatch_tol, self.mismatch_cap, self.mismatch_bounces = 0.2, 1, 0
        self.claim_mismatch: dict[str, Any] | None = None   # accepted despite a mismatch (logged; rides on the plan)
        self.logger: Callable[..., Any] | None = None       # store.log
        # v3-c6i: the verbalized-sampling slate (loop.core.calibration) and the kuhn brief's distant analogy
        self.vs_k, self.ref_gap, self.vs_cap, self.vs_bounces = 0, 0.0, 1, 0
        self.vs_problem = ""                    # the slate stayed malformed after its bounces (logged on the plan)
        self.need_analogy = False
        self.analogy_bounced = False
        # v3-c8b: diverge-then-judge (loop.core.bold.Slate: propose_slate BEFORE any edit) and the bold lane
        self.slate_gate: Any = None
        self.slate_def: Any = None              # v3-c9 C3: this session's propose_slate tool definition
        self.bold = False
        # 11-08 evidence gate (``ideation.require_evidence``): own bounce counter, then the gap is logged and parked
        self.need_evidence, self.evidence_cap, self.evidence_bounces, self.evidence_missing = False, 2, 0, ""
        self.evidence_check: Callable[[Mapping[str, Any]], str] = lambda hy: evidence_problem(hy)
        # 11-08 the mechanism memory (loop.core.tested): check_idea's tested / in-flight repeat rule
        self.memory_check: Callable[..., str] | None = None

    def on_web(self, name: str, args: Mapping[str, Any] | None) -> None:
        if self.research is not None:
            self.research.on_web(name, args)

    #: v3-c7i I7: the names / arguments a model trained on other tool sets reaches for, mapped onto this set
    ALIASES = {"Read": "read_file", "read": "read_file", "cat": "read_file", "view": "read_file",
               "Bash": "bash", "shell": "bash", "Edit": "edit_file", "Write": "write_file", "ls": "list_dir",
               "LS": "list_dir"}
    ARG_ALIASES = {"file_path": "path", "filename": "path", "old_string": "old", "new_string": "new",
                   "cmd": "command"}

    def __call__(self, name: str, args: Mapping[str, Any]) -> str:
        if name not in ("notes", "note") and not hasattr(self, f"t_{name}") and name in self.ALIASES:
            name = self.ALIASES[name]
        args = {self.ARG_ALIASES.get(k, k): x for k, x in dict(args or {}).items()}
        fn = getattr(self.research, f"t_{name}", None) if name in ("notes", "note") and self.research else \
            getattr(self, f"t_{name}", None)
        if fn is not None and (why := self.slate_refusal(name, args)):
            return why
        if fn is None:
            return f"ERROR: unknown tool {name!r}; use one of {[t['name'] for t in TOOL_DEFS]} or finish"
        try:
            out = fn(**dict(args))
        except TypeError as e:
            return f"ERROR: bad arguments for {name} ({e}); see the tool's parameters and retry"
        if name == "note" and isinstance(out, str) and out.startswith("noted"):
            out += self.note_check(args)
        return out

    def slate_refusal(self, name: str, args: Mapping[str, Any]) -> str:
        """v3-c8b: '' unless this session must propose its slate first and ``name`` would WRITE (an edit, a config
        set, a writing bash command) -- reading, research and propose_slate itself are always allowed."""
        if self.slate_gate is None or self.slate_gate.done:
            return ""
        writes = name in ("edit_file", "write_file", "set_config")
        if name == "bash":
            cmd = str(args.get("command") or "")
            targets = [t for t in _REDIRECT.findall(cmd) if t not in ("/dev/null",)]
            writes = bool(targets or _TEE.search(cmd) or _WRITER.search(cmd))
        if not writes:
            return ""
        return ("REFUSED: diverge first -- call propose_slate with your candidate ideas (see the brief); edits are "
                "unlocked once the loop has ranked them and named the one to build")

    def t_propose_slate(self, candidates: Any = None, **_ignored: Any) -> str:
        if self.slate_gate is None:
            return "ERROR: propose_slate is not part of this session"
        return self.slate_gate(candidates)

    def note_check(self, args: Mapping[str, Any]) -> str:
        """v3-c5 I3: the mechanism rules (dead / near-duplicate / repeat-of-negative / in-flight) on a noted paper's
        mechanism, at NOTE time -- advisory, nothing claimed -- so a session learns before it builds (10-13: 40% of
        turns went to intake rejects of dead / near-dup mechanisms only seen at finish). '' when it passes."""
        if self.checker is None:
            return ""
        claim = " ".join(str(args.get(k) or "") for k in ("title", "claim", "mechanism")).strip()
        try:
            got = self.checker(family="", claim=claim, mechanism_tag=str(args.get("topic") or ""),
                               mechanism_only=True)
        except Exception:  # noqa: BLE001 - a check fault never loses the note
            return ""
        if not str(got).startswith("REJECT"):
            return ""
        return (f"\nHEADS-UP (mechanism check of this note, advisory): building this mechanism as-is would be "
                f"{got[:900]}")

    def check_finish(self, args: Mapping[str, Any]) -> str:
        """ToolGate hook: '' lets ``finish`` through; a message bounces it (at most MAX_FINISH_BOUNCES)."""
        if self.research is not None and (gap := self.research.missing(args)):      # bounces once, on its own
            return f"{gap} (research bounce: fix it, then call finish again)"
        hy, ho, out = args.get("hypothesis") or {}, args.get("handoff") or {}, ""
        if self.bounces >= MAX_FINISH_BOUNCES:
            # the bounce limit: the finish goes through, but a diff the intake rules still REJECT ends the session
            # REJECTED (no smoke, no commit; 10-13: req-v3-00079's multi-family diff was smoked, committed and only
            # then rejected at intake)
            try:
                last = self.checker(family=str(hy.get("family") or ""), claim=claim_text(hy),
                                    mechanism_tag=str(hy.get("mechanism_tag") or ""),
                                    falsification=str(hy.get("falsification") or ""),
                                    handoff={k: str(ho.get(k) or "") for k in ("actions", "learnings", "concerns")},
                                    final=True, dataset_only=isinstance(args.get("dataset_spec"), Mapping),
                                    **self._mech_kw(args)) \
                    if self.checker else ""
            except Exception:  # noqa: BLE001 - a checker fault never blocks finish; intake re-checks
                last = ""
            if last.startswith("REJECT"):
                self.final_reject = last[len("REJECT"):].strip()
            elif self.slate_gate is not None and self.slate_gate.done:
                self.slate_gate.note_finish(hy)                    # review fix M3: logged, never a rejection of the build
            if self.v3 and self.need_evidence and not self.final_reject and (why := self.evidence_check(hy)):
                self.evidence_missing = why
            if self.v3 and not self.final_reject:
                if self.vs_k > 0 and not self.vs_problem:
                    from loop.core import calibration as CAL
                    self.vs_problem = CAL.problem(args, self.vs_k, self.ref_gap)
                self.below_floor = self._below_floor(hy)
                if self.claim_mismatch is None and (mm := self._claim_mismatch(hy)):
                    self._log_mismatch(mm)
            return ""
        try:                                   # the finish record intake will see: tag, claim, falsification, handoff
            out = self.checker(family=str(hy.get("family") or ""), claim=claim_text(hy),
                mechanism_tag=str(hy.get("mechanism_tag") or ""), falsification=str(hy.get("falsification") or ""),
                handoff={k: str(ho.get(k) or "") for k in ("actions", "learnings", "concerns")},
                final=True, dataset_only=isinstance(args.get("dataset_spec"), Mapping),
                declared_knobs=bool(args.get("components") or args.get("knobs")), **self._mech_kw(args)) \
                if self.checker else ""
        except Exception:  # noqa: BLE001 — a checker fault never blocks finish; intake re-checks
            out = ""
        if not out.startswith("REJECT") and self.need_prior_work and not str(hy.get("prior_work") or "").strip():
            out = ("REJECT prior_work: state in hypothesis.prior_work what prior work this builds on (cite it) "
                   "and why it is not already in the recipe")
        if not out.startswith("REJECT") and self.slate_gate is not None and not self.slate_gate.done:
            out = ("REJECT slate_first: call propose_slate with your candidate ideas first; the loop scores them "
                   "(originality x plausibility) and names the ONE to build")
        if not out.startswith("REJECT") and self.slate_gate is not None and self.slate_gate.done:
            self.slate_gate.note_finish(hy)                        # review fix M3: logged (portfolio_cites), never a REJECT
        if not out.startswith("REJECT") and self.planner is not None and self.chosen is None:
            out = ("REJECT plan_first: call propose_ideas with your DISTINCT idea plans first; the loop gates and ranks "
                   "them and names the ONE to implement")
        if not out.startswith("REJECT") and self.chosen is not None \
                and str(hy.get("idea_id") or "") != f"gen-{self.request_id}-{self.chosen + 1}":
            out = (f"REJECT idea_id: you implemented the slate's chosen idea; declare hypothesis.idea_id "
                   f"\"gen-{self.request_id}-{self.chosen + 1}\"")
        if not out.startswith("REJECT") and self.v3:       # every v3 session names a measurable diagnostic
            miss = [f for f in ("mechanistic_model", "bottleneck", "prior_attempts")
                    if self.need_fp and not str(hy.get(f) or "").strip()]
            out = (f"REJECT first_principles: state {', '.join(miss)} (see 'Think from first principles')" if miss
                   else unknown_diagnostic(hy))
            budget = (hy.get("mechanism_prediction") or {}).get("budget")
            if not out.startswith("REJECT") and not self.bold and (why := budget_problem(budget, self.rows)):
                out = (f"REJECT budget: {why}. Each mechanism_prediction.budget entry's `row` is the bracketed id of "
                       "the brief row it draws on, copied without the brackets -- not free text, a node id or a "
                       f"number. Valid ids: {self._valid_rows()}")
            if not out.startswith("REJECT") and not self.bold and (mm := self._claim_mismatch(hy)):
                if self.mismatch_bounces < self.mismatch_cap:   # bounced ONCE (own counter), then accepted + logged
                    self.mismatch_bounces += 1
                    return mm["text"]
                self._log_mismatch(mm)
            if not out.startswith("REJECT") and self.program and (why := self._components_problem(args)):
                out = f"REJECT components: {why}"
            if not out.startswith("REJECT") and (why := self._keys_problem(args)):    # v3-c7i I8
                out = f"REJECT knob_keys: {why}"
            if not out.startswith("REJECT") and self.vs_k > 0:   # v3-c6i: the candidate slate (own counter)
                from loop.core import calibration as CAL
                if why := CAL.problem(args, self.vs_k, self.ref_gap):
                    if self.vs_bounces < self.vs_cap:
                        self.vs_bounces += 1
                        return (f"candidate slate: {why}. Fix the slate (see the brief's 'Candidates, then one'), then "
                                "call finish again.")
                    self.vs_problem = why
            if not out.startswith("REJECT") and self.need_analogy and not self.analogy_bounced \
                    and (why := analogy_problem(hy)):      # v3-c6i: a kuhn conjecture cites a distant field, once
                self.analogy_bounced = True
                return f"analogy: {why}. Add it, then call finish again."
            if not out.startswith("REJECT") and not self.bold and self._below_floor(hy):
                if self.size_bounces < self.size_cap:  # bounced ONCE (own counter), then logged as is
                    self.size_bounces += 1
                    total = budget_total(budget) or 0.0
                    return (f"budget below the target share: your per-axis budget totals {total:+.4f} of the panel, "
                            f"under {self.claim_floor:.4f} (the share of the gap this session is asked for). Make the "
                            "change bigger -- a stronger dose of the same mechanism, a second component aimed at "
                            "another deficit row, or a staged program -- then call finish again. If you believe the "
                            "smaller size, call finish again unchanged: it is logged as below the floor.")
                self.below_floor = True
        if not out.startswith("REJECT") and self.need_fp and not self.v3 and (miss := IDEAS.missing_fp(hy)):
            out = (f"REJECT first_principles: state {', '.join(miss)} -- the mechanistic model of the parent, the binding "
                   "constraint, the measured quantity your change should move (and where it is read) and why earlier "
                   "attempts on this part failed. A change with no stated path from the bottleneck to the panel is "
                   "not an idea")
        if not out.startswith("REJECT") and self.need_stacks and not ("stacks_with" in ho or "conflicts_with" in ho):
            out = ("REJECT stacking: state in handoff.stacks_with / handoff.conflicts_with which incumbent legs "
                   f"({', '.join(self.need_stacks)}) your change is expected to stack with or conflict with "
                   "(empty lists when neither)")
        if not out.startswith("REJECT") and self.v3 and self.need_evidence and (why := self.evidence_check(hy)):
            if self.evidence_bounces < self.evidence_cap:        # own counter: a structural idea needs its evidence
                self.evidence_bounces += 1
                return (f"evidence ({self.evidence_bounces}/{self.evidence_cap}): missing {why}. Research it first -- "
                        "WebSearch / WebFetch the mechanism (read the method section), or cite the board digest's run "
                        "ids / reference-model facts -- then call finish again with hypothesis.evidence and "
                        "hypothesis.readout_path. Without them the idea is parked for a literature session, not built.")
            self.evidence_missing = why
        if not out.startswith("REJECT"):
            return ""
        self.bounces += 1
        return f"{out} (bounce {self.bounces}/{MAX_FINISH_BOUNCES}: fix it, then call finish again)"

    request_id = ""

    def _mech_kw(self, args: Mapping[str, Any]) -> dict[str, str]:
        """v3-c6i: the chosen candidate's one-line mechanism (what intake's signature dedup compares), when given."""
        if self.vs_k <= 0:
            return {}
        from loop.core import calibration as CAL
        c = CAL.chosen(args) or {}
        return {"mechanism": str(c.get("mechanism") or "")} if c.get("mechanism") else {}

    def _valid_rows(self) -> str:
        if self.rows:
            return ", ".join(self.rows[:40]) + (f" ... ({len(self.rows)} in all)" if len(self.rows) > 40 else "")
        return ("this brief shows no row ids: cite the per-axis deficit row of the axis you move, "
                + ", ".join(f"D-{a}" for a in AXES))

    def _claim_mismatch(self, hy: Mapping[str, Any]) -> dict[str, Any] | None:
        """The budget's weighted sum vs the session's panel claim (``predictions.panel`` sign x size, else
        ``mechanism_prediction.size``): None within ``ideation.claim_budget_tol`` (relative) or when either is
        missing, else {total, claim, rel, text}."""
        mp = hy.get("mechanism_prediction") or {}
        total = budget_total(mp.get("budget"))
        pan = (hy.get("predictions") or {}).get(PANEL)
        try:
            claim = int(pan["sign"]) * float(pan["size"]) if isinstance(pan, Mapping) else float(mp["size"])
        except (KeyError, TypeError, ValueError):
            return None
        if total is None or claim == 0 or self.mismatch_tol <= 0:
            return None
        rel = abs(total - claim) / abs(claim)
        if rel <= self.mismatch_tol:
            return None
        terms = " + ".join(f"{OFFICIAL_WEIGHTS[str(b['axis'])]:g} x {float(b['gain']):+.4f} ({b['axis']})"
                           for b in mp.get("budget") or ())
        return {"total": round(total, 6), "claim": round(claim, 6), "rel": round(rel, 4), "text": (
            f"budget and panel claim disagree: your per-axis budget weighs to {terms} = {total:+.4f} of the official "
            f"panel, but predictions.{PANEL} claims {claim:+.4f} ({rel:.0%} apart; tolerance "
            f"{self.mismatch_tol:.0%}). Each budget gain is the AXIS level gain (not already weighted); the panel "
            "claim is their weighted sum. Fix whichever is wrong so they agree, then call finish again. If both are "
            "what you mean, call finish again unchanged: it is accepted and the mismatch is logged.")}

    def _log_mismatch(self, mm: Mapping[str, Any]) -> None:
        self.claim_mismatch = {k: mm[k] for k in ("total", "claim", "rel")}
        if self.logger is not None:
            try:
                self.logger("claim_mismatch", request_id=self.request_id, **self.claim_mismatch)
            except Exception:  # noqa: BLE001 - a log fault never blocks finish
                pass

    def _below_floor(self, hy: Mapping[str, Any]) -> bool:
        total = budget_total((hy.get("mechanism_prediction") or {}).get("budget"))
        return bool(self.claim_floor > 0 and total is not None and total < self.claim_floor)

    def _keys_problem(self, args: dict[str, Any]) -> str:
        """v3-c7i I8: the finish's knob keys (``knobs``, components' ``knobs`` / ``off``) as full dotted keys of the
        worktree's config: a bare key with ONE suffix match is rewritten in place; an ambiguous / unknown one is the
        problem ('' = fine)."""
        if not (args.get("knobs") or args.get("components")) or not self.config_rel:
            return ""
        try:
            cfg = flatten(yaml.safe_load(self._resolve(self.config_rel)[0].read_text()) or {})
        except Exception:  # noqa: BLE001 - unreadable here: intake resolves against the commit
            return ""
        from loop.core import program as PG
        fixed, problems = PG.resolve_plan_keys({k: args[k] for k in ("knobs", "components") if k in args}, cfg)
        args.update(fixed)
        return "; ".join(problems[:6])

    def _components_problem(self, args: Mapping[str, Any]) -> str:
        comps = [c for c in args.get("components") or () if isinstance(c, Mapping)]
        if len(comps) < self.min_components:
            return (f"a program declares >= {self.min_components} components in `components` (each: family, "
                    "mechanism, knobs [{key, lo, hi, scale}], target = the brief row id it closes, share of the gain; "
                    "optional off = {key: value} switching it off): the loop doses each and drops each in turn")
        bad = [str(c.get("target")) for c in comps if self.rows and str(c.get("target") or "").strip("[] ")
               not in self.rows]
        return f"component target(s) {bad} are not rows of the brief" if bad else ""

    def t_revert(self, path: str = "") -> str:
        """Restore ``path`` (or every changed file) to the session's starting point (the worktree's HEAD: the parent,
        or a crossover's merge); a file the session created is removed. Only this research worktree is touched."""
        root = self.root
        repo = os.path.realpath(str(getattr(self.adapter, "repo", "") or ""))
        if not (root / ".git").is_file() or (repo and os.path.realpath(root) == repo):
            return "REFUSED: revert works only inside the session's own research worktree"

        def git(*a: str) -> tuple[int, str]:
            cp = subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, timeout=120)
            return cp.returncode, cp.stdout
        if path:
            _, rel = self._resolve(path)
            targets = [rel]
        else:
            _, changed = git("diff", "--name-only", "HEAD", "--", ".", ":(exclude).venv")
            _, staged = git("diff", "--cached", "--name-only", "HEAD", "--", ".", ":(exclude).venv")
            _, new = git("ls-files", "--others", "--exclude-standard", "--", ".", ":(exclude).venv")
            targets = sorted({x for x in (changed + staged + new).splitlines() if x and x != ".venv"})
        done = []
        for rel in targets:
            if any(path_matches(rel, pat) for pat in self.locked):
                continue
            rc, _ = git("cat-file", "-e", f"HEAD:{rel}")
            if rc == 0:
                rc, _ = git("checkout", "HEAD", "--", rel)     # index + file back to HEAD (this worktree only)
                done.append(rel) if rc == 0 else None
            elif (p := (root / rel)).is_file():
                git("rm", "-q", "--cached", "--ignore-unmatch", "--", rel)
                p.unlink()
                done.append(rel)
        self.dirty, self.last_smoke = True, None
        return f"reverted {len(done)} file(s): {', '.join(done) or 'nothing to revert'}"

    def t_diagnose(self, what: str = "", name: str = "") -> str:
        if self.diagnoser is None:
            return "ERROR: diagnose is not available in this session"
        return self.diagnoser(what, name)

    def t_history(self, action: str = "table", filter: Any = None, sort: str = "", limit: int = 0, node: str = "",
                  part: str = "diff", offset: int = 0) -> str:
        if self.historian is None:
            return "ERROR: history is not available in this session"
        return self.historian(action, filter, sort, limit, node, part, offset)

    def t_propose_ideas(self, ideas: Any = None) -> str:
        if self.planner is None:
            return "ERROR: this session implements one given idea; propose_ideas is not available"
        if self.chosen is not None:
            return (f"ERROR: the slate is settled: implement idea #{self.chosen + 1} "
                    f"(idea_id gen-{self.request_id}-{self.chosen + 1})")
        got, text = self.planner(list(ideas or []))
        if got is not None:
            self.chosen, self.slate = got, list(ideas or [])
        return text

    def t_check(self, family: str = "", claim: str = "", mechanism_tag: str = "", falsification: str = "") -> str:
        return self.checker(family=family, claim=claim, mechanism_tag=mechanism_tag, falsification=falsification) \
            if self.checker else "OK (no intake rules configured)"

    def t_check_idea(self, mechanism_tag: str = "", family: str = "", claim: str = "", **_ignored: Any) -> str:
        """v3-c7i I4: the intake's MECHANISM rules on a mechanism before any edit (dead / near-duplicate / repeat of
        a negative node / another in-flight session's claim / a saturated-negative cluster past its cap), one turn."""
        if not (mechanism_tag.strip() or claim.strip()):
            return "ERROR: pass mechanism_tag (and family, claim) of the mechanism you would build"
        out = "OK"
        if self.checker is not None:
            try:
                out = str(self.checker(family=family, claim=claim, mechanism_tag=mechanism_tag, mechanism_only=True))
            except Exception as exc:  # noqa: BLE001 - a check fault never blocks the session
                out = f"OK (check unavailable: {type(exc).__name__})"
        if not out.startswith("REJECT") and self.memory_check is not None:
            try:
                out = self.memory_check(tag=mechanism_tag, family=family, claim=claim) or out
            except Exception:  # noqa: BLE001 - the memory never blocks the session
                pass
        if self.logger is not None:
            try:
                self.logger("check_idea", request_id=self.request_id, tag=mechanism_tag[:120],
                            verdict=out.split(":", 1)[0][:80])
            except Exception:  # noqa: BLE001
                pass
        if out.startswith("REJECT"):
            return out + " -- nothing was built yet: pick another mechanism (or answer the named nodes) and check again"
        return (f"{out}: mechanism {mechanism_tag.strip() or '(claim)'} passes the mechanism rules. Now edit, then "
                "check, smoke, finish.")

    def t_set_config(self, key: str, value: Any) -> str:
        if not self.config_rel:
            return "ERROR: this task has no config file; use edit_file"
        if bad := next((pk for pk in self.protected if key == pk or key.startswith(pk + ".")), ""):
            return f"REFUSED: {key} is protected ({bad}: budgets and eval identity are fixed); choose another key"
        p, _ = self._resolve(self.config_rel)
        text = p.read_text()
        cur = flatten(yaml.safe_load(text) or {})
        if key not in cur:
            near = difflib.get_close_matches(key, list(cur), n=3, cutoff=0.5)
            return f"ERROR: unknown key {key!r}{f'; did you mean {near}' if near else ''}. Keys must already exist"
        new = _yaml_value(value, cur[key])
        old = cur[key]
        if isinstance(old, (int, float)) and not isinstance(old, bool) and \
                (not isinstance(new, (int, float)) or isinstance(new, bool)):
            return f"ERROR: {key} is a number (now {old!r}); pass a number"
        try:
            out = set_yaml_value(text, key, new)
        except ValueError as e:
            return f"ERROR: {e}"
        after = flatten(yaml.safe_load(out) or {})
        if after.get(key) != new or {k: v for k, v in after.items() if k != key} != \
                {k: v for k, v in cur.items() if k != key}:
            return f"ERROR: could not set {key} cleanly as YAML; use edit_file on {self.config_rel}"
        res = self._write(self.config_rel, lambda path, rel: path.write_text(out) and "")
        if res.startswith("REFUSED"):
            return res
        return f"set {key}: {old!r} -> {new!r}. Next: check, then smoke."

    def _resolve(self, rel: str) -> tuple[Path, str]:
        p = (self.root / (rel or ".")).resolve()
        if p != self.root and self.root not in p.parents:
            raise PermissionError(f"{rel!r} is outside the worktree")
        if _DENY.search(rel):
            raise PermissionError(f"{rel!r} is blocked (v1 state / run artifacts)")
        return p, str(p.relative_to(self.root)) if p != self.root else "."

    def _write(self, path: str, fn: Any) -> str:
        p, rel = self._resolve(path)
        if any(path_matches(rel, pat) for pat in self.locked):
            return f"REFUSED: {rel} is LOCKED (surface.locked); it may never be edited"
        if not any(path_matches(rel, pat) for pat in self.writable):
            return f"REFUSED: {rel} is not in surface.writable {list(self.writable)}"
        out = fn(p, rel)
        self.dirty = self.dirty or not out.startswith("ERROR")
        return out

    def t_read_file(self, path: str, offset: int = 1, limit: int = 400, **_ignored: Any) -> str:
        lines = self._resolve(path)[0].read_text(errors="replace").splitlines()
        start = max(1, int(offset))
        chunk = lines[start - 1:start - 1 + max(1, int(limit))]
        more = f"\n... ({len(lines)} lines total)" if start - 1 + len(chunk) < len(lines) else ""
        return "\n".join(f"{i:6d}\t{ln}" for i, ln in enumerate(chunk, start))[:_OUT_CAP] + more

    def t_list_dir(self, path: str = ".") -> str:
        return "\n".join(sorted(e.name + ("/" if e.is_dir() else "") for e in
                                self._resolve(path)[0].iterdir() if e.name != ".git"))[:_OUT_CAP]

    def t_edit_file(self, path: str, old: str, new: str) -> str:
        def edit(p: Path, rel: str) -> str:
            text = p.read_text()
            if (n := text.count(old)) != 1:
                return f"ERROR: `old` occurs {n} times in {rel}; it must occur exactly once"
            p.write_text(text.replace(old, new, 1))
            return f"edited {rel}"
        return self._write(path, edit)

    def t_write_file(self, path: str, content: str) -> str:
        def write(p: Path, rel: str) -> str:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"wrote {rel} ({len(content)} chars)"
        return self._write(path, write)

    def _venv_dir(self, python: str) -> str:
        """The real dir of the worktree's venv (its ``.venv`` link, else the project python's venv); '' = none."""
        link = self.root / ".venv"
        if link.exists():
            return os.path.realpath(link)
        return os.path.realpath(Path(python).parent.parent) if python else ""

    def _pythons(self, python: str) -> frozenset[str]:
        """Interpreter paths a command may name (lexically): the worktree's .venv python(s) and the project python."""
        out = {str(self.root / ".venv" / "bin" / n) for n in ("python", "python3")}
        if python:
            out |= {os.path.normpath(python), os.path.realpath(python)}
        if self.venv:
            out |= {os.path.join(self.venv, "bin", n) for n in ("python", "python3")}
        return frozenset(out)

    @staticmethod
    def _segments(command: str) -> list[list[str]]:
        """``command`` as simple commands (split on ; && || | & newlines and parentheses), each a token list."""
        try:
            lx = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>")
            lx.whitespace_split = True
            toks = list(lx)
        except ValueError:
            toks = command.split()
        segs, cur = [], []
        for t in toks:
            if t and set(t) <= set(";&|()") or t == "\n":
                if cur:
                    segs.append(cur)
                cur = []
            else:
                cur.append(t)
        if cur:
            segs.append(cur)
        return segs

    def _inside(self, p: str, *, write: bool = False) -> bool:
        roots = (str(self.root),) if write else (str(self.root),) + self.readonly_roots + ((self.venv,) if self.venv else ())
        if write and (p == str(self.root / ".venv") or p.startswith(str(self.root / ".venv") + os.sep)):
            return False
        return p == "/dev/null" or any(p == r or p.startswith(r + os.sep) for r in roots)

    @staticmethod
    def _pathlike(piece: str) -> bool:
        """A token piece that can name a file: '/'-containing (not a bare '/' operator, not a regex / sed address
        whose first component does not exist), or ~ / .. relative."""
        if piece.startswith(("~", "..")) or piece in ("/", "//"):     # a bare / is the filesystem root (`ls /`)
            return True
        if "/" not in piece or not piece.strip("/ .*+?^$[]{}()|\\"):
            return False
        if piece.startswith("/"):
            first = "/" + piece.lstrip("/").split("/", 1)[0]
            return os.path.lexists(first)          # /dwi/, /def foo/, //2]: no such top-level dir: not a path
        return True

    def _confined(self, command: str) -> str:
        """'' if every path in ``command`` stays in the worktree or a read-only root (11-08: ``cd`` within the worktree
        moves the base of later relative paths; the project python may run; shell variables assigned in the command
        itself may be used; writes must land inside the worktree, never in its .venv)."""
        if _DENY.search(command):
            return "names v1 ledgers/kgraph/data/idea files"
        try:
            shlex.split(command)
        except ValueError:
            return "unparseable (unbalanced quotes)"
        shell = _expanded(command)                 # what the shell itself expands ('...' and quoted heredocs: nothing)
        if _SUBST.search(shell):
            return "command substitution ($(...) / backticks): use literal paths relative to the worktree"
        local = set(_ASSIGN.findall(shell)) | set(_FOR_VAR.findall(shell))
        if bad := sorted({v for v in _VAR.findall(shell) if v not in local}):
            return (f"environment variable(s) {bad}: only shell variables assigned in the same command may be used "
                    "(the data and the environment outside the worktree are not reachable from here)")
        cwd = str(self.root)
        for seg in self._segments(command):
            if seg and seg[0] == "cd":
                tgt = seg[1] if len(seg) > 1 else str(self.root)
                if tgt == "-" or _VAR.search(tgt):
                    return "cd to a dynamic target"
                new = os.path.realpath(os.path.join(cwd, os.path.expanduser(tgt)))
                if not (new == str(self.root) or new.startswith(str(self.root) + os.sep)):
                    return f"cd {tgt!r} leaves the worktree"
                cwd = new
                seg = seg[2:]                     # (a newline is no separator here: check what follows the cd too)
            for piece in (q.lstrip("'\"") for t in seg for q in re.split(r"[=:,<>|;&()]", t)):
                if not self._pathlike(piece):
                    continue
                lex = os.path.normpath(os.path.join(cwd, os.path.expanduser(piece)))
                if lex in self.pythons:
                    continue
                if not self._inside(os.path.realpath(lex)):
                    return f"path {piece!r} is outside the worktree"
        return self._writes_confined(command)

    def _writes_confined(self, command: str) -> str:
        """'' unless a write target (a redirect / tee target, an argument of a copy / move / in-place tool) lands
        outside the worktree or inside its shared .venv (relative targets resolved against the worktree root and the
        command's cd's, conservatively both)."""
        targets = [m.group(1) for m in _REDIRECT.finditer(command)]
        targets += [t for m in _TEE.finditer(command) for t in m.group(1).split()]
        if _WRITER.search(command):
            targets += [q for t in re.split(r"[\s;&|()<>]+", command) for q in re.split(r"[=:,]", t)]
        bases = {str(self.root)}
        for seg in self._segments(command):
            if seg and seg[0] == "cd" and len(seg) > 1:
                bases.add(os.path.realpath(os.path.join(str(self.root), seg[1])))
        for t in (x.strip("'\"") for x in targets):
            if not t or t.startswith("-") or t.startswith("&") or not ("/" in t or t.startswith((".", "~"))):
                continue
            if t.startswith("/") and not self._pathlike(t):
                continue
            for b in bases:
                p = os.path.realpath(os.path.join(b, os.path.expanduser(t)))
                if not self._inside(p, write=True):
                    return f"writes {t!r} outside the worktree (or into its shared .venv)"
        return ""

    def _net_problem(self, command: str) -> str:
        """'' unless the command runs python with network code (inline, or a worktree .py script it names)."""
        if not _PY_CMD.search(command) and ".venv/bin/python" not in command and \
                not any(py in command for py in getattr(self, "pythons", ())):
            return ""
        if _NET_PY.search(command):
            return "network access from python"
        for tok in re.split(r"[\s;&|()<>]+", command):
            tok = tok.strip("'\"")
            if not tok.endswith(".py"):
                continue
            for base in {str(self.root)} | {os.path.realpath(os.path.join(str(self.root), s[1]))
                                             for s in self._segments(command) if s and s[0] == "cd" and len(s) > 1}:
                p = os.path.realpath(os.path.join(base, tok))
                if self._inside(p, write=True) and os.path.isfile(p):
                    try:
                        text = Path(p).read_text(errors="replace")
                    except OSError:
                        continue
                    if _NET_PY.search(text):
                        return f"network access in {tok}"
                    if why := next((w for rx, w in _BANNED if rx.search(text)), ""):
                        return f"{why} in {tok}"
        return ""

    def t_bash(self, command: str, description: str = "", timeout: Any = None, **_ignored: Any) -> str:
        """v3-c7i I7: ``description`` / ``timeout`` (what other tool sets pass) are accepted and ignored."""
        if why := next((w for rx, w in _BANNED if rx.search(command)), ""):
            return f"REFUSED ({why}): `{command[:200]}` is banned in the research worktree"
        if why := self._confined(command):
            return f"REFUSED (confinement: {why}): `{command[:200]}`"
        if why := self._net_problem(command):
            return f"REFUSED (network: {why}): `{command[:200]}` -- sessions run offline"
        if hit := self._locked_write(command):
            return (f"REFUSED (LOCKED): `{command[:200]}` would write {hit}, which is LOCKED (surface.locked); "
                    "it may never be edited, restored or reverted by hand -- work around it")
        before = self._locked_status()
        try:
            cp = subprocess.run(command, shell=True, cwd=self.root, capture_output=True, text=True,
                                timeout=self.bash_timeout_s, env=self._env())
        except subprocess.TimeoutExpired:
            self._restore_locked(before)
            return f"ERROR: command exceeded {self.bash_timeout_s:.0f}s"
        self.dirty = True                      # a command may have written files: re-smoke
        out = cp.stdout + ("\n[stderr]\n" + cp.stderr if cp.stderr else "")
        if undone := self._restore_locked(before):
            return (f"REFUSED (LOCKED): the command changed LOCKED file(s) {undone}; they were restored to HEAD. "
                    f"Locked files may never be edited. exit={cp.returncode}\n{out[-_OUT_CAP // 2:]}")
        return f"exit={cp.returncode}\n{out[-_OUT_CAP:]}"

    def _env(self) -> dict[str, str]:
        """The command's env: no SLURM_* vars, uv / pip frozen, offline, and the project venv first on PATH (``python``
        is the project's interpreter; 11-08: sessions could not import-check their edits)."""
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "VIRTUAL_ENV", "PYTHONHOME"))}
        env.update(SAFE_ENV)
        env.update(OFFLINE_ENV)
        venv_bin = self.root / ".venv" / "bin"
        if venv_bin.is_dir():
            env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
            env["VIRTUAL_ENV"] = str(self.root / ".venv")
        return env

    def _locked_write(self, command: str) -> str:
        """The first locked (whole-file / tree) path a write-ish command names: a redirect / tee target, or any
        argument of an in-place / copy / move / patch tool. '' when none (best effort: see _restore_locked)."""
        whole = [p for p in self.locked if "::" not in p]
        if not whole:
            return ""

        def locked(piece: str) -> bool:
            piece = piece.strip("'\"")
            if not piece or piece.startswith("-"):
                return False
            p = os.path.normpath(os.path.join(str(self.root), os.path.expanduser(piece)))
            rel = os.path.relpath(p, str(self.root))
            return not rel.startswith("..") and any(path_matches(rel, pat) or path_matches(rel + "/", pat)
                                                    for pat in whole)
        targets = [m.group(1) for m in _REDIRECT.finditer(command)]
        targets += [t for m in _TEE.finditer(command) for t in m.group(1).split()]
        if _WRITER.search(command):
            targets += [q for t in re.split(r"[\s;&|()<>]+", command) for q in re.split(r"[=:,]", t)]
        return next((t.strip("'\"") for t in targets if locked(t)), "")

    def _locked_pathspecs(self) -> list[str]:
        return [p.rstrip("/") for p in self.locked if "::" not in p]

    def _locked_status(self) -> str | None:
        """`git status --porcelain` of the locked whole-file paths (None outside a git worktree)."""
        specs = self._locked_pathspecs()
        if not specs or not (self.root / ".git").exists():
            return None
        cp = subprocess.run(["git", "-C", str(self.root), "status", "--porcelain", "--untracked-files=all", "--",
                             *specs], capture_output=True, text=True, timeout=120)
        return cp.stdout if cp.returncode == 0 else None

    def _restore_locked(self, before: str | None) -> list[str]:
        """After a bash command: every locked path whose status changed is put back to HEAD (a new file removed).
        Returns the restored paths."""
        if before is None or (after := self._locked_status()) is None or after == before:
            return []
        old = set(before.splitlines())
        done = []
        for line in after.splitlines():
            if line in old or len(line) < 4:
                continue
            for rel in (r.strip().strip('"') for r in line[3:].split(" -> ")):     # a rename: both sides
                git = ["git", "-C", str(self.root)]
                if subprocess.run([*git, "cat-file", "-e", f"HEAD:{rel}"], capture_output=True).returncode == 0:
                    subprocess.run([*git, "checkout", "HEAD", "--", rel], capture_output=True, timeout=120)
                else:
                    subprocess.run([*git, "rm", "-q", "--cached", "--ignore-unmatch", "--", rel], capture_output=True)
                    try:
                        (self.root / rel).unlink()
                    except OSError:
                        pass
                done.append(rel)
        return done

    def t_smoke(self) -> str:
        # 11-08: the Opus session is held open the whole time this tool waits; ``slurm.smoke_session_wait_s`` bounds the
        # queue wait here (past it: "not run, finish" -- the loop re-smokes after the session, no LLM held). 0 = unbounded
        # beyond the adapter's own smoke_wait_s (the old call, unchanged)
        slurm = getattr(getattr(getattr(self.adapter, "ctx", None), "cfg", None), "slurm", None)
        wait = float(getattr(slurm, "smoke_session_wait_s", 0.0) or 0.0)
        res = self.last_smoke = self.adapter.smoke(self.wt, queue_s=wait) if wait > 0 else self.adapter.smoke(self.wt)
        self.dirty = False
        if getattr(res, "infra", False):
            return (f"smoke NOT RUN: {res.reason}. The ENVIRONMENT (cache dir / weights / disk) failed, not your change; "
                    "the loop holds smokes until it is back. Do not change code because of it; finish when the code is "
                    "ready (the loop re-smokes it).")
        if getattr(res, "unavailable", False):
            return (f"smoke NOT RUN: {res.reason}. The cluster gave the smoke job no GPU slot in time; this says "
                    "nothing about your change. Do not change code because of it; finish when the code is ready "
                    "(the loop re-smokes it).")
        return f"smoke {'OK' if res.ok else 'FAILED'}: {res.reason}\n{res.log_tail[-4000:]}"


class LLMResearcher:
    def __init__(self, ctx: LoopContext, llm: Any, adapter: Any, embedder: Any = None) -> None:
        self.ctx, self.llm, self.adapter = ctx, llm, adapter
        self.embedder = embedder or HashingEmbedder()     # the selector's default: same near-dup verdicts
        self._ic: dict[str, tuple[str, Any]] = {}          # request -> (diff hash, its init-compat probe)

    def config_rel(self) -> str:
        return config_file(self.ctx, self.adapter)

    def _own_surface(self, wt: Worktree, parent_commit: str, paths: Sequence[str]) -> tuple[str, ...]:
        """``paths`` (the diff vs ``wt.base_commit``) judged on the session's OWN work vs the first parent: a locked /
        off-surface path the ancestry alone changed is dropped (drift), and every locked path / span that differs
        from ``parent_commit`` is added -- a crossover can never ship a locked file the first parent does not have."""
        surface = self.ctx.spec.surface
        own = tuple(self.adapter.diff(dataclasses.replace(wt, base_commit=parent_commit))[1])

        def locked(p: str) -> bool:
            return any(path_matches(p, pat) for pat in surface.locked)
        keep = [p for p in paths if p in own or not (locked(p) or not intake.on_surface(p, surface.writable))]
        return tuple(keep + [p for p in own if locked(p) and p not in keep])

    def check(self, request: CandidateRequest, store: Any, wt: Worktree, *, family: str = "",
              claim: str = "", mechanism_tag: str = "", falsification: str = "",
              handoff: Mapping[str, str] | None = None, snapshot: Any = None, final: bool = False,
              dataset_only: bool = False, parent_commit: str = "", mechanism_only: bool = False,
              mechanism: str = "", declared_knobs: bool = False) -> str:
        """The intake rules (intake.diagnose, the very rules admit runs) on the worktree's current diff and the
        declared hypothesis: 'OK ...' or 'REJECT <rule>: <fix>'. Before any edit (empty_diff) the MECHANISM is still
        evaluated -- the mechanism rule, the anti-repeat rule and the in-flight reservations -- and passes as 'OK (no
        diff yet)' (live 2026-10-03: 20 sessions rejected dead_mechanism at intake; 10-13: sessions edited something
        just to get a check). Only the finish (``final``) of an empty diff is rejected empty_diff (a dataset_spec-only
        finish excepted: the loop writes its config). A v3 mechanism that passes is CLAIMED for this session
        (``ideation.claims``); another in-flight session's claim on the same mechanism rejects reserved_inflight."""
        # mechanism_only (v3-c5 I3, a noted paper's mechanism): the mechanism rules alone, on no diff, nothing claimed
        diff_text, paths, cfg_diff = ("", (), {}) if mechanism_only else self.adapter.diff(wt)
        if parent_commit and parent_commit != wt.base_commit and not mechanism_only:
            # the same ancestry exemption the final check applies (a locked / off-surface file only the ANCESTRY
            # changed is no violation: never tell the agent to "revert" a drifted locked file), and the same rule:
            # any locked path / span that differs from the first parent IS one (review M1, 10-13)
            paths = self._own_surface(wt, parent_commit, paths)
        ho = handoff or {}
        probe = Candidate(request.id, self.ctx.task_id, "", wt.base_commit, diff_text, diff_hash(diff_text),
                          tuple(paths), dict(cfg_diff), HypothesisRecord(
                              id="", task_id=self.ctx.task_id, kind=HypothesisKind.TRIAL, claim=claim,
                              mechanism_tag=mechanism_tag.strip(), family="", predictions={},
                              falsification=falsification),
                          HandoffNote("", self.ctx.task_id, request.id, ho.get("actions", ""),
                                      ho.get("learnings", ""), ho.get("concerns", "")))
        fams, _ = intake.derive_families(self.ctx, probe)
        declared = family or (fams[0] if fams else "")
        probe = dataclasses.replace(probe, hypothesis=dataclasses.replace(probe.hypothesis, family=declared))
        if mechanism_only:
            if not (mechanism_tag.strip() or claim.strip()):
                return "OK (nothing to check)"
            rule, fix = self._mech_checks(request, store, probe, claim, family or declared, mechanism_tag)
            from loop.core.ideation import cluster_exempt
            if not rule and is_v3(request) and not cluster_exempt(request):   # v3-c7i I3: a saturated-negative
                from loop.core import alloc as AL                               # cluster past its cap (+ 10-21: a
                try:                                                            # negative family: cluster_rule)
                    rule, fix = AL.cluster_rule(self.ctx, family or declared, mechanism_tag, mechanism or claim, claim)
                except Exception:  # noqa: BLE001 - no cluster verdict: the other rules stand
                    rule, fix = "", fix
            if not rule and (got := self._shadow_block(request, store, family or declared, mechanism_tag, claim)):
                rule, fix = got                                                 # 10-21: program shadows explore
            return f"REJECT {rule}: {fix}" if rule else "OK" + (f" {fix}" if fix else "")
        sig = None if is_program_arm(request) else self.signature(
            request, store, family or declared, mechanism_tag, mechanism or claim)
        rule, fix = intake.diagnose(self.ctx, store, probe, operator=request.operator, embedder=self.embedder,
                                    strategy=request.strategy, snapshot=snapshot, signature=sig,
                                    dup_exempt=is_program_arm(request))
        fix = intake.session_text(rule, fix)
        if not rule and is_v3(request) and (got := self.reserved(request, store, family or declared, mechanism_tag)):
            rule, fix = got
        if not rule and (got := self._family_block(request, family or declared, mechanism_tag, claim, mechanism)):
            rule, fix = got                                     # 10-21: a negative family needs a stated difference
        if not rule and (got := self._shadow_block(request, store, family or declared, mechanism_tag, claim, list(cfg_diff))):
            rule, fix = got                                     # 10-21: an operator-program arm is in flight here
        if rule == "empty_diff":
            if final and not dataset_only:
                return f"REJECT {rule}: {fix}"
            if not mechanism_tag.strip():
                return ("OK (no diff yet): nothing to check -- pass mechanism_tag (and family, claim) to check a "
                        "mechanism before editing, or edit first")
            rule, fix = self._mech_checks(request, store, probe, claim, family or declared, mechanism_tag)
            if not rule and (got := self._family_block(request, family or declared, mechanism_tag, claim, mechanism)):
                rule, fix = got
            if not rule and (got := self._shadow_block(request, store, family or declared, mechanism_tag, claim)):
                rule, fix = got
            if rule:
                return f"REJECT {rule}: {fix}"
            self.claim(request, store, family or declared, mechanism_tag)
            return (f"OK (no diff yet): mechanism {mechanism_tag.strip()} passes the mechanism rules (not dead, not "
                    "a near-duplicate of a live / queued node, not repeating a negative node, not claimed by another "
                    "in-flight session)" + (f" {fix}" if fix else "") + ". Now make the edit, check again, smoke, "
                    "finish.")
        if rule in ("sweep_only", "below_noise"):
            intake.redirect(self.ctx, store, probe, request.id, request_id=request.id, at="check")
        if not rule and request.brief_kind == "bold":    # v3-c8b: a knob / weight change is out of this lane's scope
            from loop.core import bold as BOLD
            ad = self.adapter    # v3 review M3: the AST rule alone at session time (no judge); parent names from the parent's source
            show, repo = getattr(ad, "_show", None), getattr(ad, "repo", None)
            rule, fix = BOLD.knob_rule(self.ctx, diff_text, claim, None, (
                lambda rel: show(repo, wt.base_commit, rel)) if show and repo else None)
        if not rule and final and request.brief_kind in ("bold", "kuhn") and is_v3(request) \
                and getattr(getattr(self.ctx.cfg, "ideation", None), "struct_tune", False):
            from loop.core import struct_tune as STT      # 10-22: a structural change is tuned before it is judged
            rule, fix = STT.untunable(self.ctx, dict(cfg_diff), ["declared"] if declared_knobs else None)
        if rule:
            return f"REJECT {rule}: {fix}{self._stack_hint(store, snapshot) if rule == 'multi_family' else ''}"
        note = ""
        if intake.needs_init_compat(self.ctx, probe, fams) and hasattr(self.adapter, "init_compat"):
            # EARLY INIT-COMPAT PROBE (2026-10-06: 75 init_incompatible rejections, 49 moonshots, up to 35 a day, each
            # a whole session lost at intake): the intake's own check on the current worktree, once per diff
            if (got := self._ic.get(request.id)) is None or got[0] != probe.diff_hash:
                try:
                    got = self._ic[request.id] = (probe.diff_hash, self.adapter.init_compat(wt, parent_commit=wt.base_commit))
                except Exception as exc:  # noqa: BLE001 -- a probe fault never blocks; intake checks after finish
                    got = (probe.diff_hash, None)
                    note = f" (init-compat probe unavailable: {type(exc).__name__})"
            if got[1] is not None and not got[1].ok and request.brief_kind == "bold":
                got = (got[0], self._relax(request, store, got[1]))     # v3-c8b: identity relaxed, bans kept
            if got[1] is not None and not got[1].ok:
                store.log("init_compat_probe", request_id=request.id, reason=str(got[1].reason)[:600])
                if got[1].swap:
                    return (f"REJECT {got[1].rule}: {str(got[1].reason)[:900]}. Intake rejects this. A new init / "
                            "architecture must construct through model.py's ViT(variant=model.type), load through its "
                            "load_pretrained and give a finite forward")
                return (f"REJECT init_incompatible: {str(got[1].reason)[:900]}. Intake rejects this. Keep every parent "
                        "parameter's name and shape loadable from the parent checkpoint: add new modules beside the "
                        "pretrained ones, zero-initialised (residual / gate at 0) so the initial forward and loss match "
                        "the parent's; never rename, resize or drop a pretrained parameter -- or change the init "
                        "itself (another model.type / checkpoint: checked as an init swap)")
            if got[1] is not None and got[1].swap:
                fb = got[1].checks.get("flop_budget") or {}
                note = note or (f" init-swap check OK ({got[1].checks['init_swap']['to']}: constructs, loads, finite "
                                "forward, legal source; FLOP budget " + (f"projected {fb['projected']:.3f}" if
                                "projected" in fb else "measured by the smoke") + "). The parent's hyperparameters "
                                "were set for its own init: variants / a branched plan in finish screen retunes "
                                "beside it")
            note = note or (" init-compat probe OK (the parent checkpoint loads; initial loss and parameter count "
                            "within bounds)" if got[1] is not None else "")
        if cold := NV.mech_dup_rule(self.ctx, store, probe, request.strategy, operator=request.operator)[1]:
            note += f" {cold}"
        if is_v3(request) and mechanism_tag.strip():
            self.claim(request, store, family or declared, mechanism_tag)
        return f"OK: {len(paths)} path(s), family {declared}.{note} Next: smoke (if not yet), then finish."

    def _shadow_block(self, request: CandidateRequest, store: Any, family: str, tag: str, claim: str,
                      config_keys: Sequence[str] = ()) -> tuple[str, str] | None:
        """('shadowed_by_program', fix) for an explore / literature proposal near an in-flight operator-program arm
        (:func:`loop.core.program.shadow_rule`); None otherwise. Not a repair / port session or a program arm."""
        from loop.core.ideation import cluster_exempt
        if not is_v3(request) or cluster_exempt(request) or request.brief_kind not in ("explore", "literature") \
                or not (tag.strip() or claim.strip()):
            return None
        from loop.core import program as PG
        try:
            rule, fix = PG.shadow_rule(self.ctx, store, family, tag, claim, list(config_keys))
        except Exception:  # noqa: BLE001 - no shadow verdict: the other rules stand
            return None
        return (rule, fix) if rule else None

    def _family_block(self, request: CandidateRequest, family: str, tag: str, claim: str, mechanism: str = ""
                      ) -> tuple[str, str] | None:
        """('negative_family', fix) when the mechanism falls in a NEGATIVE FAMILY cluster (:func:`alloc.family_rule`)
        and the claim carries no sound ``differs_from_cluster``; None otherwise. Not a repair / port / program arm."""
        from loop.core.ideation import cluster_exempt
        if not is_v3(request) or cluster_exempt(request) or not (tag.strip() or claim.strip()):
            return None
        from loop.core import alloc as AL
        try:
            rule, fix = AL.family_rule(self.ctx, family, tag, mechanism or claim, claim)
        except Exception:  # noqa: BLE001 - no family verdict: the other rules stand
            return None
        return (rule, fix) if rule else None

    def signature(self, request: CandidateRequest, store: Any, family: str, tag: str, text: str
                  ) -> tuple[str, tuple[str, ...]] | None:
        """v3-c6i: the mechanism signature (loop.core.mechanisms) intake's dedup compares for a v3 session (None for
        any other: the claim + diff near_dup stays)."""
        if not is_v3(request) or not hasattr(store, "get_node"):
            return None
        from loop.core import mechanisms as MS
        anc = MS.ancestors(store, request.parent_ids)
        return MS.signature(family, tag, text, MS.claims_of(store, anc)), tuple(request.parent_ids)

    def _mech_checks(self, request: CandidateRequest, store: Any, probe: Candidate, claim: str, family: str,
                     tag: str) -> tuple[str, str]:
        """The intake's mechanism rules on a probe WITHOUT a diff (check before any edit, a noted paper at note
        time): dead_mechanism / near_dup_mechanism (intake.mech_rule), the embedding near_dup on the claim (v3-c5 I3:
        10-13, a node was rejected near_dup only at intake), repeat_of_negative, and (v3) another in-flight session's
        claim. ('', advisory-or-'') when every rule passes."""
        if is_program_arm(request):
            return "", ""
        rule, fix = intake.mech_rule(self.ctx, store, probe, request.strategy, request.operator)
        vec = self.embedder.embed(intake.candidate_text(probe)) if claim.strip() and not rule else None
        if vec is not None and (sig := self.signature(request, store, family, tag, claim)) is not None:
            from loop.core import mechanisms as MS     # v3-c6i: the mechanism signature, not claim + diff
            if (got := MS.dup_rule(self.ctx, store, sig[0], sig[1], claim))[0]:
                return got
        elif vec is not None and request.strategy is not Strategy.DESIGN:
            from loop.contracts import NodeStatus
            from loop.select.embed import cosine
            near = next((n for n in store.nodes() if n.embedding is not None and n.status is not NodeStatus.REJECTED
                         and cosine(vec, n.embedding) > self.ctx.cfg.intake.near_dup_cos), None)
            if near is not None:
                return "near_dup", (f"too close to {near.id} ({near.family}/{near.mechanism_tag}); change the "
                                    "mechanism or dose, or reword the claim to what is actually new")
        if vec is not None:
            got = intake.repeat_rule(store, probe, vec)
            if got[0]:
                return got
        if not rule and is_v3(request):
            got = self.reserved(request, store, family, tag)
            if got[0]:
                return got
        return rule, fix

    @staticmethod
    def _live_ids(store: Any) -> set[str]:
        from loop.contracts import K_INFLIGHT
        return {e.get("id", "") for e in store.get_state(K_INFLIGHT) or [] if not e.get("queued")}

    def reserved(self, request: CandidateRequest, store: Any, family: str, tag: str) -> tuple[str, str]:
        """('reserved_inflight', fix) when another IN-FLIGHT session claimed this mechanism (the same normalized tag,
        or >= half of the tag words shared within the same family), else ('', '')."""
        from loop.core.ideation import K_CLAIMS
        if not tag.strip() or is_program_arm(request):
            return "", ""
        mine, words, live = NV.norm_tag(tag), NV.tag_words(tag), self._live_ids(store)
        for rid, c in (store.get_state(K_CLAIMS) or {}).items():
            if rid == request.id or rid not in live or not isinstance(c, Mapping):
                continue
            theirs, tw = NV.norm_tag(str(c.get("tag") or "")), NV.tag_words(str(c.get("tag") or ""))
            jac = len(words & tw) / len(words | tw) if words | tw else 0.0
            if mine == theirs or (jac >= 0.5 and c.get("family") == family):
                return "reserved_inflight", (f"in-flight session {rid} is already building {c.get('family')}/"
                                             f"{c.get('tag')}; pick another mechanism, axis or component")
        return "", ""

    def claim(self, request: CandidateRequest, store: Any, family: str, tag: str) -> None:
        """Record this session's mechanism (``ideation.claims``; claims of sessions no longer in flight dropped)."""
        from loop.core.ideation import K_CLAIMS
        if not hasattr(store, "set_state"):
            return
        live = self._live_ids(store) | {request.id}
        cur = dict(store.get_state(K_CLAIMS) or {})
        new = {k: v for k, v in cur.items() if k in live}
        new[request.id] = {"family": family, "tag": tag.strip(), "ts": time.time()}
        if new != cur:
            store.set_state(K_CLAIMS, new)

    def dataset_only(self, store: Any, wt: Worktree, fin: dict[str, Any]) -> str:
        """A finish with a ``dataset_spec`` and NO change (10-13: such sessions failed empty_diff, or were rejected
        family_mismatch): point the config's ``data.dataset_dir`` at the dataset (an existing one with the same content
        when the bank holds or builds it, else the spec's name under ``databank.root``) and file the node under the
        family that owns that key. '' when done, else why not (the session then fails empty_diff as before)."""
        db, rel = getattr(self.ctx.cfg, "databank", None), self.config_rel()
        if db is None or not db.enabled or not rel:
            return "no data bank / config file"
        got, why = DB.validate(fin["dataset_spec"])
        if why:
            return why
        bk, h = DB.book(store), DB.spec_hash(got)
        name = next((n for n, x in bk["registered"].items() if x.get("spec_sha") == h), "") or next(
            (n for n, r in bk["requests"].items() if DB.spec_hash(r["spec"]) == h and r["state"] != "refused"),
            "") or got["name"]
        p = Path(wt.path) / rel
        try:
            p.write_text(set_yaml_value(p.read_text(), "data.dataset_dir", str(Path(db.root) / name)))
        except (OSError, ValueError) as exc:
            return f"config: {exc}"
        if fam := intake.family_of(self.ctx, "data.dataset_dir"):
            fin["hypothesis"] = {**fin["hypothesis"], "family": fam}
        return ""

    def precheck(self, request: CandidateRequest, store: Any, parents: list) -> str:
        """The brief's mechanism shortlist (``request.shortlist``: family|tag) run through the intake's mechanism rule
        (dead / near-duplicate) BEFORE the session, as a brief section: 10-13, 40% of turns went to intake rejects
        (near_dup_mechanism 78, dead 34) of mechanisms the brief itself had pointed at. '' without a shortlist."""
        from types import SimpleNamespace
        items = tuple(getattr(request, "shortlist", ()) or ())
        if not items:
            return ""
        pc = parents[0].commit if parents and parents[0] is not None else ""
        lines = []
        for it in items:
            fam, _, tag = it.partition("|")
            probe = SimpleNamespace(parent_commit=pc, changed_paths=(), config_diff={}, diff_text="",
                                    hypothesis=SimpleNamespace(mechanism_tag=tag, claim=tag.replace("_", " "),
                                                               falsification="", family=fam),
                                    handoff=SimpleNamespace(actions="", learnings="", concerns=""))
            try:
                rule, fix = intake.mech_rule(self.ctx, store, probe, request.strategy, request.operator)
            except Exception as exc:  # noqa: BLE001 - a pre-check never blocks the session
                rule, fix = "unchecked", type(exc).__name__
            lines.append(f"- {fam}/{tag}: " + (f"REJECT {rule}: {fix[:300]}" if rule else
                                              "OK" + (f" ({fix[:200]})" if fix else "")))
        return ("\n\n## Pre-checked mechanisms (the intake's mechanism rule, run before this session)\nThe brief points at "
                "these mechanisms (re-test candidates, the latest ideas). REJECT = a session building it as-is is "
                "rejected at intake: bring another mechanism, or name every listed node in hypothesis.claim and say "
                "what answers their failure.\n" + "\n".join(lines))

    def _bar_sizing(self) -> str:
        """11-08: with a project knowledge file (loop.knowledge), the size floor: 99% of the nanobrain sessions'
        self-predicted gains sat within +-.004 -- noise the loop can never measure."""
        from loop import knowledge as KN
        try:
            if not (KN.enabled(self.ctx) and KN.paths(self.ctx)):
                return ""
            k = float(getattr(KN.cfg_of(self.ctx), "bar_sd", 2.0) or 2.0)
            sds = ", ".join(f"{KN._short(a)} {k * float(v):.4f}" for a, v in (self.ctx.cfg.axis_sd or {}).items())
        except Exception:  # noqa: BLE001
            return ""
        return (f" Size floor (the brief's 'The bar'): the change must plausibly move ONE named axis by >= {k:g} paired "
                f"sd ({sds}); a predicted +-.004 on the panel is noise the loop cannot measure -- not a candidate.")

    def _python(self) -> str:
        """The project's interpreter (``loop.project.python``, else the adapter's) -- '' when none is configured."""
        proj = getattr(getattr(self.ctx, "cfg", None), "project", None)
        py = (proj.get("python") if isinstance(proj, Mapping) else getattr(proj, "python", "")) or ""
        return str(py or getattr(self.adapter, "python", "") or "")

    def _inflight_block(self, request: CandidateRequest, store: Any) -> str:
        """11-08: what other sessions are building / proposed recently, as of THIS session's start (loop.core.tested;
        the brief itself may have been rendered long before)."""
        from loop.core import tested as TS
        try:
            return TS.session_block(self.ctx, store, request.id)
        except Exception:  # noqa: BLE001 - never blocks a session
            return ""

    def _off_clock(self) -> bool:
        from loop.core.levers import opt
        return bool(opt(self.ctx, "smoke_off_clock"))

    def _web_n(self) -> int:
        from loop.core.levers import opt
        return int(opt(self.ctx, "web_max_searches"))

    def _stack_hint(self, store: Any, snapshot: Any) -> str:
        return "\n" + briefs.stackable_line(self.ctx, store, snapshot)

    def limits(self, request: CandidateRequest, store: Any = None) -> tuple[int, float]:
        """(max tool calls, minutes): a moonshot gets its own, larger caps; with ``store``, a session starting with
        >= ``ideation.slow_sessions_n`` sessions in flight gets x ``ideation.timeout_scale`` minutes."""
        r = self.ctx.cfg.researcher
        turns, minutes = (r.moonshot_max_turns, r.moonshot_timeout_min) if request.strategy is Strategy.MOONSHOT \
            else (r.max_turns, r.timeout_min)
        if request.strategy is not Strategy.MOONSHOT:
            from loop.core import bold as BOLD            # review fix S3: portfolio sessions read history + slate: more turns
            if BOLD.portfolio_on(self.ctx, getattr(request, "brief_kind", "") or ""):
                turns = max(turns, int(r.portfolio_max_turns))
        if store is not None and hasattr(store, "get_state"):
            from loop.contracts import K_INFLIGHT
            from loop.core.levers import opt
            live = sum(1 for e in store.get_state(K_INFLIGHT) or [] if not e.get("queued"))
            if live >= int(opt(self.ctx, "slow_sessions_n")):
                minutes *= float(opt(self.ctx, "timeout_scale"))
        return turns, minutes

    def research(self, request: CandidateRequest, store: Any, snapshot: Any) -> ResearchResult:
        parents = [store.get_node(pid) for pid in request.parent_ids]
        missing = [p for p, n in zip(request.parent_ids, parents) if n is None and p != ROOT_NODE_ID]
        if missing or not parents:
            return ResearchResult("failed", None, f"bad_request: unknown parent(s) {missing}")
        parent = parents[0]
        parent_commit = parent.commit if parent is not None else self.adapter.root_commit()
        depth = (parent.debug_depth if parent else 0) + 1
        cap = self.ctx.cfg.researcher.max_debug
        if request.operator is Operator.DEBUG and depth > cap:
            return ResearchResult("failed", None, f"debug_cap: lineage already debugged {depth - 1}/{cap} times")
        if request.operator is Operator.CROSSOVER and (len(parents) < 2 or None in parents[1:]):
            return ResearchResult("failed", None, "bad_request: crossover needs two known parents")
        wt = self.adapter.prepare_worktree(parent_commit=parent_commit, name=f"loop-{request.id}")
        if request.operator is Operator.DEBUG and parent is not None:   # the fix is judged against
            ctl = store.get_node(parent.parent_id or ROOT_NODE_ID)       # the crashed node's control
            wt = dataclasses.replace(wt, base_commit=ctl.commit if ctl else self.adapter.root_commit())
        if request.operator is Operator.CROSSOVER:     # diff + control = the parents' common ancestor
            ctl = store.get_node(intake.common_ancestor(store, *request.parent_ids[:2]))
            wt = dataclasses.replace(wt, base_commit=ctl.commit if ctl else self.adapter.root_commit())
        try:
            return self._run(request, store, snapshot, parents, parent_commit, depth, wt)
        finally:
            try:
                self.adapter.reap_worktree(wt)
            except Exception:  # noqa: BLE001 — best effort; the driver re-reaps orphans
                pass

    def _run(self, request: CandidateRequest, store: Any, snapshot: Any, parents: list,
             parent_commit: str, depth: int, wt: Worktree) -> ResearchResult:
        cfg, surface = self.ctx.cfg, self.ctx.spec.surface
        merge = "n/a"
        if request.operator is Operator.CROSSOVER:      # a stack merges every further leg in turn
            got = [(p.id, *self.adapter.merge(wt, other_commit=p.commit)) for p in parents[1:]]
            merge = "clean" if all(ok for _, ok, _ in got) else "\n".join(
                f"{pid}: " + ("clean" if ok else f"CONFLICTS (merge aborted; re-apply its change by hand):\n"
                                                   f"{c[:6000 // len(got)]}") for pid, ok, c in got)
        base = getattr(self.adapter, "repo", None)
        tools = WorktreeTools(self.adapter, wt, writable=tuple(surface.writable),
                              locked=tuple(surface.locked),
                              readonly_roots=((str(base),) if base else ()) + briefs.references(self.ctx),
                              config_rel=self.config_rel(), protected=tuple(surface.protected_keys),
                              checker=lambda **kw: self.check(request, store, wt, snapshot=snapshot,
                                                              parent_commit=parent_commit, **kw),
                              python=self._python())
        turns, minutes = self.limits(request, store)
        web = wants_web(request, self.ctx)
        tools.need_prior_work = web and not bounded_web(request, self.ctx)
        tools.request_id = request.id
        ic = getattr(self.ctx.cfg, "ideation", None)
        tools.v3 = v3 = is_v3(request)
        if v3:
            from loop.core.levers import opt
            tools.diagnoser = DX.Diagnoser(self.ctx, store, self.adapter)
            if opt(self.ctx, "history"):                  # v3-c9 C1: the trajectory dashboard, a partial view per session
                cols = HX.session_columns(self.ctx, request.id, getattr(request, "branch", "") or "")
                tools.historian = HX.Historian(self.ctx, store, cols, request_id=request.id,
                                               branch=getattr(request, "branch", "") or "",
                                               ablated=tuple(getattr(request, "ablated", ()) or ()))
                if callable(getattr(store, "log", None)):
                    store.log("history_view", request_id=request.id, branch=getattr(request, "branch", "") or "",
                              kind=request.brief_kind, columns=list(cols), frac=float(opt(self.ctx, "history_view_frac")))
            tools.claim_floor, tools.rows = float(getattr(request, "claim_floor", 0.0) or 0.0), \
                tuple(getattr(request, "rows", ()) or ())
            tools.program = request.brief_kind == "program"
            tools.size_cap, tools.min_components = int(opt(self.ctx, "claim_bounces")), \
                int(opt(self.ctx, "program_min_components"))
            tools.mismatch_tol, tools.mismatch_cap = float(opt(self.ctx, "claim_budget_tol")), \
                int(opt(self.ctx, "claim_mismatch_bounces"))           # v3-c5 I1: budget vs panel claim
            tools.logger = getattr(store, "log", None)
            tools.vs_k, tools.ref_gap = int(getattr(request, "k_candidates", 0) or 0), \
                float(getattr(request, "ref_gap", 0.0) or 0.0)                 # v3-c6i: the candidate slate
            tools.ref_kind = str(getattr(request, "ref_kind", "") or "")         # v3-c6r: the gap's definition
            tools.vs_cap = int(opt(self.ctx, "vs_bounces"))
            tools.need_analogy = request.brief_kind == "kuhn"
            if opt(self.ctx, "require_evidence") and request.brief_kind in tuple(opt(self.ctx, "evidence_kinds") or ()):
                tools.need_evidence, tools.evidence_cap = True, int(opt(self.ctx, "evidence_bounces"))
                tools.evidence_check = lambda hy, _c=self.ctx, _s=store: evidence_problem(hy, _c, _s)
            if opt(self.ctx, "smoke_off_clock"):
                tools.off_clock = ("smoke", "diagnose")
            from loop.core import tested as TS          # 11-08: the mechanism memory (check_idea's repeat rule)
            tools.memory_check = lambda _c=self.ctx, _s=store, _r=request.id, **kw: TS.check_text(_c, _s, _r, **kw)
            from loop.core import bold as BOLD          # v3-c8b: the bold lane + diverge-then-judge
            tools.bold = request.brief_kind == BOLD.KIND
            if request.brief_kind in BOLD.slate_kinds(self.ctx) and request.operator is Operator.DRAFT:
                judge = BOLD.llm_slate_judge(self.llm) if hasattr(self.llm, "complete_json") else None
                tools.slate_gate = BOLD.Slate(self.ctx, store, request.id, request.brief_kind, judge=judge,
                                              logger=getattr(store, "log", None),
                                              history=HX.current_table if opt(self.ctx, "history") else None)
                tools.slate_def = BOLD.slate_tool(self.ctx, request.brief_kind)
                tools.vs_k = 0                           # the slate carries the candidates and their predictions
        tools.need_fp = bool(v3 and IDEAS.is_fp_session(request)
                             or ic and ic.enabled and ic.first_principles and IDEAS.is_fp_session(request))
        slate = bool(ic and ic.enabled and ic.max_ideas > 1 and IDEAS.is_slate_session(request)) and not v3
        if slate:
            tools.planner = lambda ideas: IDEAS.propose(self.ctx, store, snapshot, request, wt.base_commit, ideas)
        if wants_research(request):
            tools.research = NT.ResearchStage(self.ctx, request.id, getattr(request, "branch", ""))
        if snapshot is not None and not v3 and request.brief_kind not in ("legs", "expose", "clock") \
                and getattr(snapshot, "incumbent_id", ROOT_NODE_ID) != ROOT_NODE_ID:
            from loop.legs import own_legs
            tools.need_stacks = tuple(own_legs(store, snapshot.incumbent_id))
        run = self.llm.run_agent(
            "researcher", system="You are a careful ML research engineer. You change code only "
                                 "through the provided tools and end with `finish`.",
            prompt=self._prompt(request, store, snapshot, parents, parent_commit, depth, merge, wt),
            cwd=wt.path, tools=list(TOOL_DEFS) + (list(V3_TOOL_DEFS) if v3 else [])
            + (list(NT.tool_defs()) if tools.research else [])
            + ([IDEAS.PROPOSE_TOOL] if slate else []) + ([tools.slate_def or BOLD_SLATE_TOOL] if tools.slate_gate is not None else []),
            executor=tools, finish_schema=finish_schema(self.ctx, v3, bold=tools.bold),
            max_turns=turns, timeout_s=minutes * 60.0, web=web)

        def out(res: ResearchResult) -> ResearchResult:      # web use rides on the result (node notes)
            return dataclasses.replace(res, web=web, web_searches=run.web_searches, web_fetches=run.web_fetches,
                                       mechanism_type=str(((run.finish or {}).get("hypothesis") or {})
                                                          .get("mechanism_type") or ""),
                                       dose_knob="".join(str(((run.finish or {}).get("hypothesis") or {})
                                                             .get("dose_knob") or "").split()),
                                       identical=((run.finish or {}).get("hypothesis") or {})
                                       .get("identical_at_default") is True,
                                       plan=self._plan(request, run.finish or {}, tools, wt),
                                       stacks_with=_ids(((run.finish or {}).get("handoff") or {}).get("stacks_with")),
                                       conflicts_with=_ids(((run.finish or {}).get("handoff") or {})
                                                           .get("conflicts_with")))
        if not run.ok or run.finish is None:
            status = next((s for s in ("timeout", "max_turns", "backend_unavailable") if run.error.startswith(s)),
                          "failed")                  # v3-c7i I7: max_turns is the session's budget, not an outage
            return out(ResearchResult(status, None, run.error or "no_finish"))
        if tools.final_reject:                          # the bounce limit on a diff intake rejects: no smoke, no commit
            rule = tools.final_reject.split(":", 1)[0].strip() or "rejected"
            diff_text, paths, cfg_diff = self.adapter.diff(wt)
            if rule == "empty_diff" or not diff_text.strip():
                return out(ResearchResult("failed", None, "empty_diff: the agent changed nothing"
                                          + ("; its dataset_spec was still requested"
                                             if run.finish.get("dataset_spec") else "")))
            hy, ho = run.finish["hypothesis"], run.finish["handoff"]
            h = HypothesisRecord(id=f"h-{request.id}", task_id=self.ctx.task_id, kind=HypothesisKind.TRIAL,
                                 claim=claim_text(hy), mechanism_tag=str(hy.get("mechanism_tag") or "").strip(),
                                 family=str(hy.get("family") or ""), predictions={}, falsification=str(
                                     hy.get("falsification") or ""), created_ts=time.time())
            return out(ResearchResult(rule, Candidate(request.id, self.ctx.task_id, "", wt.base_commit, diff_text,
                                                      diff_hash(diff_text), tuple(paths), dict(cfg_diff), h,
                                                      HandoffNote(f"ho-{request.id}", self.ctx.task_id, request.id,
                                                                  str(ho.get("actions") or ""),
                                                                  str(ho.get("learnings") or ""),
                                                                  str(ho.get("concerns") or ""), time.time())),
                                      f"bounce_limit_reject: {tools.final_reject[:400]}"))
        if isinstance(run.finish.get("dataset_spec"), Mapping) and not self.adapter.diff(wt)[0].strip() \
                and not self.dataset_only(store, wt, run.finish):
            tools.dirty = True                         # a dataset_spec-only finish: the config now points at it
        # the session's own smoke counts as the first queueing, unavailable or not (live 2026-10-02: a session's
        # smoke + 2 more here held each researcher 3 h behind a full association)
        smoke = tools.last_smoke if tools.last_smoke and not tools.dirty else self.adapter.smoke(wt)
        tries, polled = 0, 0.0                         # the environment failed: wait out the gate's hold, keep the work
        while getattr(smoke, "infra", False):
            w = float(getattr(self.adapter, "smoke_hold_wait", lambda: 0.0)())
            if w == INFRA.CANARY_POLL_S:               # another session's canary is in flight: poll for its verdict;
                if polled >= INFRA_CANARY_WAIT_S:      # these polls are not one of the INFRA_SMOKE_WAITS tries (live
                    break                              # review: N waiters burned all 3 in seconds, then re-issued)
                polled += w
            elif tries >= INFRA_SMOKE_WAITS:
                break
            else:
                tries += 1
            time.sleep(min(INFRA_WAIT_MAX_S, max(1.0, w + 1.0)))
            smoke = self.adapter.smoke(wt)
        for _ in range(SMOKE_UNAVAILABLE_RETRIES):     # no slot in time: queue again (the work is kept)
            if not getattr(smoke, "unavailable", False):
                break
            smoke = self.adapter.smoke(wt)
        diff_text, paths, cfg_diff = self.adapter.diff(wt)   # after smoke: what is committed is checked
        if not diff_text.strip():
            return out(ResearchResult("failed", None, "empty_diff: the agent changed nothing"
                                      + ("; its dataset_spec was still requested" if run.finish.get("dataset_spec") else "")))
        if wt.base_commit != parent_commit:
            # a crossover's (a debug's) diff is taken against another commit than the first parent: a locked /
            # off-surface file the ANCESTRY changed (a base-epoch drift, e.g. configs/smoke.yaml) is no violation
            # unless the session's own work changed it vs the first parent (10-13: 6 of 14 crossovers failed
            # surface_violation this way) -- and ANY locked path / span that differs from the first parent is one,
            # whatever the base diff says (review M1: restoring probe.py to the merge-base content left no base diff)
            paths = self._own_surface(wt, parent_commit, paths)
        bad = [p for p in paths if any(path_matches(p, pat) for pat in surface.locked)
               or not intake.on_surface(p, surface.writable)]
        if bad:
            return out(ResearchResult("failed", None, f"surface_violation: {bad}"))
        fin, now = run.finish, time.time()        # finish is already schema-validated
        hy, ho = fin["hypothesis"], fin["handoff"]
        h = HypothesisRecord(
            id=f"h-{request.id}", task_id=self.ctx.task_id, kind=HypothesisKind.TRIAL,
            claim=claim_text(hy),
            mechanism_tag=hy["mechanism_tag"].strip(), family=hy["family"],
            predictions={t: AxisPrediction(int(p["sign"]), float(p["size"]))
                         for t, p in hy["predictions"].items()},
            falsification=hy["falsification"], created_ts=now,
            levers=LeverTags({k: int(v) for k, v in fin["levers"].items() if int(v)}))
        handoff = HandoffNote(f"ho-{request.id}", self.ctx.task_id, request.id, ho["actions"],
                              ho["learnings"], ho["concerns"], now)

        def cand(commit: str, ic: Any = None) -> Candidate:
            return Candidate(request.id, self.ctx.task_id, commit, wt.base_commit, diff_text,
                             diff_hash(diff_text), tuple(paths), dict(cfg_diff), h, handoff, ic)
        if getattr(smoke, "unavailable", False):        # backend unavailable: the driver re-issues the request
            return out(ResearchResult("backend_unavailable", cand(""), f"{SMOKE_UNAVAILABLE}: {smoke.reason}"))
        if not smoke.ok:
            return out(ResearchResult("smoke_failed", cand(""), f"smoke: {smoke.reason}"))
        msg = fin.get("commit_message") or f"loop {request.operator.value}: {h.mechanism_tag}"
        commit = self.adapter.commit(wt, message=f"{msg}\n\nrequest={request.id} hypothesis={h.id}")
        ic = self.adapter.init_compat(wt, parent_commit=wt.base_commit)
        if tools.bold:                                  # v3-c8b: step-0 identity relaxed (never the banned inits)
            ic = self._relax(request, store, ic)
        if not ic.ok:
            return out(ResearchResult(ic.rule, cand(commit, ic), f"init_compat: {ic.reason}"))
        return out(ResearchResult("ok", cand(commit, ic)))

    @staticmethod
    def _relax(request: CandidateRequest, store: Any, ic: Any) -> Any:
        """A bold session's init-compat verdict with only the step-0 identity parts relaxed (loop.core.bold)."""
        from loop.core import bold as BOLD
        got = BOLD.relax_identity(ic)
        if got is not ic and hasattr(store, "log"):
            store.log("bold_identity_relaxed", request_id=request.id, was=str(getattr(ic, "reason", ""))[:600])
        return got

    def _plan(self, request: CandidateRequest, fin: Mapping[str, Any], tools: Any, wt: Worktree) -> dict[str, Any]:
        """The finish's first-principles record, idea id, validated variants and branched plan (loop.variants runs
        them); the variants are checked against the config AFTER the session's edit."""
        hy = fin.get("hypothesis") or {}
        rec = {f: str(hy.get(f) or "") for f in IDEAS.FP_FIELDS if hy.get(f)}
        if isinstance(hy.get("mechanism_prediction"), Mapping):
            rec["mechanism_prediction"] = dict(hy["mechanism_prediction"])
        out: dict[str, Any] = {"first_principles": rec} if rec else {}
        if getattr(tools, "need_evidence", False):           # 11-08 evidence gate (intake parks a missing one)
            if getattr(tools, "evidence_missing", ""):
                out["evidence_missing"] = tools.evidence_missing
            if hy.get("evidence"):
                out["evidence"] = list(hy["evidence"]) if isinstance(hy["evidence"], (list, tuple)) else [str(hy["evidence"])]
                out["readout_path"] = str(hy.get("readout_path") or "")
        if getattr(tools, "v3", False) and isinstance(hy.get("mechanism_prediction"), Mapping):
            mp = dict(hy["mechanism_prediction"])
            if (total := budget_total(mp.get("budget"))) is not None:      # the panel claim IS the budget's sum
                if mp.get("size") is not None:
                    mp["declared_size"] = mp["size"]
                mp["size"] = round(total, 6)
            out["mechanism_prediction"] = mp
            rec["mechanism_prediction"] = mp
            if getattr(tools, "below_floor", False):
                out["claim_below_floor"] = True
            if getattr(tools, "claim_mismatch", None):           # accepted on the second try: kept for calibration
                out["claim_mismatch"] = dict(tools.claim_mismatch)
        if getattr(tools, "v3", False) and getattr(tools, "vs_k", 0) > 0 and isinstance(fin.get("candidates"), list):
            from loop.core import calibration as CAL           # v3-c6i: every candidate kept, the chosen one scored
            out["candidates"] = [dict(c) for c in fin["candidates"] if isinstance(c, Mapping)]
            out["choice"] = str(fin.get("choice") or "")
            if (ch := CAL.chosen(fin)) is not None:
                out["chosen"] = int(fin["chosen"])
                out["prediction"] = {k: ch.get(k) for k in ("p_neg", "median", "p_gap")} | {
                    "gap": tools.ref_gap, "gap_kind": getattr(tools, "ref_kind", ""), "unit": "visible_composite"}
                out["mechanism"] = str(ch.get("mechanism") or "")
            if getattr(tools, "vs_problem", ""):
                out["vs_problem"] = tools.vs_problem
        if getattr(tools, "v3", False) and isinstance(hy.get("analogy"), Mapping):
            out["analogy"] = dict(hy["analogy"])
        if getattr(tools, "v3", False) and isinstance(fin.get("components"), list):
            out["components"] = [dict(c) for c in fin["components"] if isinstance(c, Mapping)]
        if hy.get("idea_id"):
            out["idea_id"] = str(hy["idea_id"])
        if hy.get("component"):
            out["component"] = str(hy["component"])
        if not getattr(tools, "bold", False) and \
                (knobs := FAC.declared(fin.get("knobs") if isinstance(fin.get("knobs"), list) else ())):
            # (a bold node's tuning is loop.tune's, within ideation.bold_tune_max: no separate designed wave)
            out["knobs"] = [dataclasses.asdict(k) for k in knobs]
        if isinstance(fin.get("dataset_spec"), Mapping):   # loop.databank builds it (the driver's intake)
            out["dataset_spec"] = dict(fin["dataset_spec"])
        if tools.chosen is not None:
            out["slate"] = len(tools.slate)
        if getattr(tools, "slate_gate", None) is not None and tools.slate_gate.done:   # v3-c8b diverge-then-judge
            built, show, repo = "", None, None
            try:
                built = self.adapter.diff(wt)[0]          # v3 review S3: the BUILT diff's category + novelty
                show, repo = getattr(self.adapter, "_show", None), getattr(self.adapter, "repo", None)
            except Exception:  # noqa: BLE001 - no diff: the slate alone
                built = ""
            got = tools.slate_gate.plan(built, (lambda rel: show(repo, wt.base_commit, rel)) if show and repo else None)
            if "prediction" in got:
                got["prediction"] = {**got["prediction"], "gap": getattr(tools, "ref_gap", 0.0),
                                     "gap_kind": getattr(tools, "ref_kind", ""), "unit": "visible_composite"}
            out.update(got)
        ic = getattr(self.ctx.cfg, "ideation", None)
        if not (fin.get("variants") or fin.get("plan") or fin.get("campaign") or fin.get("search_space")):
            return out
        config: dict[str, Any] = {}
        if rel := self.config_rel():
            try:
                config = flatten(yaml.safe_load((Path(wt.path) / rel).read_text()) or {})
            except (OSError, yaml.YAMLError):
                config = {}
        if fin.get("campaign") or fin.get("search_space"):     # loop.campaign reads these from ideation:<node>
            out.update(campaign=bool(fin.get("campaign")), search_space=IDEAS.validate_space(config, fin.get("search_space")))
        if ic is None or not ic.enabled or not (fin.get("variants") or fin.get("plan")):
            return out
        vs, problems = IDEAS.validate_variants(self.ctx, config, fin.get("variants"), ic.max_variants)
        out.update(variants=vs, branches=IDEAS.validate_plan(fin.get("plan")), problems=problems)
        return out

    def _prompt(self, request: CandidateRequest, store: Any, snapshot: Any, parents: list,
                parent_commit: str, depth: int, merge: str, wt: Worktree | None = None) -> str:
        cfg, surface = self.ctx.cfg, self.ctx.spec.surface
        lines: list[str] = []
        v3 = is_v3(request)
        for n in (p for p in parents if p is not None):
            lines.append(f"{n.id} [{n.family}/{n.mechanism_tag}] commit={n.commit[:10]}" if v3 else
                         briefs.node_line(snapshot, n))
            if not v3 and (post := snapshot.nodes.get(n.id)):
                lines.append(f"  P(U>0)={post.p_positive:.2f} axes: " + ", ".join(
                    f"{a}={q.mean:+.4f}±{q.sd:.4f}" for a, q in post.axes.items()))
            flushed = "handoffs" in request.ablated   # fork-and-flush: the workspace (diff), not its story
            if not flushed and n.hypothesis_id and (h := store.get_hypothesis(n.hypothesis_id)):
                lines.append(f"  hypothesis (that session's own claim, unverified): {h.claim[:400]}")
            lines += [] if flushed else [f"  handoff (self-reported by that session; hypotheses, not facts): "
                                         f"did {ho.actions[:300]} | learned {ho.learnings[:300]} | "
                                         f"concerns {ho.concerns[:300]}" for ho in store.handoffs(node_id=n.id)]
            if d := briefs.read_text(n.diff_path, 6000):
                lines.append("  parent diff:\n" + d)
            if request.operator is Operator.DEBUG:     # the crash itself first, then any other failure
                fails = sorted((r for r in store.runs(node_id=n.id) if r.log_tail and r.status.value != "done"),
                               key=lambda r: r.status.value != "crash")
                lines += [f"  failing run {r.id} ({r.status.value}"
                          + (f"; already relaunched with the OOM mitigation '{r.mitigation}' and it still "
                             f"failed: cut memory in the code" if r.mitigation else "")
                          + f") log tail:\n{r.log_tail[-3000:]}" for r in fails[:1]]
        fam = request.family_hint or (parents[0].family if parents[0] else "")
        pms = [pm for pm in store.postmortems() if pm.node_id in request.parent_ids
               or (fam and (o := store.get_node(pm.node_id)) is not None and o.family == fam)]
        config: dict[str, Any] = {}
        if wt is not None and (rel := self.config_rel()):
            try:
                config = flatten(yaml.safe_load((Path(wt.path) / rel).read_text()) or {})
            except (OSError, yaml.YAMLError):
                config = {}
        bold = v3 and request.brief_kind == "bold"           # v3-c8b: the bold lane's own prompt (loop.core.bold)
        pack = "## Surface\n" + briefs.hard_constraints(self.ctx) + "\n### Families\n" + "\n".join(
            briefs.family_lines(self.ctx, config)) + _bank_block(self.ctx, store) if v3 else ""
        if bold:
            from loop.core import bold as BOLD
            pack = pack.replace(f"- Init compatibility (keeping the current init): {briefs.carried_lesson()}",
                                f"- Init (bold lane): {P.rule('bold_absolute', BOLD.INIT_RULE)}")
        return briefs.render(
            "researcher_bold" if bold else "researcher", task_id=self.ctx.task_id, operator=request.operator.value,
            strategy=request.strategy.value, island=request.island, parent_commit=parent_commit,
            operator_instructions=EXPOSE_TEXT if request.brief_kind == "expose" else
            LEGS_TEXT if request.brief_kind == "legs" else
            OPERATOR_TEXT[request.operator].format(
                depth=depth, cap=cfg.researcher.max_debug, merge=merge,
                other=", ".join(p.id for p in parents[1:] if p is not None)),
            parent_id=request.parent_ids[0], brief=(request.brief or "(none)") + (
                self.precheck(request, store, parents) + self._inflight_block(request, store) if v3 else ""),
            parent_context="\n".join(lines) or "(root baseline: no changes yet)",
            postmortems=briefs.postmortem_block([] if "postmortem" in request.ablated else pms, limit=6),
            context_pack=pack if v3 else
            briefs.context_pack(self.ctx, store, snapshot, config=config, request_id=request.id),
            panel_sd=f"{cfg.panel_sd:.4f}", max_turns=self.limits(request, store)[0],
            timeout_min=f"{self.limits(request, store)[1]}" + (" (the smoke and diagnose tools' wall time does not "
                                                               "count)" if v3 and self._off_clock() else ""), families=sorted(cfg.families), axes=list(cfg.axes),
            sizing=(BOLD_SIZING if bold else V3_SIZING if v3 else MOONSHOT_SIZING
                    if request.strategy is Strategy.MOONSHOT else DEFAULT_SIZING) + self._bar_sizing(),
            web_research=(WEB_BOUNDED.format(n=self._web_n()) if bounded_web(request, self.ctx) else
                          NT.prompt(self.ctx, request.id) if wants_research(request) else WEB_TEXT)
            if wants_web(request, self.ctx) else "",
            novelty="" if v3 else NV.block(self.ctx, store, snapshot, request),
            ideation=self._bold_ideation() if bold else self._ideation(request, store, snapshot, parents),
            record="(see the brief)" if v3 else self._record(request, store, snapshot))

    @staticmethod
    def _bold_ideation() -> str:
        """v3-c8b: the bold lane's first-principles record -- no diagnostics summary, no per-axis budget rows."""
        from loop import diagnostics as DG
        return BOLD_FP.format(quantities=", ".join(sorted(DG.quantities())), axes_example=axes_example())

    def _ideation(self, request: CandidateRequest, store: Any, snapshot: Any, parents: list) -> str:
        """The first-principles procedure, the diagnostics of the parent, the slate (plan-before-code) and the
        variants / branched-plan instructions ('' for a session that is not ideation)."""
        ic = getattr(self.ctx.cfg, "ideation", None)
        if not IDEAS.is_fp_session(request) or not is_v3(request) and (ic is None or not ic.enabled):
            return ""
        from loop import diagnostics as DG
        from loop.agents import dossiers as DS
        pid = request.parent_ids[0] if request.parent_ids else ROOT_NODE_ID
        if is_v3(request) and "(j) Incumbent diagnostics" in (request.brief or ""):
            return briefs.render("ideation", diagnostics=(
                "## Diagnostics\nThe incumbent's diagnostic flags are in the brief (levers section (j)), each with the "
                "prior attempts at the mechanism it implies and their verdicts: a flag is a symptom; a mechanism "
                "already tried for it needs a new answer to those attempts.\n"), slate="",
                quantities=", ".join(sorted(DG.quantities())))
        diag = DS.diag_summary(store, pid)
        slate = SLATE_TEXT.format(lo=ic.min_ideas, hi=ic.max_ideas, top=ic.implement_top) \
            if ic and ic.max_ideas > 1 and IDEAS.is_slate_session(request) and not is_v3(request) else ""
        return briefs.render("ideation", diagnostics=(
            f"## Diagnostics of the parent {pid} (the diagnostic world model, computed)\n{diag}\n" if diag else
            "## Diagnostics\n(no diagnostic summary is available for the parent: build your model from the code, the "
            "run metrics and the ledger)\n"), slate=slate, quantities=", ".join(sorted(DG.quantities())))

    def _record(self, request: CandidateRequest, store: Any, snapshot: Any) -> str:
        """loop.meta's track record (researcher; a crossover also sees the crossover/compound record)."""
        from loop import meta                    # loop.meta -> loop.tune -> loop.agents
        try:
            return meta.track_record(self.ctx, store, snapshot, "researcher", examples=2) + (
                "\n\nYour crossovers / compounds:\n" + meta.track_record(self.ctx, store, snapshot, "crossover", examples=2)
                if request.operator is Operator.CROSSOVER else "")
        except Exception as exc:  # noqa: BLE001 - analytics never blocks a researcher
            return f"(track record unavailable: {type(exc).__name__})"
