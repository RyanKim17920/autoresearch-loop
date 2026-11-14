"""kernel.model — THE typed vocabulary. Pure data plus :meth:`TaskSpec.load`.

WHY THIS EXISTS. Two facts about the current tree decide the shape of this
module.

FIRST: every threshold this system decides on is a bare literal in Python, and
the literals disagree with each other. ``same_seed_sd`` is written down twice
with a 1.8x gap; the graduation bar is written down twice (``GRADUATION_BAR``
and ``SUSPECT_THRESHOLD``); the trial budget is written down twice and the two
copies were 60 and 120 while the operator read the 120. ``kernel/stats.py``
collapsed the noise numbers onto one model that reads a spec. This module is the
spec: every remaining threshold becomes a DECLARED FIELD of :class:`TaskSpec`,
and the value lives in a task YAML. Nothing here is a source of truth about a
number -- the defaults exist so that ``TaskSpec()`` reproduces today's behaviour
exactly, and ``tests/test_kernel_model.py`` asserts each one against the live
constant it was transcribed from. A default that drifts from its constant is a
test failure, not a comment that went stale.

SECOND: 174 lineages died on ``spec_invalid_kill`` and 353 on ``anchor_missing``
-- specs that named things that were not there. pydantic is adopted (plan
section 6) for exactly three properties, not for tidiness:

  * ``extra="forbid"`` + ``ValidationError.errors()[*].loc`` IS the "refuse and
    name the offending key" behaviour that the schema-grounding pass currently
    hand-rolls, in a form the LLM-facing loop can print verbatim;
  * ``model_json_schema()`` generates the schema shown to the author, so the
    prompt and the validator cannot drift -- a documented contributor to those
    174 kills;
  * a ``field_validator`` rejecting non-finite makes ``meas-nan-rejection`` a
    TYPE property rather than a runtime check somebody must remember to call.
    ``min(inf, nan) -> +inf`` is how a crashed eval once became a cycle's best
    draw; a NaN cannot now be constructed into a :class:`SlotScore` at all.

WHAT IS DELIBERATELY NOT HERE.

No ``TaskSpec.adapter()``. The plan's own section 9 calls a lazy import inside
that method what it is: adapters import these types, so a method that imports an
adapter is a real cycle hidden from static analysis by the same device the
section forbids elsewhere. ``adapter`` is a STRING here; ``cli.py`` resolves it
once and passes the resolved object DOWN as a parameter.

No I/O beyond :meth:`TaskSpec.load` reading one YAML file. No subprocess, no
logging config, no imports from anywhere in this repo outside ``kernel``.
"""

from __future__ import annotations

import ast
import math
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import (BaseModel, ConfigDict, Field, ValidationError,
                      field_validator, model_validator)


# ---------------------------------------------------------------------------
# YAML loading — duplicate keys are a REFUSAL
# ---------------------------------------------------------------------------
#
# MOVED, not rewritten, from config_integrity.py, which plan section 6 says
# "survives verbatim". It survives by moving down a layer: TaskSpec.load needs
# exactly this refusal, and kernel/ may not import config_integrity, so the
# loader lives here and config_integrity imports it back. One copy, same class.
#
# The docstring below is kept byte-identical too, including its two example keys,
# which are the trained repo's rather than this layer's. Rewording it would make
# the move un-diffable for the sake of two nouns in a comment; the layering test
# reads the task vocabulary that would actually leak (a task name, a locked path)
# and neither of those is one.


class DuplicateKeyError(ValueError):
    """A YAML mapping declared the same key twice (safe_load would hide it)."""


class UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that raises on duplicate keys in ANY mapping node.

    yaml.safe_load silently keeps the LAST duplicate; a re-declared /
    mis-nested ``dino:`` block or a doubled ``kde_concentration`` is exactly
    the corruption BUG 1 describes, so we surface it instead of hiding it.
    """

    def construct_mapping(self, node, deep=False):  # type: ignore[override]
        seen: set = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise DuplicateKeyError(str(key))
            seen.add(key)
        return super().construct_mapping(node, deep)


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(text: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` in a task spec's text.

    Machine-specific values (repo and state paths, SLURM account / partition)
    live in the environment, not in the committed spec. A reference to an unset
    variable with no default is left as written, so the spec still parses and the
    path check that reads it names the missing variable.
    """
    def sub(m: "re.Match[str]") -> str:
        val = os.environ.get(m.group(1))
        if val:
            return val
        return m.group(2) if m.group(2) is not None else m.group(0)
    return _ENV_REF.sub(sub, text)


def load_yaml(path: Path | str) -> Any:
    """Parse one YAML document, refusing duplicate keys.

    Used by :meth:`TaskSpec.load`. A spec that declares ``bar_sigmas`` twice is
    a spec whose author does not know what the system will run, and keeping the
    last copy is the failure mode, not the fix. ``${VAR}`` / ``${VAR:-default}``
    references are expanded first (:func:`expand_env`).
    """
    return yaml.load(expand_env(Path(path).read_text()), Loader=UniqueKeyLoader)


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _finite(value: float, what: str) -> float:
    """REFUSE NaN/inf, and refuse a bool wearing a float's type.

    ``bool`` is a subclass of ``int``, so ``True`` would otherwise validate as
    the score 1.0. A measurement that is a flag is not a measurement.
    """
    if isinstance(value, bool):
        raise ValueError(f"{what} is a bool, not a measurement")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} is not finite: {value!r} — that is not a "
                         f"measurement")
    return number


# ---------------------------------------------------------------------------
# UNMEASURED — a scale the task has NOT measured, and every use of it REFUSES
# ---------------------------------------------------------------------------
#
# See :class:`Noise` for the argument. The mechanics, and why the sentinel is a
# VALUE rather than a ``None`` every caller has to remember to test for:
#
# There are ~20 places in this tree that read a sigma, and they do not go
# through one accessor. Some ask ``kernel.stats.NoiseModel`` for a threshold
# (``noise.floor()``, ``noise.accept_bar(n)``); others reach straight into the
# spec and multiply — ``policy/gates.py`` does ``sd = ctx.spec.noise
# .seed_transfer_sd`` and then ``rule.sigmas * sd``. A ``None`` sentinel makes
# the first group refuse and the SECOND GROUP CRASH with "unsupported operand
# type(s) for *: 'float' and 'NoneType'", a message that names neither the
# field nor the spec nor the fix, from a call site that never opted in to the
# check. And a reader who fixed only the accessors would leave every direct
# multiplier still reading a sigma it was never given.
#
# So the sentinel carries its own refusal. Every arithmetic and ordering
# protocol a threshold computation could reach for is bound to :meth:`_refuse`,
# which raises :class:`UnmeasuredNoise` naming (a) the field, (b) the spec that
# declined to measure it, and (c) the measurement that would resolve it. That
# makes "refuse rather than compute" a property of the VALUE: a consumer added
# next year inherits the refusal without anybody remembering to guard it, which
# is the only version of this that stays true.
#
# It is deliberately NOT float-coercible. ``__float__`` refuses too, so
# ``float(sd)``, ``sum(...)`` and ``json.dumps`` cannot launder it back into a
# number. The one thing it does support is ``==`` against another sentinel —
# answering "is this declared?" is not a threshold computation — plus ``repr``,
# so a log line can print what it saw.


class UnmeasuredNoise(ValueError):
    """A threshold was computed from a scale the task never measured."""


#: The YAML token. One spelling, matched case-sensitively: ``unmeasured`` in a
#: spec is a typo, and a typo that quietly means something is how this defect
#: started.
UNMEASURED = "UNMEASURED"

#: Leaf field name -> the experiment that produces the number. Printed in every
#: refusal, because a developer who hits this needs to learn WHAT TO RUN, not
#: merely that something is missing. Keyed on the leaf so one recipe serves both
#: the load-time refusal (field absent) and the use-time one (field UNMEASURED).
MEASUREMENT_RECIPES: Mapping[str, str] = {
    "same_seed_sd": (
        "take ONE trained artifact and re-run its EVALUATION n>=10 times; the "
        "sd of the resulting paired deltas is the number (this costs no "
        "training)"),
    "seed_transfer_sd": (
        "train n>=10 models from ONE fixed config at n DIFFERENT TRAINING "
        "seeds, evaluate each against the same baseline; the sd of the "
        "resulting paired deltas is the number. The spread across UNTRAINED or "
        "differently-configured models is a different quantity and does not "
        "substitute"),
    "replicate_sd": (
        "re-run ONE config end to end, training included, n>=10 times; the sd "
        "of the resulting paired deltas is the number"),
    "floor_sigmas": (
        "this is a POLICY multiplier in sigma units, not a measurement -- "
        "choose the sigma multiple below which this task calls a delta noise"),
    "inert_tolerance": (
        "this is an ABSOLUTE epsilon, not a sigma -- choose the config-value "
        "difference below which a change is numerically inert for this task"),
}


def _missing_noise_fields(exc: "ValidationError") -> tuple[str, ...]:
    """The ``noise`` leaves pydantic reported as absent, in declared order.

    TWO SHAPES, AND THE FIRST IS THE ONE THAT MATTERS. A spec with a partial
    ``noise:`` block yields one error per missing leaf (``loc`` is
    ``("noise", "replicate_sd")``). A spec with NO ``noise:`` block at all --
    which is what both of the specs that were silently inheriting looked like
    -- yields a single error at ``("noise",)``, and reporting that one verbatim
    would say "noise: Field required" to an author who needs to be told which
    five fields and what to run for each. So the whole-block case expands to
    every declared field.
    """
    missing: list[str] = []
    whole_block = False
    for err in exc.errors():
        loc = tuple(err.get("loc", ()))
        if err.get("type") == "missing" and loc == ("noise",):
            whole_block = True
            continue
        if (err.get("type") == "missing"
                and len(loc) == 2 and loc[0] == "noise"):
            missing.append(str(loc[1]))
            continue
        # ANYTHING ELSE WRONG WITH THE SPEC AND WE SAY NOTHING. The noise
        # message below is long and specific, and substituting it for
        # pydantic's would hide a misspelt key three blocks away behind a
        # lecture about calibration. A spec with two problems gets the report
        # that names both.
        return ()
    if whole_block:
        return tuple(Noise.model_fields)
    return tuple(missing)


def _undeclared_noise_message(path: object, missing: tuple[str, ...]) -> str:
    """SILENCE IS NOT A STATE — say so, name the fields, say what to run.

    This replaces pydantic's "Field required" with the only thing that helps:
    which task file, which scales, what experiment produces each of them, and
    the fact that ``UNMEASURED`` is an available answer. Before this, the same
    spec loaded clean and ran on another task's calibration.
    """
    lines = [
        f"{path} declares no value for {len(missing)} required noise "
        f"field(s): {', '.join(missing)}.",
        "",
        "A task spec that says NOTHING about a noise scale used to inherit "
        "one -- the first task's measured sigmas were the kernel's defaults, "
        "so silence resolved to another task's calibration and every "
        "sigma-denominated threshold (the graduation bar, the noise floor, "
        "the screen null, the verdict margin, the first-trial kill bars) was "
        "computed from numbers this task never measured, with nothing logged. "
        "Silence is therefore a REFUSAL now, and each field needs an explicit "
        "answer:",
        "",
    ]
    for field in missing:
        recipe = MEASUREMENT_RECIPES.get(field, "declare a value")
        lines.append(f"  noise.{field}: {recipe}.")
    lines += [
        "",
        f"If this task genuinely has not measured one of the three sigmas, "
        f"write `{UNMEASURED}` as its value. That is a real, declarable "
        f"position: it records the gap on the record and makes every "
        f"sigma-denominated consumer REFUSE to compute rather than quietly "
        f"borrow a number. What it must never be is left blank.",
    ]
    return "\n".join(lines)


class Unmeasured:
    """A declared-absent scale. Any attempt to compute with it REFUSES.

    Constructed only by :class:`Noise`'s validator, from the literal
    :data:`UNMEASURED` in a task YAML. ``spec`` is stamped in by
    :meth:`TaskSpec.load` through pydantic's validation context, so the refusal
    can name the file whose author has to fix it.
    """

    __slots__ = ("field", "spec")

    def __init__(self, field: str, spec: str = "<spec constructed in Python, "
                                               "no file>") -> None:
        self.field = field
        self.spec = spec

    def __repr__(self) -> str:
        return f"UNMEASURED({self.field})"

    def __hash__(self) -> int:
        return hash((Unmeasured, self.field))

    def __eq__(self, other: object) -> bool:
        # The ONE safe question — "did the task declare this?" — is not a
        # threshold computation, so it answers instead of refusing.
        return isinstance(other, Unmeasured) and other.field == self.field

    def __bool__(self) -> "Any":
        # Refuse the truthiness shortcut too. ``if sd:`` is a caller deciding
        # whether to apply a threshold, and "no" is the wrong answer: the right
        # one is that this caller may not decide at all.
        return self._refuse("take the truth value of")

    def _refuse(self, operation: str) -> "Any":
        recipe = MEASUREMENT_RECIPES.get(
            self.field.rsplit(".", 1)[-1],
            "declare a measured value for it in the task spec")
        raise UnmeasuredNoise(
            f"{self.field} is declared {UNMEASURED} by {self.spec}, so this "
            f"system REFUSES to {operation} it. A sigma-denominated threshold "
            f"read off some other task's calibration is not a threshold. "
            f"TO RESOLVE: {recipe}; then write that number under "
            f"`{self.field}:` in {self.spec}. Until then every threshold "
            f"denominated in {self.field} is undefined, and a gate that cannot "
            f"compute its threshold REFUSES rather than passes.")

    def _arith(self, *_args: object) -> "Any":
        return self._refuse("compute a threshold with")

    def _order(self, *_args: object) -> "Any":
        return self._refuse("compare a measurement against")

    # Every protocol a threshold computation could reach for.
    __mul__ = __rmul__ = __add__ = __radd__ = _arith
    __sub__ = __rsub__ = __truediv__ = __rtruediv__ = _arith
    __neg__ = __pos__ = __abs__ = __round__ = _arith
    __float__ = __int__ = _arith
    __lt__ = __le__ = __gt__ = __ge__ = _order


#: A declared noise scale: a finite positive number, or a task saying out loud
#: that it has not measured one. ABSENT is neither, and does not load.
Measured = float | Unmeasured


def _spec_name(info: Any) -> str:
    """Which spec file is being validated, for a refusal that names it.

    ``TaskSpec.load`` passes the path through pydantic's validation context;
    a :class:`Noise` built directly in Python has none, and says so rather
    than pointing a developer at a file that is not the one at fault.
    """
    context = getattr(info, "context", None) or {}
    return str(context.get("spec") or "<spec constructed in Python, no file>")


class _Frozen(BaseModel):
    """Immutable, and an undeclared key is a REFUSAL that names itself.

    ``extra="forbid"`` is the load-bearing half: it is what makes
    ``ValidationError.errors()[*].loc`` point at the offending key instead of
    the value being silently dropped and the trial running something nobody
    declared.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Outcome — the four terminal states of a trial
# ---------------------------------------------------------------------------


class Outcome(str, Enum):
    """What a trial produced. FOUR states, and the distinctions are measured.

    SCORED  — a number exists and is paired against a baseline.
    VOID    — no number exists. 93.6% of dead lineages are here. VOID IS NOT
              DEAD: the lineage may be re-opened (policy/reopen.py).
    INERT   — it ran, it scored, and the diff changed nothing that computes; the
              number is real but it is a measurement of the baseline.
    RUNNING — dispatched, not yet terminal. Distinct from VOID on purpose: a
              driver that died mid-wave leaves RUNNING rows, and re-typing them
              as voids is how 908 trials were once written off.
    """

    SCORED = "scored"
    VOID = "void"
    INERT = "inert"
    RUNNING = "running"


# ---------------------------------------------------------------------------
# Finding — what a PRE-REGISTERED kill condition said about one hypothesis
# ---------------------------------------------------------------------------


class Finding(str, Enum):
    """The read-out of one :class:`Hypothesis` against one :class:`Measurement`.

    FOUR states, and the fourth is the one the system keeps re-learning the hard
    way. ``INSTRUMENT_NOT_YET_INFORMATIVE`` says the RUN failed, not the idea:
    the trial voided, or it ran and the diff computed nothing, so no reading of
    the hypothesis exists. It is deliberately NOT ``REFUTED`` and deliberately
    not ``INCONCLUSIVE`` — ``orchestrator.py`` fabricating ``score=0.0`` on
    voids is this same confusion in a different costume, and it polluted every
    mean and every parent selection downstream. Keeping it a separate word is
    what stops a future amendment quietly reading an instrument failure as a
    clean negative.

    A kill condition may return the first three and MAY NOT return the fourth:
    whether the instrument worked is decided by the outcome of the run, never
    by the author's own predicate.
    """

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INSTRUMENT_NOT_YET_INFORMATIVE = "INSTRUMENT_NOT_YET_INFORMATIVE"


#: The only Python node types a pre-registered kill condition may use.
#:
#: A WHITELIST, and the interesting property is not "it declares no import" but
#: that there is no ``Call``, ``Attribute`` or ``Subscript`` in it: no
#: expression in this grammar can reach anything outside the two numbers handed
#: to it. That is what makes it safe to ``exec`` a predicate an LLM wrote, which
#: is the whole point of committing the kill condition as CODE before the run
#: rather than as the prose that every one of the archive's 984 distinct
#: ``if_fail`` texts is.
_KILL_GRAMMAR: frozenset[type] = frozenset({
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Expr,
    ast.If, ast.Return, ast.Compare, ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.USub, ast.UAdd, ast.Not, ast.BinOp, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.Name, ast.Load, ast.Constant,
    ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
})

#: What a kill condition is allowed to conclude. See :class:`Finding`.
_KILL_RETURNS: frozenset[str] = frozenset(
    f.value for f in Finding if f is not Finding.INSTRUMENT_NOT_YET_INFORMATIVE)


def _check_kill_condition(source: str) -> str:
    """REFUSE a kill condition that cannot be run, or that could run anything.

    This is a ``field_validator`` and not a runtime check for the same reason
    ``_finite`` is: a pre-registration whose predicate turns out to be
    unrunnable AFTER the GPU spend is not a pre-registration. An unparseable or
    out-of-grammar condition cannot be constructed into a :class:`Hypothesis`
    at all, so there is no code path in which one reaches a measurement.
    """
    text = str(source).strip()
    if not text:
        raise ValueError(
            "a kill condition is CODE committed before the run, and this one "
            "is empty — the archive's 1,074 if_fail texts are prose, and "
            "prose is a condition no reader can evaluate, which is the "
            "defect this field replaces")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ValueError(f"kill condition does not parse: {exc}") from None
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != "verdict"):
        raise ValueError(
            "a kill condition is exactly one function named 'verdict' and "
            "nothing else")
    args = tree.body[0].args
    if (len(args.args) != 2 or args.posonlyargs or args.kwonlyargs
            or args.vararg or args.kwarg or args.defaults):
        raise ValueError(
            "verdict takes exactly two positional parameters — the low and "
            "high ends of the interval the measurement is read at")
    returned: set[str] = set()
    for node in ast.walk(tree):
        if type(node) not in _KILL_GRAMMAR:
            raise ValueError(
                f"{type(node).__name__} is not in the kill-condition grammar. "
                f"A kill condition compares its two arguments against literals "
                f"and returns a Finding; it calls nothing and reads nothing")
        if isinstance(node, ast.Return):
            value = getattr(node.value, "value", None)
            if not isinstance(node.value, ast.Constant) or not isinstance(value, str):
                raise ValueError(
                    "every return in a kill condition is a literal Finding name")
            returned.add(value)
    unknown = sorted(returned - _KILL_RETURNS)
    if unknown:
        raise ValueError(
            f"kill condition returns {unknown}, which is not one of "
            f"{sorted(_KILL_RETURNS)}. In particular a predicate may not "
            f"declare its own instrument uninformative: whether the run "
            f"produced a reading is decided by the outcome, not by the author")
    if not returned:
        raise ValueError("a kill condition that returns nothing kills nothing")
    return text


# ---------------------------------------------------------------------------
# Verdict — what a gate decided, and on WHAT
# ---------------------------------------------------------------------------


class Verdict(_Frozen):
    """One gate's decision, carrying the evidence for it.

    The four evidence fields are not decoration. Phase-4 measured that 178
    ``gate_blocked`` rows carry a reason string and nothing else: no failed
    axis, no input, no predicate version. That is why the audit could not say
    whether 19 leakage voids were real leaks, and why 172 rows had to be
    re-typed in bulk. A verdict that records only WHY makes the next audit
    exactly as hard as this one.

    ``axis`` is the specific check inside the gate that fired (a gate is
    routinely multi-axis: "the eval gate judged the mechanism" and "the node was
    busy" are the same ``gate_blocked`` today). ``input_digest`` identifies WHAT
    was judged — a diff hash, a config hash — so the row can be re-checked later
    against the artifact. ``predicate_version`` dates the rule, so a re-typing
    pass can tell "the gate changed" from "the trial changed".
    """

    gate: str
    ok: bool
    stage: str = ""
    axis: str = ""
    input_digest: str = ""
    predicate_version: str = ""
    detail: str = ""


# ---------------------------------------------------------------------------
# Candidate -> TrialSpec -> Claim
# ---------------------------------------------------------------------------


class Hypothesis(_Frozen):
    """The research standard: the four things an idea must carry to be answerable.

    FIELD PRESENCE WAS NEVER THE PROBLEM; EVALUABILITY WAS. Measured here over
    the live archive (``overfitX/batch{07..11}/ideation/proposals.jsonl``, 1,084
    proposals): a kill condition is present on 1,074 of them and ALL 1,074 are
    prose — 984 distinct strings, the most common of which reads "delta below
    -0.004 or the run voids", i.e. it folds a threshold and an INSTRUMENT
    FAILURE into one sentence with one verdict. An ``expected_uplift`` is
    present on only 9.3%, and every present value is one of two template
    constants: 0.012 on 91 rows and 0.02 on 10. A field that is 100% present
    and 100% boilerplate looks declared and closes nothing. So each field below
    is typed as the thing that makes it EVALUABLE:

    ``claim`` is the falsifiable statement. ``predicted_effect`` is an effect
    SIZE in score units and must be positive — "improves" is not a prediction,
    and a direction cannot be compared against a power curve. ``validation`` is
    the CHEAP step that could refute this before the full run is paid for.
    ``kill_condition`` is committed CODE, not prose:

        def verdict(ci_lo, ci_hi):
            if ci_lo > 0:
                return "SUPPORTED"
            if ci_hi <= 0:
                return "REFUTED"
            return "INCONCLUSIVE"

    ``seeds`` is how many seeds the verdict will be read on, and it is what
    ``kernel.stats.NoiseModel.accept_bar`` is handed to decide whether this
    claim is answerable AT ALL. It defaults to 1 — the screen — and one seed is
    not a replication, so a hypothesis that does not declare a confirm budget
    cannot be supported by anything. That refusal is the point: the m=1 MDE is
    +0.0095, and a claim that predicts less than its own measurement can
    resolve is refused at authoring time rather than after the GPU spend.
    """

    claim: str
    predicted_effect: float
    validation: str
    kill_condition: str
    seeds: int = 1

    @field_validator("claim", "validation")
    @classmethod
    def _is_stated(cls, value: str, info: Any) -> str:
        if not str(value).strip():
            raise ValueError(
                f"{info.field_name} is blank — a declared-but-empty field is "
                f"how this archive came to carry a kill condition on 99% of "
                f"its rows and an evaluable one on none of them")
        return str(value)

    @field_validator("predicted_effect", mode="before")
    @classmethod
    def _is_a_size(cls, value: Any) -> float:
        number = _finite(value, "predicted_effect")
        if number <= 0.0:
            raise ValueError(
                f"predicted_effect is {number!r}: it is an effect SIZE in score "
                f"units, not a direction, and a prediction of zero predicts "
                f"nothing")
        return number

    @field_validator("kill_condition")
    @classmethod
    def _is_executable(cls, value: str) -> str:
        return _check_kill_condition(value)

    @field_validator("seeds")
    @classmethod
    def _is_a_count(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError("seeds is how many seeds the verdict is read on")
        return int(value)


class Candidate(_Frozen):
    """An idea, before anything has been authored for it.

    Fields transcribed from ``shared_types.Direction``, which is the live shape
    an ideator emits; only the ones with a reader survive the move.
    ``requires_code_change`` is a ONE-WAY VETO exactly as Direction documents:
    True blocks config-only suppression, None/False licence it alike.

    ``research`` is the pre-registration and it is a DIFFERENT field from
    ``hypothesis``: the latter is the prose one-liner an ideator emits today,
    the former is the four typed fields that make the idea answerable without a
    human. It is optional here because the archive is full of Candidates that
    predate the standard; ``agent.ideate.validate`` requires it.
    """

    id: str
    family: str = ""
    hypothesis: str = ""
    source: str = "ideator"
    priority: int = 0
    lineage_id: str = ""
    base_ref: str = ""
    mechanism_signature: str = ""
    complexity: str = "standard"
    trial_budget: int | None = None
    requires_code_change: bool | None = None
    duplicate_of: str | None = None
    prior_outcome: dict[str, Any] | None = None
    research: Hypothesis | None = None


class TrialSpec(_Frozen):
    """The PLAN: what will be authored, gated BEFORE any editor call.

    This is the cheap refusal point the graft exists to preserve. ``config_diff``
    and ``code_change`` here are INTENTIONS, and ``pre_author`` gates
    (knobs_resolve, protected_keys, cap_magnitude) judge them for free. What
    downstream evidence reads is the :class:`Claim`, not this.

    ``carried`` / ``revalidation`` are the two escape hatches the seed-identity
    gate already honours: ``carried`` re-scores an OLD artifact and can never be
    a fresh win; ``revalidation`` re-confirms an old mechanism at the CURRENT
    seed, waiving only the baseline-side seed check.
    """

    trial_id: str
    lineage_id: str
    run_name: str
    base_ref: str
    track: str = ""
    seed: int | None = None
    output_dir: str = ""
    marker: str = ""
    config_diff: dict[str, Any] = Field(default_factory=dict)
    code_change: list[dict[str, Any]] = Field(default_factory=list)
    composability: str = "coupled"
    exclusive_group: str = ""
    move_type: str = ""
    carried: bool = False
    revalidation: bool = False


class Claim(_Frozen):
    """What the diff ACTUALLY did — read out of the worktree, never asserted.

    THE GRAFT. 353 lineages died on ``anchor_missing``: specs naming symbols
    that were absent at ``base_ref``. A claim extracted from the file that was
    written cannot name a symbol that does not exist, because it is read out of
    the file that does.

    ``files`` is ``git diff --name-only``, and it is the field the pre-dispatch
    locked-path gate must read. It must NEVER be re-derived from a spec field:
    the gate's current repair still reads ``target_location``, i.e. still a
    self-declared field, and a spec that simply omits the file it edits is
    invisible to it. The diff is not omissible.
    """

    trial_id: str
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    config_keys: tuple[str, ...] = ()
    move_type: str = ""
    composability: str = "coupled"
    exclusive_group: str = ""
    mechanism_signature: str = ""
    diff_digest: str = ""
    summary: str = ""


# ---------------------------------------------------------------------------
# SlotScore — re-homed here from kernel/stats.py, as that module's note says
# ---------------------------------------------------------------------------


class SlotScore(_Frozen):
    """One paired run-vs-base comparison, kept per slot AND in aggregate.

    ``aggregate`` is the WEIGHTED MEAN, i.e. ``sum(w_i * d_i) / sum(w_i)``. With
    the shipped task spec's weights (classification 3, segmentation 2, survival
    2, robustness 1) the denominator is 8 -- the panel is 8 SLOTS, not 12 tasks,
    and a plain mean over tasks is the arithmetic that made robustness look
    three times more important than it is.

    NON-FINITE IS A CONSTRUCTION ERROR, not a downstream check. This is the
    ``meas-nan-rejection`` parity row expressed as a type: there is no way to
    hold a NaN in this object, so there is no code path in which one can win.
    """

    deltas: dict[str, float]
    weights: dict[str, float]
    aggregate: float
    total_weight: float
    unpaired: tuple[str, ...] = ()

    # mode="before", and that is load-bearing: an "after" validator sees values
    # pydantic has ALREADY coerced, by which point ``True`` has become 1.0 and
    # the bool is undetectable. The raw input is the only place the distinction
    # between a measurement and a flag still exists.
    @field_validator("deltas", "weights", mode="before")
    @classmethod
    def _slots_are_measurements(cls, value: Any, info: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {slot: _finite(number, f"slot {slot!r} in {info.field_name}")
                for slot, number in value.items()}

    @field_validator("aggregate", "total_weight", mode="before")
    @classmethod
    def _scalars_are_measurements(cls, value: Any, info: Any) -> Any:
        if not isinstance(value, (int, float)):
            return value
        return _finite(value, str(info.field_name))

    @property
    def n_slots(self) -> int:
        return len(self.deltas)


# ---------------------------------------------------------------------------
# Measurement — the only thing the ledger will accept as evidence
# ---------------------------------------------------------------------------


class Measurement(_Frozen):
    """A trial's terminal record: WHAT was produced, and against WHAT.

    ``score`` and ``outcome`` must agree, and that is CHECKED rather than
    trusted. SCORED and INERT both carry a number -- an inert result is a real
    measurement, of the baseline -- while VOID and RUNNING must carry none: a
    void with a number attached is how a failed run's leftover reading gets read
    as evidence. An all-unpaired score is refused for the same reason: with every
    slot unpaired there is no denominator, and the scarcity of same-seed paired
    deltas -- 351 of the 2,866 rows commit 5 migrated, with more rows unstamped
    than paired -- is why that is a refusal and not a zero. (The figure of 35
    that ``plan.md`` cites is stale by its own source note; see
    ``kernel.ingest.PAIRED_DELTA_CLAIM``.) ``validity`` names the noise
    regime the delta is read at (``kernel.stats`` vocabulary) so the sigma is a
    property of the row, not of whoever later reads it.

    ``failed_axes`` is structured, not a status string, for the reason
    ``shared_types.TrialResult`` already records: ``gate_blocked`` alone cannot
    distinguish "the eval gate judged the mechanism" from "the node was busy",
    and only the first should close a lineage.
    """

    trial_id: str
    lineage_id: str
    cycle: int
    outcome: Outcome
    base_ref: str = ""
    seed: int | None = None
    head_commit: str = ""
    job_id: str = ""
    output_dir: str = ""
    validity: str = ""
    status: str = ""
    score: SlotScore | None = None
    claim: Claim | None = None
    failed_axes: tuple[str, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def _a_score_is_paired_or_absent(
            cls, value: SlotScore | None) -> SlotScore | None:
        if value is not None and value.total_weight <= 0.0:
            raise ValueError(
                "this measurement carries zero paired weight — every slot was "
                "unpaired, so there is no denominator and no delta")
        return value

    @model_validator(mode="after")
    def _the_outcome_and_the_number_agree(self) -> "Measurement":
        carries_a_number = self.outcome in (Outcome.SCORED, Outcome.INERT)
        if carries_a_number and self.score is None:
            raise ValueError(
                f"outcome is {self.outcome.value} but no score is attached — a "
                f"result that produced no number is VOID, and typing it as one "
                f"that did is how a lineage gets closed on evidence nobody has")
        if not carries_a_number and self.score is not None:
            raise ValueError(
                f"outcome is {self.outcome.value} but a score is attached — a "
                f"void's leftover reading is not a measurement of anything")
        return self


class KillRecord(_Frozen):
    """Why a lineage stopped, and whether it may come back.

    ``reopenable`` is the field that separates "we measured this and it lost"
    from "we never measured it": 93.6% of dead lineages produced ``n_valid ==
    0``, i.e. they were killed by infrastructure, not by evidence. A kill record
    that cannot express that difference is how a research direction gets written
    off for a shared-output-dir race.

    ``n_valid`` is the count that decides it, kept on the record so
    ``policy/reopen.py`` does not have to re-derive it from a projection that
    may have moved.
    """

    lineage_id: str
    reason: str
    cycle: int
    n_valid: int = 0
    n_void: int = 0
    reopenable: bool = False
    void_statuses: tuple[str, ...] = ()
    detail: str = ""


# ---------------------------------------------------------------------------
# TaskSpec — the declared blocks
# ---------------------------------------------------------------------------
#
# TWO RULES DECIDE WHETHER A FIELD GETS A DEFAULT HERE.
#
# A THRESHOLD of the decision procedure is task-independent, so it defaults to
# the value the live tree runs today, with the constant it was transcribed from
# named beside it. tests/test_kernel_model.py imports those constants and asserts
# equality, so this file cannot drift from the behaviour it claims to reproduce.
# Where the plan's draft YAML proposed a DIFFERENT number, the live number is
# used and the plan's is recorded in the comment as a pending policy decision --
# a "port" that silently changes a threshold is the failure this exercise exists
# to end.
#
# An INVENTORY of the task -- file paths, config dotpaths, metric names, phase
# names, env var names, output layout -- gets NO default and, where the system is
# unsafe without it, no default AT ALL so the spec cannot load without declaring
# it. That is what keeps ``kernel/`` free of any task's vocabulary, and it is why
# ``slots`` and ``surface`` are required rather than defaulted: a task with no
# declared constitution and no declared score must not load.


class Identity(_Frozen):
    """Phase machine + artifact layout: where a cycle's state lives."""

    manifest: str = "MANIFEST.json"
    stop_key: str = "stop"          # absolute; --force may override a phase, never a stop
    phases: tuple[str, ...] = ()
    #: The SUBSTRINGS of a phase name that REFUSE dispatch, as distinct from
    #: ``phases``, which is the vocabulary. Conflating the two would make
    #: ``run_rounds`` -- the phase in which dispatch is supposed to happen --
    #: block dispatch, so the blocking set is declared separately rather than
    #: derived. Substrings, not exact names, because the live phase strings are
    #: composite (``step0_reseed``, ``consolidating``) and the marker is what a
    #: pending step writes into them.
    #: Ported from ``orchestrator.BLOCKING_PHASE_MARKERS``.
    blocking_phases: tuple[str, ...] = ("reseed", "consolidat", "step0", "reset")
    #: Task inventory: the plan's draft wrote an absolute /data path, but the
    #: live layout resolves a RELATIVE batch dir against the repo. Declared.
    output_root: str = ""

    # -- where the durable store and its dual-written JSON halves live -------
    #
    # ``kernel.ledger`` resolves NO filename: it is handed paths. These four
    # fields are where those paths come from, so the artifact names stay
    # declared beside the layout that owns them instead of being literals one
    # layer below the spec. ``journal`` and ``receipt`` are ``str.format``
    # templates; the placeholders are the only part of the name a caller may
    # vary, which is what stops a second writer inventing a second address for
    # the same artifact.
    #: The SQLite store, relative to ``home``. One file, all cycles.
    ledger_db: str = "ovx.db"
    #: Append-only dispatch journal, relative to the batch dir. ``{cycle}``.
    journal: str = "sdk_journal_cycle{cycle:02d}.jsonl"
    #: The v1 commit ledger, relative to the batch dir. Still dual-written.
    commit_ledger: str = "commit_ledger.json"
    #: Launch receipts, relative to the batch dir. TRIAL-keyed: ``{n}`` is the
    #: trial ordinal, so N trials of one lineage are N files instead of one file
    #: that trial N+1 overwrote. Measured on live batch11: 26 receipt files
    #: against 68 ``trial_dispatched`` events -- 42 launches whose job id,
    #: worktree and config were destroyed by their own successor.
    #:
    #: THE DIRECTORY LEVEL IS WHERE THE ORDINAL GOES, AND THE BASENAME DOES NOT
    #: MOVE. The same basename addresses a DIFFERENT artifact elsewhere in the
    #: tree -- the receipt a runner stamps into a run's own output directory --
    #: and five modules key on it. Renaming the file would have made a change
    #: about trial keying reach an artifact it has no business touching.
    receipt: str = "receipts/{track}/{lineage_id}/trial{n:02d}/launch_receipt.json"
    #: The address receipts had BEFORE the trial keying, kept because every batch
    #: already on disk is written in it. This is a READ-ONLY compatibility
    #: address: ``kernel.ingest`` and the resume-plan binder must still find
    #: batch07-batch12's receipts, and a reader that saw only the new layout
    #: would report every live job in those batches as ORPHAN -- which is the
    #: input to redispatch, i.e. a silent double-launch.
    receipt_legacy: str = "receipts/{track}/{lineage_id}/launch_receipt.json"


class Noise(_Frozen):
    """THE noise model's declared scales. Consumed by ``kernel.stats``.

    NOTHING HERE HAS A DEFAULT, AND THAT IS THE POINT.  Until 2026-09 the three
    sigmas below were defaulted to ``0.0016 / 0.0034 / 0.0016`` -- ONE task's
    measurements, sitting in the task-independent kernel.  The consequence was
    not a stale comment, it was a silent assertion: a spec that declared no
    noise did not get "unknown", it got the first task's calibration, and every
    sigma-denominated threshold in the tree (``decision.bar_sigmas * sd``,
    ``noise.floor_sigmas * sd``, the screen null, the verdict margin, the
    first-trial kill bars, the confirm depth) was then computed from another
    task's numbers with nothing logged.  Two shipped specs were running that
    way when it was found.

    THE THREE STATES, and why a two-state schema was not enough:

    ``seed_transfer_sd: 0.0034``
        DECLARED.  The task measured it; consumers compute.
    ``seed_transfer_sd: UNMEASURED``
        DECLARED ABSENT.  The task states, on the record, that it has not
        measured this scale.  Every consumer REFUSES -- see :class:`Unmeasured`
        -- which is the same discipline ``policy.gates.resolve_gate`` applies to
        an unknown gate name and ``panel_score`` to a cross-eval delta: refuse
        the computation rather than perform a meaningless one.
    key absent entirely
        NOT A STATE.  It is an unfinished spec, and it fails at
        :meth:`TaskSpec.load` naming the field.  Silence must not resolve to a
        number, and it must not quietly resolve to UNMEASURED either: "I have
        not measured this" is a claim its author has to make out loud.

    WHY THE OTHER TWO FIELDS ARE STILL REQUIRED.  ``floor_sigmas`` and
    ``inert_tolerance`` are not measured scales -- the first is a dimensionless
    policy multiplier ("the floor is 1.18 sigma"), the second an absolute
    epsilon -- so neither may be UNMEASURED.  They are still REQUIRED, because
    a default is a default: this class is task-independent and may not answer a
    task's question on its behalf.  A task that adopts the same floor POLICY as
    another task writes the multiplier down; it does not inherit it by omission.
    """

    #: Re-run the SAME trained artifact's evaluation N times; the sd of the
    #: resulting paired deltas. UNMEASURED is legal and makes consumers refuse.
    same_seed_sd: Measured
    #: Train N models from ONE config at N DIFFERENT training seeds, evaluate
    #: each; the sd of the resulting paired deltas. The widest of the three.
    seed_transfer_sd: Measured
    #: Re-run ONE config end to end (training included) N times; the sd of the
    #: resulting paired deltas.
    replicate_sd: Measured
    #: The noise floor AS A MULTIPLE of the sigma a delta is read at. Required,
    #: never UNMEASURED: it is a policy choice in sigma units, not a
    #: measurement, so a task can hold this fixed while refusing the sigma.
    floor_sigmas: float
    #: The absolute epsilon below which a config change is numerically inert.
    #: Not sigma-denominated, so it survives an UNMEASURED sigma.
    inert_tolerance: float

    # ``Unmeasured`` is a plain class, not a pydantic model: it has to be able
    # to refuse ``__float__``, and a BaseModel that refuses coercion is a
    # contradiction. ``arbitrary_types_allowed`` is what lets it stand in the
    # union; the base's ``frozen`` and ``extra="forbid"`` are preserved.
    model_config = ConfigDict(frozen=True, extra="forbid",
                              arbitrary_types_allowed=True)

    @field_validator("same_seed_sd", "seed_transfer_sd", "replicate_sd",
                     mode="before")
    @classmethod
    def _scale_or_declared_absent(cls, value: Any, info: Any) -> Any:
        """Accept a positive finite number, or the literal ``UNMEASURED``.

        A zero or negative sigma is refused for the reason it always was: a
        zero sigma turns every delta into infinite significance, which is a
        louder version of the same defect this field exists to stop. A task
        that cannot measure the scale says ``UNMEASURED`` and gets refusal, not
        a sigma of zero and a gate that passes everything.
        """
        field = f"noise.{info.field_name}"
        spec = _spec_name(info)
        if isinstance(value, Unmeasured):
            return value
        if isinstance(value, str):
            if value == UNMEASURED:
                return Unmeasured(field, spec)
            raise ValueError(
                f"{field} in {spec} is the string {value!r}. The only legal "
                f"non-numeric value is the exact token {UNMEASURED} (case "
                f"sensitive), which means the task declares on the record that "
                f"it has not measured this scale and every sigma-denominated "
                f"threshold must refuse. Anything else is a typo, and a typo "
                f"here silently recalibrates every gate.")
        number = _finite(value, field)
        if number <= 0.0:
            raise ValueError(
                f"{field} in {spec} is {value!r}. A noise scale must be a "
                f"POSITIVE finite number: at zero every delta reads as "
                f"infinite significance and every gate passes. If this task "
                f"has not measured the scale, declare it {UNMEASURED} — that "
                f"makes the gates refuse instead of pass.")
        return number

    @field_validator("floor_sigmas", "inert_tolerance", mode="before")
    @classmethod
    def _policy_number(cls, value: Any, info: Any) -> Any:
        """Required, positive, and NEVER ``UNMEASURED``.

        Neither of these is a measured scale — ``floor_sigmas`` is a
        dimensionless policy multiplier and ``inert_tolerance`` an absolute
        epsilon — so "I could not measure it" is not an available answer. A
        task may keep the same floor POLICY as another task while refusing that
        task's sigma; it just has to write the multiplier down.
        """
        field = f"noise.{info.field_name}"
        spec = _spec_name(info)
        if isinstance(value, (str, Unmeasured)):
            raise ValueError(
                f"{field} in {spec} may not be {UNMEASURED}: it is not a "
                f"measured scale. {MEASUREMENT_RECIPES[info.field_name]}. "
                f"(The sigma it multiplies may well be {UNMEASURED} — that is "
                f"where the refusal belongs, and it composes: a declared "
                f"floor_sigmas times an UNMEASURED sigma still refuses.)")
        number = _finite(value, field)
        if number <= 0.0:
            raise ValueError(f"{field} in {spec} must be a positive finite "
                             f"number, got {value!r}")
        return number


class Screen(_Frozen):
    """The cheap first pass: many candidates, few seeds each."""

    seeds: int = 1
    #: Names the retention rule. ``pareto_per_instance`` is the one
    #: ``kernel.stats.pareto_front`` implements. READ THE CAVEAT BEFORE RELYING
    #: ON IT: the counterfactual study that is cited for it actually evaluated
    #: the UNION OF PER-SLOT ARGMAXES (31 runs), not 8-D non-dominance, which is
    #: 398/489 = 81% of candidates and 33x the survivor budget. Phase-4 section
    #: 6.4 rules that the two must not be conflated and that neither is turned on
    #: until the calibration probe exists. Declared, not relied on.
    keep_by: str = "pareto_per_instance"
    #: 0.0 == exact-equality argmax == exactly the measured rule, so declaring it
    #: changes nothing. Any non-zero value is a claim about per-slot sigma, whose
    #: only estimate rests on 8 df with a 95% CI (0.68x-1.92x) that spans the
    #: verdict. plan section 8.1's ~15 GPU-h probe is what earns a non-zero here.
    margin_sigmas: float = 0.0


class Confirm(_Frozen):
    """The expensive second pass: few candidates, replicated.

    ``seeds`` and ``test`` were declared here from the start and had NO
    consumer: 749 of 753 measurements in this project's history are
    single-seed, and ``n_members: 0`` in three consecutive cycles is what a
    confirm stage that never executed looks like from the graduation gate. The
    three fields below are what ``policy.confirm`` needed in order to become
    that consumer, and every one of them is a number the TASK must own -- a
    literal in ``policy/`` or ``kernel/`` would be the same
    policy-number-in-a-task-independent-module defect ``policy.select.screen``
    already refuses ("hard-coding k here would put a policy number in a module
    that is supposed to have none").
    """

    seeds: int = 4
    test: str = "signed_rank"
    #: How many of ``Screen.confirmable`` the confirm stage may take. ZERO IS
    #: A REAL ANSWER and is the default: it means this task has not wired the
    #: stage, which must stay distinguishable from a cycle that ran it and
    #: confirmed nothing. ``policy.confirm.is_wired`` reads exactly this.
    top_k: int = 0
    #: The DOTTED CONFIG KEY the confirm stage rotates to get a fresh seed.
    #: There is no task-independent name for "the seed": for one task it fixes
    #: which partition of the corpus is held out, for another it initialises
    #: weights, and a core module that guessed would silently confirm at a key
    #: nothing reads. Empty means the stage is not wired.
    seed_key: str = ""
    #: The FIRST fresh seed. Counted up from, skipping anything the screen
    #: already used. Deliberately disjoint from the screening seed: repeating
    #: one seed is not replication, and ``graduation`` counts DISTINCT seeds
    #: precisely so that it cannot be.
    seed_base: int = 0


class Decision(_Frozen):
    """What counts as a result."""

    #: GRADUATION_BAR / seed_transfer_sd == 0.01 exactly. The plan's rounded 2.94
    #: gives 0.009996. NOT a loosening: at current effect sizes this graduates
    #: nothing, and min_members is the route past it, not a smaller bar.
    #: WRITTEN AS THE DIMENSIONLESS MULTIPLIER IT IS. This was spelled
    #: ``0.01 / 0.0034`` — one task's graduation bar over that same task's
    #: seed-transfer sd — which put two of that task's measurements in the
    #: task-independent kernel just as surely as ``Noise`` did. The decimal is
    #: bit-identical (``0.01 / 0.0034 == 2.9411764705882355`` exactly, and
    #: ``2.9411764705882355 * 0.0034 == 0.01`` exactly; both are asserted in
    #: tests/test_kernel_model.py against the shipped spec). What survives here
    #: is the POLICY — "the bar is ~2.94 sigma" — which is task-independent;
    #: the scale it multiplies is not, and lives in the spec.
    bar_sigmas: float = 2.9411764705882355
    min_seeds: int = 2         # graduation.GRADUATION_MIN_SEEDS
    min_members: int = 2       # graduation.GRADUATION_MIN_MEMBERS
    clean_cycles: int = 2      # config.CLEAN_CYCLES_FOR_GRADUATION
    verdict_sigmas: float = 2.0   # interactions.VERDICT_SIGMA (additive band)
    champion_new_seed_evidence: bool = True
    merge_operator: bool = False  # GEPA: merge HURTS on 2 of 3 models
    screen: Screen = Field(default_factory=Screen)
    confirm: Confirm = Field(default_factory=Confirm)


class Slots(_Frozen):
    """The panel decomposition every delta is computed over.

    All five fields are task inventory. ``weights`` is what makes the aggregate
    a weighted mean over 8 SLOTS rather than a plain mean over 12 tasks -- the
    arithmetic that made robustness look three times more important than it is
    -- so a task that declares no weights gets an unweighted mean, explicitly,
    rather than one silently assumed for it.
    """

    score_key: str
    weights: dict[str, float] = Field(default_factory=dict)
    metric_tails: tuple[str, ...] = ()
    panel_members_file: str = ""
    #: The per-slot panel vector's keys inside a run's ``summary.json``, in
    #: panel order. THE ONLY PLACE THE 8 SLOTS ARE NAMED: ``kernel/ingest.py``
    #: takes them as a required parameter because it may not name a task, and
    #: ``policy/select.py`` never names one at all.
    #:
    #: Empty by default, and the default is load-bearing rather than lazy. A
    #: task that declares no vector gets 1-slot migrated rows, and
    #: ``policy.select.retain`` REFUSES a 1-slot matrix instead of silently
    #: running greedy idxmax under the per-instance Pareto name. Declaring the
    #: keys is what earns the rule, and the keys are checked: a run is indexed
    #: only when every one of them is present and finite.
    vector: tuple[str, ...] = ()


class FullSlots(_Frozen):
    """Concurrent GPU-holding trials. ``adaptive`` picks the contended cap."""

    cap: int = 8            # config.TRAIN_FULLS_FREE
    contended: int = 5      # config.TRAIN_FULLS_CONTENDED
    adaptive: bool = True
    external_gate: int = 24  # config.EXTERNAL_LAUNCH_GATE — cap must stay UNDER


class Stall(_Frozen):
    """When to stop a track that is not improving."""

    #: config.STALL_WINDOW. The plan's draft said 8, which is the window that was
    #: REMOVED for killing the train track prematurely; 400 is the bounded value
    #: that replaced it (>3x the largest per-track budget, so the budget gate
    #: fires first on a healthy cycle).
    window: int = 400
    eps_sigmas: float = 1.0
    exclude_infra_voids: bool = True


class Budget(_Frozen):
    """Every limit that moves during a cycle."""

    #: Task inventory (keyed by track name). The plan's draft said 62 for the
    #: train track; 62 was the number the operator READ while the driver enforced
    #: 60 -- config.track_trial_budget exists because of exactly that divergence.
    trials_per_cycle: dict[str, int] = Field(default_factory=dict)
    #: config.MAX_TRIALS_PER_LINEAGE. The plan's draft said 5 — 2.4x tighter, and
    #: Phase-4 section 2.2 rules it a POLICY CHANGE, not a port.
    max_trials_per_lineage: int = 12
    full_slots: FullSlots = Field(default_factory=FullSlots)
    stall: Stall = Field(default_factory=Stall)
    escalation_cap_per_cycle: int = 20   # config.escalation_cap()
    wave_fault_abort: int = 3            # config.WAVE_FAULT_ABORT
    #: llm._PROMPT_CHAR_BUDGETS['opus'], in CHARACTERS. The plan's draft said
    #: 180000 "prompt_tokens"; the live budget is a character count per model,
    #: and silently reinterpreting chars as tokens would cut the budget ~3.3x.
    prompt_chars: int = 600_000


class Reopen(_Frozen):
    """Whether a killed lineage may come back, and what it costs."""

    max_grants: int = 2          # config.LINEAGE_REOPEN_CAP
    grant_ttl_h: float = 1.0     # config.LINEAGE_REOPEN_GRANT_TTL_H
    reclaim_cap: int = 3         # config.LINEAGE_REOPEN_RECLAIM_CAP
    #: A grant is earned only when EVERY failure was infra-class. 1.0 == all of
    #: them; the plan declares it so the bar is visible rather than implicit.
    min_infra_fraction: float = 1.0
    obsolete_after_cycles: int = 1


class Caps(_Frozen):
    """The attrition ceilings. Voids do NOT eat the develop budget."""

    infra_retry: int = 8         # config.INFRA_RETRY_CAP
    spec_invalid_kill: int = 3   # config.SPEC_INVALID_KILL_CAP
    #: config.NO_VALID_PROGRESS_CAP == MAX_TRIALS_PER_LINEAGE + INFRA_RETRY_CAP.
    #: The plan's draft said 13, which follows mechanically from its own 5.
    no_valid_progress: int = 20
    defer_attempt: int = 6       # config.DEFER_ATTEMPT_CAP
    void: int = 3                # config.VOID_CAP
    reopen: Reopen = Field(default_factory=Reopen)


class VoidCluster(_Frozen):
    """When a family of voids becomes worth an ideation call."""

    min_count: int = 3          # run_cycle.VOID_CLUSTER_BLOCKING
    min_fraction: float = 0.20  # run_cycle.VOID_CLUSTER_FRACTION
    #: run_cycle.VOID_CLUSTER_MIN_WINDOW — below this the fraction is not
    #: estimable and the absolute floor is used instead. The plan's trigger block
    #: dropped this literal entirely.
    min_window: int = 12
    window_h: float = 24.0      # config.VOID_TRIGGER_WINDOW_H


class Triggers(_Frozen):
    """Mid-cycle ideation triggers. Five literals, named."""

    void_cluster: VoidCluster = Field(default_factory=VoidCluster)
    #: run_cycle.FRONTIER_REFRESH_EVERY — committed trials between refreshes.
    frontier_refresh_every: int = 5
    #: run_cycle.NO_IMPROVE_BLOCKING — consecutive committed trials, no
    #: improvement. The plan's draft paired the 5 with "no_progress_waves" and
    #: the 3 with "new_evidence_commits", swapping which literal is which.
    no_improve_commits: int = 3


class FirstTrialKill(_Frozen):
    """The score-based early kill. A KILL predicate ONLY, never a promoter.

    The counter-fact bounds it: the 16 lineages whose first scored trial was
    ABOVE the bar also never produced a later delta above +0.0034. A first-trial
    delta separates "certainly dead" from "probably dead" and nothing else.
    """

    #: config.FIRST_TRIAL_KILL_DELTA_2SIGMA == -2 * seed_transfer_sd.
    sigmas: float = -2.0
    min_valid: int = 2           # config.FIRST_TRIAL_KILL_MIN_VALID
    #: config.FIRST_TRIAL_KILL_DELTA_CATASTROPHIC == -5 * seed_transfer_sd, a
    #: SINGLE-trial carve-out with no min_valid. The plan's draft collapsed both
    #: rules into one -4.0 bar, which DROPS the path 4 of 22 measured kills
    #: actually fired on (Phase-4 section 2.2). Kept as two rules.
    catastrophic_sigmas: float = -5.0
    catastrophic_min_valid: int = 1


class Ideation(_Frozen):
    """The novelty check's two declared thresholds. Both were MEASURED here.

    ``duplicate_jaccard`` is layer (b): the token-set overlap on the STATED
    MOTIVATION (ASI-Arch's rule — two ideas with different configs and the same
    rationale are the same idea) above which a proposal is a replay. Swept
    against this project's own 1,084-proposal corpus (881 cross-cycle
    judgements, 51 of them known exact-id replays), the flagged rate is 6.5% at
    EVERY threshold from 0.35 to 0.95 and recall of the known replays is 51/51
    throughout. It only moves below 0.3, where it explodes (9.4% at 0.20, 33.6%
    at 0.15, 88.5% at 0.10) — that is vocabulary overlap, not duplication. 0.5
    sits in the middle of the flat plateau, so the value is chosen where the
    answer does NOT depend on it. A rate that swung with this number would not
    be a finding, and that is exactly why it is swept rather than inherited:
    ShinkaEvolve's 0.95 and OpenEvolve's 0.99 are other people's corpora.

    ``min_motivation_tokens`` is the FAIL-CLOSED floor. Jaccard on a two-word
    motivation is not an estimate of anything, so below this the check REFUSES
    rather than returning "novel". The observed minimum on the real corpus is
    13 tokens and the 10th percentile is 54, so 8 never fires on a genuine
    proposal and always fires on a stub.
    """

    duplicate_jaccard: float = 0.5
    min_motivation_tokens: int = 8


class SweepPolicy(_Frozen):
    """What makes a sweep proposal substantive.

    Replaces a 20-CHARACTER count used as a quality bar: 394 rejections over 20
    distinct proposals, and the inspected sample was entirely infra-repair ideas
    caught by the bare noun "sweep".
    """

    require: str = "mechanism_or_threshold_claim"
    judge: str = "policy.gates:sweep_justification_is_substantive"


class Surface(_Frozen):
    """The constitution: what may be written, and what may never be set.

    THE HIGHEST-CONVICTION CORRECTION IN THE PHASE-4 EVIDENCE. The plan's first
    draft declared four protected keys (``data.dataset_id``, ``train.max_steps``,
    ``train.max_flops``, ...) and MEASURED against the live config they match
    ZERO real keys — porting it would have left 11 protected keys unprotected.
    These 11 are transcribed from ``consolidation/protected_keys.PROTECTED_KEYS``
    and a test asserts set equality.

    ``protected_keys`` and ``cap_ceilings`` are DIFFERENT concepts and the
    distinction is load-bearing: the first says "you may not SET this key" and is
    checked against a declared diff; the second says WHAT THE DECLARED BUDGET IS,
    so a value arriving by any route — an inherited config, a hand-edited
    worktree YAML, a runtime assignment — is judged on its MAGNITUDE. A diff
    check cannot see those routes; this is the second, independent check.

    CEILING, not equality: the smoke leg legitimately SHRINKS the budget. Buying
    a score by training LONGER is the gaming move; training less is not.

    The training-data path is DELIBERATELY ABSENT from every shipped
    ``protected_keys`` list. The training-data path is the CURATION LEVER and
    must stay movable. It cannot be used to cheat because the eval identity above
    it is frozen and the scoring code lives behind ``locked``.

    All three lists are task inventory and carry NO default: a spec that forgot
    to declare its constitution must not load with an empty one. Fail-closed is
    the whole point — the pre-C2 registry's block prefixes matched nothing in the
    live config, and "protected zero real keys" looked exactly like "protected
    everything, nothing tried to breach it".
    """

    writable: tuple[str, ...]
    locked: tuple[str, ...]
    #: EXACT dotpaths, never block prefixes. A key K matches a config dotpath C
    #: iff C == K or C startswith K + "." — segment-safe, so a locked key catches
    #: its nested children but never a sibling leaf that merely shares a prefix.
    protected_keys: tuple[str, ...]
    #: Written as integers. ``1.0e18`` is NOT a number in YAML 1.1 (the exponent
    #: needs an explicit sign), so it would load as the STRING "1.0e18" and every
    #: ceiling comparison would be a comparison against text.
    #:
    #: THE ANNOTATION IS THE CHECK. ``dict[str, int]`` already refuses "1.0e18"
    #: (``int_parsing``) and 1.5 (``int_from_float``), and it names the offending
    #: key in ``errors()[*].loc`` -- which is the whole reason pydantic is here.
    #: A hand-written ``field_validator`` re-asserting "these are ints" stood
    #: here and could never fire: an after-validator runs on ALREADY-COERCED
    #: values, so it saw an int in every case pydantic let through, and never ran
    #: at all in the cases pydantic rejected. It also claimed to catch ``True``
    #: and did not -- pydantic coerces it to 1 before any validator sees it.
    cap_ceilings: dict[str, int] = Field(default_factory=dict)
    no_new_top_level_module: bool = True


class EnvIPC(_Frozen):
    """NOT overrides: the parent->child channel across the dispatch boundary.

    ``whitelist`` is the only set of vars allowed to reach a training
    subprocess; everything else in the namespace is banned rather than
    forwarded. ``strip`` is the set that must never be INHERITED: a stray
    logging-mode var in the driver's shell silently silenced every full run once,
    and the smoke leg kept working, so nothing looked broken.
    """

    whitelist: tuple[str, ...] = ()
    strip: tuple[str, ...] = ()


class Track(_Frozen):
    """One dispatch lane. ``retired`` is a REFUSAL string, never a no-op.

    A retired track whose dispatch silently returned "nothing to do" would be
    indistinguishable from a track that ran and found nothing. History stays
    readable; only dispatch stops.
    """

    dispatch: str = "waves"
    wave_size: int = 8
    lineage_concurrency: int = 26   # TrackConfig.lineage_concurrency (train)
    max_fix_attempts: int = 3       # TrackConfig.max_fix_attempts (train)
    retired: str = ""

    @property
    def is_retired(self) -> bool:
        return bool(self.retired.strip())


class LLMRole(_Frozen):
    """Where one role's judgment is bought."""

    tier: str = "cheap"
    backend: str = ""
    transport: str = ""
    fallback: str = ""
    n: int = 1


class Publish(_Frozen):
    """Volume is not signal. 490 of 636 board submissions came from one sweep."""

    adapter_hook: str = ""
    max_per_cycle: int = 2


class TaskSpec(_Frozen):
    """ONE task, fully declared. ``core/`` contains no literal for any of it.

    ``adapter`` is a dotted ``module:Class`` STRING and stays one. Resolving it
    here would make ``kernel.model`` import an adapter while adapters import
    ``kernel.model`` — a cycle, and hiding it behind a lazy import inside a
    method is the metric-gaming device plan section 9 forbids by name. The
    resolution belongs in ``cli.py``, which nothing imports.
    """

    name: str
    adapter: str
    repo: str
    home: str = ""
    #: Required, not defaulted: a task with no declared scoring surface and no
    #: declared constitution must not load. Everything else has a task-
    #: independent default that reproduces today's behaviour.
    slots: Slots
    surface: Surface
    identity: Identity = Field(default_factory=Identity)
    #: REQUIRED, and every field inside it is too. A ``default_factory=Noise``
    #: stood here until 2026-09 and is precisely how a spec that declared no
    #: noise came to run on the first task's calibration: the factory answered
    #: a question the spec never asked. Silence is now a load failure that
    #: names the field.
    noise: Noise
    decision: Decision = Field(default_factory=Decision)
    budget: Budget = Field(default_factory=Budget)
    caps: Caps = Field(default_factory=Caps)
    triggers: Triggers = Field(default_factory=Triggers)
    first_trial_kill: FirstTrialKill = Field(default_factory=FirstTrialKill)
    ideation: Ideation = Field(default_factory=Ideation)
    sweep_policy: SweepPolicy = Field(default_factory=SweepPolicy)
    env_ipc: EnvIPC = Field(default_factory=EnvIPC)
    #: stage name -> ordered gate names. A gate is a plain
    #: ``Callable[[GateCtx], Verdict]`` in a name->function dict; adapter gates
    #: arrive as ``module:function`` strings. No Gate base class, no registry
    #: module: a stage runner is ~30 lines and a registry is a dict.
    gates: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    #: Which of a launch gate's INDEPENDENT axes JUDGE the mechanism, i.e. whose
    #: failure makes the void eval-class. A launch gate reports one
    #: ``<name>_ok`` boolean per axis and only some of them looked at the
    #: mechanism at all; the rest fail before anything did -- a busy node, a
    #: taken slot, the wrong environment, a nested worktree, an artifact absent
    #: from disk. ``policy.reopen.failure_class`` used to carry ONE task's axis
    #: names as a frozenset literal, which both put that task's benchmark
    #: vocabulary inside a task-independent package and left every other task's
    #: axes untypeable. Names are matched case-insensitively against the axis
    #: names AND against the aliases a tool writes into ``failed_axes``, so a
    #: task declares both spellings it emits.
    #:
    #: THE DEFAULT IS EMPTY, AND THAT IS A DELIBERATE DIRECTION OF FAILURE.
    #: A task that declares nothing gets "no axis judged the mechanism", so a
    #: structured gate block types ``infra`` and the lineage stays re-openable.
    #: The opposite default (every axis judges) is the failure this whole area
    #: exists to end: an eval verdict CLOSES a lineage permanently, and c8
    #: buried ~128 untested directions exactly that way. Typing an eval void as
    #: infra instead costs at most ``caps.infra_retry`` re-runs of a spec that
    #: re-fails identically -- bounded, recoverable, and visible in the journal.
    #: Burying untested work is none of those. Note the unstructured paths are
    #: untouched: ``spec_rejected``, and a ``gate_blocked`` carrying no axes at
    #: all, are still eval by status.
    eval_gate_axes: tuple[str, ...] = ()
    tracks: dict[str, Track] = Field(default_factory=dict)
    llm_roles: dict[str, LLMRole] = Field(default_factory=dict)
    #: Where a role the task does NOT declare in ``llm_roles`` is bought: "qwen" (the loop's historic default for
    #: the cheap judges) or "opus" (every undeclared role goes to Opus; no qwen call is made).
    llm_default_backend: str = "qwen"
    publish: Publish = Field(default_factory=Publish)
    #: The ``loop/`` package's own knobs, kept untyped here on purpose: the
    #: kernel does not know the loop, and ``loop.spec`` owns every key, default
    #: and refusal inside it (unknown key = refusal, same discipline as above).
    loop: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> "TaskSpec":
        """Read one task YAML. Duplicate keys and undeclared keys are REFUSALS.

        The two failure modes this refuses are the two that were actually
        observed: a duplicate key (``yaml.safe_load`` keeps the last copy, so
        the file says one thing and the run does another) and an undeclared key
        (a threshold the author believed they set, silently dropped). The second
        is reported by ``ValidationError.errors()[*].loc``, which names the exact
        path — ``('budget', 'stall', 'windwo')`` — rather than the whole file.
        """
        data = load_yaml(path)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: a task spec must be a YAML mapping, got "
                             f"{type(data).__name__}")
        try:
            # ``context`` is how the spec's own name reaches the field
            # validators, so an UNMEASURED sentinel minted here can later name
            # the file whose author has to resolve it -- twenty call sites
            # downstream, in a process that has long since forgotten the path.
            return cls.model_validate(data, context={"spec": str(path)})
        except ValidationError as exc:
            missing = _missing_noise_fields(exc)
            if not missing:
                raise
            raise ValueError(_undeclared_noise_message(path, missing)) from exc

    def live_tracks(self) -> tuple[str, ...]:
        """Track names that still dispatch. Retirement stops dispatch, not
        history, so a retired track stays in ``tracks`` and stays readable."""
        return tuple(sorted(n for n, t in self.tracks.items() if not t.is_retired))

    def stage(self, name: str) -> tuple[str, ...]:
        """The gates declared for one stage, in order. Unknown stage -> ()."""
        return tuple(self.gates.get(name, ()))
