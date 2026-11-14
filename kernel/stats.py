"""kernel.stats — THE noise model and everything derived from it.

WHY THIS EXISTS. This system decides what is real by comparing a paired delta to
a noise scale, and until now it kept FIVE different answers to "what is the
noise scale" in five files:

    consolidation/interactions.py:57   SAME_SEED_SIGMA         = 0.0009
    consolidation/revalidation.py:95   SAME_SEED_SD            = 0.0016
    consolidation/revalidation.py:94   SEED_TRANSFER_SD        = 0.0034
    config.py:915                      NOISE_FLOOR             = 0.004
    config.py:940/:971                 EXPLOIT_SEED_TRANSFER_SD = 0.0034
                                       EXPLOIT_REPLICATE_SD    = 0.0016

The 0.0009/0.0016 pair is not a rounding difference, it is a 1.8x disagreement
about the same measurement, and it changed verdicts: at 0.0009 a +0.003
interaction reads as 3.3 sigma ("synergy, condense it"); at 0.0016 the same
number reads as 1.9 sigma ("additive, inside the band"). ``agents/briefs.py``
already documented this in prose -- it says outright that quoting the narrower
figure "overstated every same-seed measurement by ~1.8x" -- and then called
``interactions_from_state`` with no sigma override, so the arithmetic went on
using 0.0009 anyway. Fixing the comment did not fix the number. This module is
where the number lives so that there is exactly one of it.

THE MODULE IS NOT THE AUTHORITY ON THE NUMBERS. Every value is read from the
task spec through :meth:`NoiseModel.from_spec`. :data:`DEFAULTS` exists so that
``from_spec({})`` is explainable and so that this commit is a behavioural no-op
everywhere except the one place that was measurably wrong -- it is a fallback,
not a source of truth. A caller that wants a different noise model declares it
in YAML; it never edits this file.

WHAT IS DELIBERATELY NOT HERE. No scipy, no numpy, no I/O, no logging config,
and nothing from this repo above ``kernel`` itself -- the one import is
``kernel.model``, which is a frozen data leaf. :func:`signed_rank_p` enumerates
the exact null distribution rather than calling ``scipy.stats.wilcoxon``,
because at the sample sizes this project actually has (``confirm: {seeds: 4}``;
n=8 is the biggest realistic panel) the exact test is ~15 lines and a normal
approximation at n=4 is not a test, it is a decoration.

There is ONE way to spell every threshold read: build a :class:`NoiseModel` and
call its method. The free functions that used to shadow ``sigma``, ``is_noise``,
``accept_bar`` and ``interaction`` against a module-global are gone; they had no
caller in the tree, and a second spelling of a threshold read is how a call site
comes to disagree with the spec without anybody editing a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# ``UnmeasuredNoise`` is re-exported, not raised here: every refusal in this
# module is raised BY THE SENTINEL, at the arithmetic that wanted the number,
# so that the message can name the computation as well as the field. Callers
# that want to catch it import it from here beside the model it comes out of.
from kernel.model import (UNMEASURED, Decision, Noise, SlotScore, Unmeasured,
                          UnmeasuredNoise)

__all__ = [
    "SAME_SEED", "SEED_TRANSFER", "REPLICATE", "DEFAULTS",
    "SlotScore", "Interaction", "NoiseModel", "UNMEASURED", "UnmeasuredNoise",
    "paired_delta", "signed_rank_p", "pareto_front", "expected_max_sigmas",
]

# ---------------------------------------------------------------------------
# Validity vocabulary
# ---------------------------------------------------------------------------
#
# A delta's SIGMA depends on how it was measured, not on how big it is. Three
# regimes, and the aliases are the vocabularies already in the tree
# (kgraph/db.py:32-35) so a stored row can be handed straight in.

SAME_SEED = "same_seed"           # same parent, same train seed: a paired 2x2 cell
SEED_TRANSFER = "seed_transfer"   # re-measured after the cycle's seed rotation
REPLICATE = "replicate"           # same config re-run at the same seed

_VALIDITY_ALIASES: Mapping[str, str] = {
    "paired_same_seed": SAME_SEED,   # kgraph db.V_PAIRED
    "paired": SAME_SEED,
    "cross_seed": SEED_TRANSFER,     # kgraph db.V_CROSS
    "cross": SEED_TRANSFER,
    # V_UNPAIRED / V_UNKNOWN have no tighter bound than the cross-seed sd, so
    # they are read at seed-transfer scale. This mirrors queries.py's
    # UNTRUSTED_NOISE_FLOOR >= CROSS_SEED_NOISE_FLOOR rule.
    "unpaired": SEED_TRANSFER,
    "unknown": SEED_TRANSFER,
}

# ---------------------------------------------------------------------------
# DEFAULTS — read off kernel.model, NOT restated here
# ---------------------------------------------------------------------------
#
# THIS IS NOT A SECOND TABLE OF THE NUMBERS, and it used to be. Every value is
# taken from ``kernel.model``'s declared defaults, which are the ones
# tests/test_kernel_model.py asserts against the live constants one at a time.
# Restating them here re-created, one layer down, the exact defect this module
# was written to end: the two tables had already drifted apart on
# ``inert_tolerance`` (0.0002 here against the live 1e-4 there), so a caller's
# answer to "how loose is inert" depended on which file it happened to read.
#
# ``DEFAULTS`` is still a FALLBACK and not an authority: it exists so that
# ``from_spec({})`` reproduces ``TaskSpec()``. A task that wants different
# numbers declares them in YAML.
#
# The two multipliers stay unrounded ratios rather than the plan's decimals,
# because the decimals do NOT reproduce the constants they claim to:
# 1.18 * 0.0034 = 0.004012, not NOISE_FLOOR = 0.004, and 2.94 * 0.0034 =
# 0.009996, not GRADUATION_BAR = 0.01. A test asserts both identities.

_DECISION = Decision()

#: THERE IS NO FALLBACK FOR ``noise``, AND THAT IS THE FIX.
#:
#: ``DEFAULTS["noise"]`` used to be ``Noise().model_dump()`` -- the first
#: task's three measured sigmas, reachable by any spec that simply failed to
#: mention them. Two shipped specs reached them. ``kernel.model.Noise`` now
#: declares no defaults at all, so there is nothing here to restate. A spec
#: that wants a noise model declares one; ``TaskSpec.load`` refuses a file that
#: did not, and :meth:`NoiseModel.from_spec` resolves an undeclared key to a
#: value that REFUSES rather than to a number.
#:
#: ``decision`` keeps its fallbacks because every leaf below is DIMENSIONLESS
#: -- a count of seeds, or a multiplier in sigma units. A sigma multiple is a
#: policy this system may hold across tasks; the SCALE it multiplies is a
#: measurement no task may hold on another's behalf. That is the line, and it
#: is the whole reason one of these dicts is empty and the other is not.
DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "noise": {},
    # Only the four leaves this module actually reads; Decision declares more
    # (min_members, clean_cycles, confirm) for consumers that do not exist yet.
    "decision": {
        "bar_sigmas": _DECISION.bar_sigmas,
        "min_seeds": _DECISION.min_seeds,
        "verdict_sigmas": _DECISION.verdict_sigmas,
        "screen": {"margin_sigmas": _DECISION.screen.margin_sigmas},
    },
}


# ---------------------------------------------------------------------------
# SlotScore
# ---------------------------------------------------------------------------
#
# Homed in ``kernel/model.py`` (plan section 2) and re-exported here, because
# ``kernel.stats.SlotScore`` is the name ``paired_delta``'s callers use. Its
# constructor refuses non-finite, so ``meas-nan-rejection`` is a type property;
# ``paired_delta`` keeps its own refusal anyway, because it can name the
# offending SLOT and input value, which a constructor seeing only the computed
# delta cannot.


@dataclass(frozen=True)
class Interaction:
    """A 2x2 factorial gamma with the sigma it was actually read at."""

    gamma: float
    sigma: float
    n_sigma: float
    verdict: str          # synergy | additive | interference


# ---------------------------------------------------------------------------
# NoiseModel
# ---------------------------------------------------------------------------

#: Which leaves each block has, read off the models rather than off DEFAULTS.
#: ``decision`` is deliberately the subset this module reads (see DEFAULTS).
_SECTION_KEYS: Mapping[str, tuple[str, ...]] = {
    "noise": tuple(Noise.model_fields),
    "decision": tuple(DEFAULTS["decision"]),
}


def _section(spec: Any, name: str) -> Mapping[str, Any]:
    """Pull ``spec[name]`` whether ``spec`` is a mapping or an attr object.

    ``from_spec`` has to work today against a parsed-YAML dict and tomorrow
    against the pydantic ``TaskSpec`` of commit 3 without either side caring.
    """
    if spec is None:
        return {}
    section = spec.get(name) if isinstance(spec, Mapping) else getattr(spec, name, None)
    if section is None:
        return {}
    if isinstance(section, Mapping):
        return section
    # The key set comes from the MODEL, not from ``DEFAULTS``. It used to come
    # from ``DEFAULTS[name]``, which worked only while every field had a
    # default to be listed by -- so the moment ``noise`` correctly stopped
    # having defaults, this silently returned {} for a fully-declared TaskSpec
    # and every field read as "not declared". The declared fields of a block
    # are the block's business, and ``model_fields`` is where they are.
    return {k: getattr(section, k) for k in _SECTION_KEYS[name]
            if hasattr(section, k)}


def _field(section: Any, name: str, fallback: Any) -> Any:
    """``section[name]`` whether ``section`` is a mapping or an attr object.

    The merge in :meth:`NoiseModel.from_spec` is SHALLOW, so a spec that
    declares ``decision.screen`` at all replaces the whole default sub-block;
    this reads the leaf against its own default instead of losing it.
    """
    if isinstance(section, Mapping):
        got = section.get(name)
    else:
        got = getattr(section, name, None)
    return fallback if got is None else got


def _is_measurement(value: Any) -> bool:
    """A real, finite number: not a bool, not a string that parses like one.

    ONE predicate, because it was written out four times -- in ``is_noise``,
    ``paired_delta``, ``pareto_front`` and ``interaction`` -- and three of the
    four had already diverged (``interaction`` accepted ``True`` and ``"0.05"``
    as cells, the others refused them). Four copies of "what counts as a
    measurement" is the same defect as five copies of the noise scale, one
    scale down.

    ``bool`` is excluded deliberately: it is a subclass of ``int``, so ``True``
    would otherwise read as the score 1.0.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return math.isfinite(float(value))


def _positive(value: Any, what: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{what} must be a finite positive number, got {value!r}")
    return number


def _non_negative(value: Any, what: str) -> float:
    """Like :func:`_positive` but 0.0 is legal -- and is the meaningful default.

    A zero MARGIN is not a zero sigma: it is exact equality, the rule that was
    actually measured. A negative one would silently invert front occupancy.
    """
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{what} must be a finite non-negative number, "
                         f"got {value!r}")
    return number


def _spec_label(spec: Any) -> str:
    """Best available name for the spec being read, for a refusal to quote."""
    for attribute in ("source_path", "path", "name"):
        value = getattr(spec, attribute, None)
        if value:
            return str(value)
    if isinstance(spec, Mapping) and spec.get("name"):
        return str(spec["name"])
    return "the task spec" if spec is not None else "the default (empty) spec"


def _declared(noise: Mapping[str, Any], field: str, spec: Any) -> Any:
    """Read one ``noise`` leaf. AN ABSENT ONE IS NOT A NUMBER.

    WHERE "SILENCE IS LOUD" LIVES, AND WHY IT IS NOT HERE. A
    :class:`kernel.model.TaskSpec` can never reach this function with a field
    missing -- its loader requires an explicit answer for every noise field and
    refuses the FILE by name if it does not get one. That is the right place
    for the loud failure: a spec has an author and a path, so the refusal can
    tell somebody what to edit.

    ``from_spec`` also accepts a bare mapping, which is how ad-hoc callers and
    tests build a model, and there is no file to name there. So an absent field
    resolves to :class:`Unmeasured` -- NOT to a number, and NOT to a silent
    pass. A caller that omitted a scale gets a model that REFUSES at whatever
    threshold actually needed it, naming the field. The invariant this module
    has to hold is the same in both paths: nothing here ever invents a value.
    """
    if field in noise:
        return noise[field]
    return Unmeasured(f"noise.{field}", _spec_label(spec))


def _scale(noise: Mapping[str, Any], field: str, spec: Any) -> "float | Unmeasured":
    """A measured sigma, or the sentinel, carried through un-coerced."""
    value = _declared(noise, field, spec)
    if isinstance(value, Unmeasured):
        return value
    if value == UNMEASURED:
        return Unmeasured(f"noise.{field}", _spec_label(spec))
    return _positive(value, f"noise.{field}")


def _policy_number(noise: Mapping[str, Any], field: str,
                   spec: Any) -> "float | Unmeasured":
    """``floor_sigmas`` / ``inert_tolerance``: a positive number, or refusal.

    These two are NOT measured scales -- a dimensionless multiplier and an
    absolute epsilon -- so a spec may not declare them ``UNMEASURED``
    (``kernel.model.Noise`` refuses that outright). But an ad-hoc mapping that
    simply omitted one still must not be handed a number, so the sentinel is
    the answer here too, for the same reason as above.
    """
    value = _declared(noise, field, spec)
    if isinstance(value, Unmeasured):
        return value
    return _positive(value, f"noise.{field}")


@dataclass(frozen=True)
class NoiseModel:
    """The declared noise scales, and every threshold derived from them.

    Build it with :meth:`from_spec`. Do not construct thresholds from literals
    anywhere else: the whole point of this type is that ``0.0034`` appears in a
    YAML file and nowhere in Python.
    """

    same_seed_sd: float | Unmeasured
    seed_transfer_sd: float | Unmeasured
    replicate_sd: float | Unmeasured
    floor_sigmas: float | Unmeasured
    inert_tolerance: float | Unmeasured
    bar_sigmas: float
    min_seeds: int
    verdict_sigmas: float
    screen_margin_sigmas: float

    @classmethod
    def from_spec(cls, spec: Any = None) -> "NoiseModel":
        """Read the ``noise:`` and ``decision:`` blocks of a task spec.

        ``decision`` keys fall back to :data:`DEFAULTS`, so a spec may declare
        only the dimensionless policy it disagrees with. ``noise`` KEYS DO NOT
        FALL BACK: there is nothing to fall back to, because a measured scale
        belongs to one task and this module belongs to none.

        An absent noise key, or one declared ``UNMEASURED``, becomes
        :class:`Unmeasured`, which REFUSES at whatever threshold reaches for it
        rather than here -- so the error names the field AND the computation
        that wanted it. A declared value that is not a finite positive number
        is still refused outright: a zero sigma turns every delta into infinite
        significance, which is a louder version of the defect this module
        exists to close, not an escape from it.
        """
        noise = dict(_section(spec, "noise"))
        decision = dict(DEFAULTS["decision"]) | dict(_section(spec, "decision"))
        screen = _field(decision, "screen", DEFAULTS["decision"]["screen"])
        return cls(
            same_seed_sd=_scale(noise, "same_seed_sd", spec),
            seed_transfer_sd=_scale(noise, "seed_transfer_sd", spec),
            replicate_sd=_scale(noise, "replicate_sd", spec),
            floor_sigmas=_policy_number(noise, "floor_sigmas", spec),
            inert_tolerance=_policy_number(noise, "inert_tolerance", spec),
            bar_sigmas=_positive(decision["bar_sigmas"], "decision.bar_sigmas"),
            min_seeds=max(1, int(decision["min_seeds"])),
            verdict_sigmas=_positive(decision["verdict_sigmas"],
                                     "decision.verdict_sigmas"),
            screen_margin_sigmas=_non_negative(
                _field(screen, "margin_sigmas",
                       DEFAULTS["decision"]["screen"]["margin_sigmas"]),
                "decision.screen.margin_sigmas"),
        )

    def sigma(self, validity: str = SEED_TRANSFER) -> float:
        """The sd a delta of this provenance is read at."""
        key = _VALIDITY_ALIASES.get(str(validity), str(validity))
        if key == SAME_SEED:
            return self.same_seed_sd
        if key == REPLICATE:
            return self.replicate_sd
        if key == SEED_TRANSFER:
            return self.seed_transfer_sd
        raise ValueError(f"unknown delta validity {validity!r}")

    def floor(self, validity: str = SEED_TRANSFER) -> float:
        """``|delta|`` below this is noise at that provenance.

        THERE IS NO DEFAULT SPEC AND SO NO DEFAULT FLOOR.  This used to say
        "at the default spec, 0.004", which was true only because a spec-less
        model silently carried one task's sigmas -- the defect this module was
        rewritten to close.  The floor is whatever the SPEC's declared ratio
        and scale multiply to; each shipped spec derives its own from a ratio
        rather than a rounded sigma count, and states why in its own file.  A
        task that declared its scale ``UNMEASURED`` has no floor at all, and
        this raises rather than inventing one.
        """
        return self.floor_sigmas * self.sigma(validity)

    def is_noise(self, delta: float, validity: str = SEED_TRANSFER) -> bool:
        """A non-finite delta is NOT a measurement, so it is noise.

        ``min(inf, nan)`` returning ``+inf`` is how a crashed eval once became
        the best draw in a cycle (consolidation/graduation.py:201-209). Nothing
        in this module lets a NaN through as signal.
        """
        if not _is_measurement(delta):
            return True
        return abs(float(delta)) < self.floor(validity)

    def accept_bar(self, n_seeds: int) -> float:
        """The bar a mean replicated delta must clear on ``n_seeds`` seeds.

        THE BAR DOES NOT MOVE. It is ``bar_sigmas * seed_transfer_sd`` -- 0.01
        at the default spec, i.e. exactly today's ``GRADUATION_BAR`` -- for
        every ``n_seeds >= min_seeds``. It is NOT divided by ``sqrt(n)``:
        replication buys the RIGHT to be believed, not a cheaper threshold, and
        consolidation/graduation.py says so in-tree ("the bar does not move").
        Dividing here would have quietly loosened graduation from 0.01 to 0.0071
        at two seeds while the plan's own comment claims the collapse is "NOT a
        loosening".

        Under-replicated evidence returns ``inf``: unreachable, because one seed
        is not a replication and no delta should be able to clear the bar on it.
        """
        if int(n_seeds) < self.min_seeds:
            return math.inf
        return self.bar_sigmas * self.seed_transfer_sd

    def screen_margin(self, validity: str = SEED_TRANSFER) -> float:
        """The score-unit band to hand :func:`pareto_front` as ``margin``.

        ``decision.screen.margin_sigmas * sigma(validity)``. Read at the
        seed-transfer sd by default because that is the widest scale the spec
        declares and no PER-SLOT sd exists yet -- the measured per-slot spread
        is 2.7x-6.8x the aggregate's, on 8 df, so quoting the aggregate sd here
        would UNDERSTATE the band. The default ``margin_sigmas`` of 0.0 makes
        this 0.0, i.e. exact-equality occupancy; see :func:`pareto_front`.
        """
        return self.screen_margin_sigmas * self.sigma(validity)

    def screen_null(self, n_screened: int,
                    validity: str = SEED_TRANSFER) -> float:
        """The best a screen of ``n_screened`` PURE-NOISE candidates throws.

        ``expected_max_sigmas(n) * sigma(validity)``. This is the number that
        must be printed beside a cycle's best, because a screen reports a
        MAXIMUM and the maximum of n null draws is not zero. At the default
        spec a 62-wide single-seed screen is expected to produce +0.00793 from
        noise alone -- 79% of the 0.01 graduation bar -- and c11's observed
        best of +0.0105 is the only draw above +0.01 in 648 measurements,
        which is what the null predicts rather than what a win looks like.
        """
        return expected_max_sigmas(n_screened) * self.sigma(validity)

    def interaction(self, a: float, b: float, ab: float) -> Interaction:
        """``gamma_AB`` from three deltas measured against a SHARED parent.

        ``a``, ``b`` and ``ab`` are ``Y(P+A)-Y(P)``, ``Y(P+B)-Y(P)`` and
        ``Y(P+A+B)-Y(P)``, so ``gamma = ab - a - b`` -- algebraically identical
        to consolidation/interactions.py's four-cell form, without needing the
        parent cell twice.

        Read at the SAME-SEED sd, because a factorial quad that is not
        same-seed, same-parent is not a contrast at all; enforcing that is the
        caller's job (``interactions.ContrastError``), and this function assumes
        it has already been done.
        """
        for name, value in (("a", a), ("b", b), ("ab", ab)):
            if not _is_measurement(value):
                raise ValueError(f"interaction cell {name} is not finite: {value!r}")
        sigma = self.same_seed_sd
        gamma = float(ab) - float(a) - float(b)
        n_sigma = gamma / sigma
        if n_sigma > self.verdict_sigmas:
            verdict = "synergy"
        elif n_sigma < -self.verdict_sigmas:
            verdict = "interference"
        else:
            verdict = "additive"
        return Interaction(gamma=gamma, sigma=sigma, n_sigma=n_sigma,
                           verdict=verdict)


# ``DEFAULT_NOISE = NoiseModel.from_spec()`` STOOD HERE AND IS DELETED.
#
# It was the defect's distribution mechanism. Built from an EMPTY spec, it was
# a fully-populated noise model carrying one task's measured sigmas, exported
# under a name that reads as "the obvious one to use", and five modules --
# config, two consolidation modules and two agent modules -- imported it and
# never passed a spec. Nothing about that was visible at any call site.
#
# There is no task-independent answer to "what is the noise scale", so this
# module no longer offers one. The v1-specific modules that legitimately want
# v1's numbers now read them from v1's own spec, resolved in ``config.py``
# (``config.TASK_NOISE``) -- the layer that is ALLOWED to name a task, and
# already does, since it owns ``DEFAULT_TASK_SPEC`` and the ``--task``
# selection. Anything else builds a :class:`NoiseModel` from the spec it was
# handed.


# ---------------------------------------------------------------------------
# paired_delta
# ---------------------------------------------------------------------------

def paired_delta(run: Mapping[str, float], base: Mapping[str, float],
                 weights: Mapping[str, float] | None = None) -> SlotScore:
    """Per-slot deltas and the weighted aggregate, for slots present in BOTH.

    A slot measured on the run but not on the base (or vice versa) is UNPAIRED
    and contributes nothing to the aggregate; it is reported by name in
    ``SlotScore.unpaired`` instead of being quietly averaged in. Of this
    project's entire run history only 35 deltas were ever valid paired
    contrasts, so an unpaired slot is named, never absorbed.

    A non-finite score on either side REFUSES the whole comparison. A panel with
    a NaN in it is a broken eval, and the historical failure mode is that the
    NaN wins.
    """
    weights = weights or {}
    deltas: dict[str, float] = {}
    used: dict[str, float] = {}
    unpaired: list[str] = []
    total = 0.0
    for slot in sorted(set(run) | set(base)):
        if slot not in run or slot not in base:
            unpaired.append(slot)
            continue
        pair = (run[slot], base[slot])
        for value in pair:
            if not _is_measurement(value):
                raise ValueError(f"slot {slot!r} carries a non-finite score: "
                                 f"{value!r} — that is not a measurement")
        weight = float(weights.get(slot, 1.0))
        if weight < 0.0:
            raise ValueError(f"slot {slot!r} has a negative weight {weight!r}")
        deltas[slot] = float(pair[0]) - float(pair[1])
        used[slot] = weight
        total += weight
    aggregate = (sum(deltas[s] * used[s] for s in deltas) / total) if total else 0.0
    return SlotScore(deltas=deltas, weights=used, aggregate=aggregate,
                     total_weight=total, unpaired=tuple(unpaired))


# ---------------------------------------------------------------------------
# signed_rank_p — EXACT, by enumeration
# ---------------------------------------------------------------------------

def signed_rank_p(deltas: Sequence[float], alternative: str = "two-sided") -> float:
    """Exact Wilcoxon signed-rank p, by enumerating all ``2**n`` sign patterns.

    Zero deltas are dropped (Wilcoxon's own convention); with none left the
    answer is 1.0, because no evidence is not evidence against the null. Ties in
    ``|delta|`` get AVERAGE ranks and the enumeration runs over those tied
    ranks, which makes this the exact CONDITIONAL permutation test rather than
    the tie-free textbook table -- exact for the null it actually states.

    ``n > 12`` REFUSES rather than falling back to a normal approximation. 4096
    patterns is instant; the point of refusing is that an approximation silently
    substituted for an exact test is how a decision procedure stops being one.
    This project's confirm stage runs 4 seeds and its panel is 8 slots, so the
    ceiling has never been near.

    ``alternative``: ``"two-sided"`` (default) or ``"greater"`` -- the one-sided
    "this candidate beat the base" question.
    """
    kept = [float(d) for d in deltas
            if isinstance(d, (int, float)) and not isinstance(d, bool)
            and math.isfinite(float(d)) and float(d) != 0.0]
    n = len(kept)
    if n == 0:
        return 1.0
    if n > 12:
        raise ValueError(f"signed_rank_p enumerates 2**n patterns and refuses "
                         f"n > 12; got n={n}. Use a different test, do not "
                         f"approximate this one.")
    ranks = _average_ranks([abs(d) for d in kept])
    observed = sum(r for r, d in zip(ranks, kept) if d > 0.0)
    total = sum(ranks)
    centre = total / 2.0
    hits = 0
    for pattern in range(1 << n):
        w = sum(ranks[i] for i in range(n) if pattern >> i & 1)
        if alternative == "greater":
            if w >= observed:
                hits += 1
        elif alternative == "two-sided":
            if abs(w - centre) >= abs(observed - centre) - 1e-12:
                hits += 1
        else:
            raise ValueError(f"unknown alternative {alternative!r}")
    return hits / float(1 << n)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ascending ranks, ties sharing their average. 1-based, as Wilcoxon's are."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# expected_max_sigmas — the null a screen is read against
# ---------------------------------------------------------------------------

def expected_max_sigmas(n: int) -> float:
    """``E[max of n iid N(0,1)]``, in sigmas. The screen's null.

    A screening stage reports the best of ``n`` draws, and the expectation of
    that statistic under a true null is not 0 -- it grows with ``n``. Printing
    it beside the cycle best is the whole of S1's step 4, and it is the one
    line that stops a noise ceiling being read as a result.

    THE PROJECT'S OWN FIGURE OF 2.41 SIGMAS FOR N=62 IS WRONG, and this
    function does not reproduce it. ``docs/prior_art.md`` S1 quotes
    ``E[max of 62] = 2.41 sigma = +0.0082``; ``docs/phase4_evidence.md`` 5.4
    reports a 200,000-draw Monte Carlo giving +0.00793 at sigma 0.0034, i.e.
    **2.332** sigmas. This integral agrees with the MEASURED table, not with
    the quoted multiplier: it returns 2.1803 / 2.3317 / 2.9555 against the
    Monte Carlo's 2.1794 / 2.3324 / 2.9559 for n = 42 / 62 / 384, every one
    inside 0.001 sigma. Phase-4 evidence is authoritative over the older
    document, and the difference matters in the safe direction only by luck:
    2.41 would overstate the null by 0.08 sigma and bury small real effects
    under a ceiling that is not there.

    Computed by Simpson's rule on ``int x n Phi(x)^(n-1) phi(x) dx`` over
    [-9, 12]. No scipy and no numpy, per this module's contract; the closed
    form for n=2 (``1/sqrt(pi)``) comes back exact to 6 decimals and a test
    asserts it. The grid is fixed at 2,000 intervals because the answer stops
    moving in the 6th decimal there -- 100,000 intervals returns the same
    digits -- so a finer grid would buy only runtime.

    ``n=1`` is 0.0: one draw's expected maximum is its expected value, and a
    screen of one has no multiplicity to correct for. ``n < 1`` is a refusal.
    """
    count = int(n)
    if count < 1:
        raise ValueError(f"expected_max_sigmas needs n >= 1, got {n!r}")
    if count == 1:
        return 0.0
    steps, lo, hi = 2000, -9.0, 12.0
    width = (hi - lo) / steps
    total = 0.0
    for i in range(steps + 1):
        x = lo + i * width
        cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        weight = 1.0 if i in (0, steps) else (4.0 if i % 2 else 2.0)
        total += weight * x * count * (cdf ** (count - 1)) * pdf
    return total * width / 3.0


# ---------------------------------------------------------------------------
# pareto_front — per-instance SET-COVER domination
# ---------------------------------------------------------------------------

def pareto_front(matrix: Mapping[str, Mapping[str, float]],
                 keep: int | None = None,
                 margin: float = 0.0) -> list[str]:
    """GEPA's per-instance Pareto survivors. NOT top-N-by-score.

    A candidate is dominated iff EVERY instance-front it occupies also contains
    another SURVIVOR. Sole occupancy of even one front makes it un-prunable, no
    matter how bad its aggregate is; occupying no front at all makes it
    droppable however good its aggregate is (the "every" is vacuously true). The
    loop sorts ASCENDING by aggregate and drops greedily, re-checking against
    the survivors that remain -- so dropping a candidate can PROMOTE a
    previously-redundant one into sole occupancy, and that one is then kept.

    This is the measured difference, not a stylistic one. GEPA Table 3 (ICLR
    2026 Oral, Qwen3-8B, evolution harness held fixed): greedy argmax +6.05,
    beam-search top-N by score +5.11, per-instance Pareto +12.44. Diversity by
    score RANK buys nothing; diversity by instance COVERAGE roughly doubles the
    gain. A top-N implementation of this function would be a different
    algorithm wearing its name.

    ``matrix`` is ``{candidate: {instance: score}}`` and must be RECTANGULAR --
    every candidate scored on every instance. A ragged panel is refused rather
    than imputed: comparing candidates on instance sets that differ is the
    unpaired-contrast mistake in another costume.

    ``keep`` is an optional FLOOR on the survivor count: the loop stops dropping
    once that many remain. It can only make the surviving set larger, never
    smaller -- it is not a cap, because capping would mean evicting a sole
    occupant, which is exactly the eviction this rule exists to forbid.

    ``margin`` IS THE SIGNIFICANCE BAND, in score units, and it is what makes
    the rule usable on a real panel. A candidate occupies an instance's front
    when its score is within ``margin`` of the best on that instance, so a lead
    smaller than the margin buys no sole occupancy and earns no survival.
    ``margin=0.0`` is exact equality -- GEPA's own rule, and the default -- so
    the band is opt-in. Pass ``NoiseModel.screen_margin()``; do not compute one
    here, because this function has no noise model and must not grow one.

    WHY A BAND IS NEEDED AT ALL. Measured over this project's 489 conclusion-
    bearing paired deltas: the literal non-dominated set in 8 dimensions is
    398/489 = 81% of candidates, 33x the survivor budget, i.e. not a selection
    rule. Sole-slot occupancy cuts that to 31 runs (2.6x budget) -- affordable,
    and every one of the 32 (cycle, slot) maxima was a SOLE occupant with zero
    ties, which is precisely why exact equality cannot discriminate: at that
    resolution every winner wins alone, however small its lead.

    THE 8-DEGREES-OF-FREEDOM CAVEAT -- READ BEFORE SETTING A NON-ZERO MARGIN.
    A margin is a claim about per-slot sigma, and the only estimate available
    rests on 8 df from 6 same-config/different-seed reseed groups, one of which
    pools runs whose base scores span 0.00496 -- wider than the seed-transfer
    sigma it is estimating. Per-slot sd runs 0.0022-0.0135, i.e. 2.7x-6.8x the
    aggregate's 0.0020, and the 95% CI on the estimate is 0.68x-1.92x. The
    verdict flips INSIDE that CI: at 1.00x, 12 of 32 slot maxima beat noise; at
    1.25x, 7 of 32; at 1.92x, 2 of 32, against a pure-noise expectation of 1.6.
    An independent pool-MAD estimate already sits at the 1.25x reading. So a
    non-zero margin chosen today is a guess dressed as a threshold.

    THE EXPERIMENT THAT SETTLES IT: 10 runs of ONE config at ONE commit on 10
    DISTINCT seeds, ~15 GPU-h, taking each slot's sd from df=8 to df>=9 per
    group. It is affordable out of the 3,383.8 idle slot-hours the budget audit
    measured, and it is the only thing that converts ``margin_sigmas`` from a
    guess into a number. Until it runs, leave the margin at 0.0.

    Returns survivor ids ordered by descending aggregate, ties by id.
    """
    if not matrix:
        return []
    instances: set[str] | None = None
    for candidate, scores in matrix.items():
        keys = set(scores)
        if instances is None:
            instances = keys
        elif keys != instances:
            raise ValueError(
                f"pareto_front needs a rectangular matrix; candidate "
                f"{candidate!r} is scored on {sorted(keys)} but others on "
                f"{sorted(instances)}")
        for instance, value in scores.items():
            if not _is_measurement(value):
                raise ValueError(f"candidate {candidate!r} instance "
                                 f"{instance!r} score is not finite: {value!r}")
    assert instances is not None
    if not instances:
        raise ValueError("pareto_front needs at least one instance")
    band = _non_negative(margin, "pareto_front margin")

    def aggregate(candidate: str) -> float:
        scores = matrix[candidate]
        return sum(float(scores[i]) for i in instances) / len(instances)

    # Fronts are computed ONCE over the whole pool: "best on this instance" is a
    # property of the measurement, not of who is still in the room. At band 0.0
    # the test is exact equality, as GEPA's is; a declared band widens the front
    # to everyone statistically level with the best, which is the ONLY way sole
    # occupancy stops being an artefact of float resolution.
    fronts: list[frozenset[str]] = []
    for instance in sorted(instances):
        best = max(float(matrix[c][instance]) for c in matrix)
        fronts.append(frozenset(c for c in matrix
                                if float(matrix[c][instance]) >= best - band))

    survivors = set(matrix)
    for candidate in sorted(matrix, key=lambda c: (aggregate(c), c)):
        if keep is not None and len(survivors) <= keep:
            break
        occupied = [f for f in fronts if candidate in f]
        if all((f & survivors) - {candidate} for f in occupied):
            survivors.discard(candidate)
    return sorted(survivors, key=lambda c: (-aggregate(c), c))
