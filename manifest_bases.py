#!/usr/bin/env python3
"""manifest_bases.py -- ONE answer to "which artifacts belong to WHICH base".

WHY THIS MODULE EXISTS
----------------------
Every MANIFEST field that describes the cycle's control -- ``baseline_dir``,
``panel``, ``baseline_readout``, ``consolidated_config``, ``train_seed``,
``raw_baseline_commit``, ``preprobe_stack_commit`` -- is a property of ONE BASE.
The MANIFEST spells them at top level, which is exactly right while a cycle has
exactly one base and silently wrong the moment it has two: an iBOT trial scored
against the I-JEPA baseline still produces a signed ``paired_delta``, still gets
committed, still reads as the effect of a knob. That is this project's #1
failure mode -- reporting a number nobody measured -- in its purest form.

So the MANIFEST grows an OPTIONAL per-base section::

    "base_ref": "9b0739d5...",              # the cycle's DEFAULT base
    "baseline_dir": "/data/.../c9-base",    # legacy top level = default's value
    "panel": "/.../batch09/panel.json",
    "bases": {
        "9b0739d5...": {"baseline_dir": ..., "panel": ..., "train_seed": ...},
        "29dac96c...": {"baseline_dir": ..., "panel": ..., "train_seed": ...}
    }

and this module is the ONLY place that reads it. Every caller
(``config.get_baseline_dir``, ``lib/common.resolve_baseline_dir``,
``panel_delta.train_baseline_dir``, ``agents/l3._resolve_infra_panel``) routes
through :func:`base_artifacts`. A second copy of this logic is precisely the bug
class being closed, so there is no tool-side mirror: ``lib/common.py`` imports
this file rather than restating it.

THE TWO RULES THAT MAKE IT SAFE
-------------------------------
1. **Legacy is the DEFAULT base's fallback, and ONLY the default base's.**
   A MANIFEST with no ``bases`` key resolves exactly as it does today, field for
   field -- that is the back-compat contract. With ``bases`` present, the
   top-level value still backs the base named by ``MANIFEST.base_ref``, because
   that is whose value it has always been.

2. **A NON-default base never falls back to top level.** Asking for base B's
   ``baseline_dir`` when ``bases`` has no entry for B returns ``None``
   (UNKNOWN), never base A's dir. Callers already have a loud "nothing
   resolved" branch; sending them base A's artifact instead would manufacture
   the cross-base delta this module exists to prevent.

``None`` therefore always means "not recorded", NEVER "matches" -- the same
discipline ``config_integrity.baseline_dir_base_ref`` states for its own reads.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

__all__ = [
    "BASE_SCOPED_FIELDS",
    "MIN_REF_CHARS",
    "base_artifact",
    "base_artifacts",
    "base_token",
    "dispatch_blocker",
    "dispatch_exclusion_note",
    "is_declared_base",
    "is_dispatchable",
    "partition_dispatchable",
    "require_declared_base",
    "baseline_dir",
    "default_base_ref",
    "is_default_base",
    "known_bases",
    "match_base_key",
    "panel",
    "refs_match",
]

#: Every MANIFEST field whose value belongs to ONE base. A field listed here may
#: appear at top level (the default base's value) and/or under
#: ``bases[<sha>]``. Adding a field here is the whole cost of making it
#: per-base: every reader goes through :func:`base_artifacts`.
#:
#: ``baseline_output_dir`` is the historical alias of ``baseline_dir`` and is
#: carried so :func:`baseline_dir` keeps the exact two-key fallback order the
#: single-base resolvers have always used.
BASE_SCOPED_FIELDS: tuple[str, ...] = (
    "baseline_dir",
    "baseline_output_dir",
    "panel",
    "baseline_readout",
    "consolidated_config",
    "train_seed",
    "reseed_anchor_seed",
    "raw_baseline_commit",
    "preprobe_stack_commit",
)

#: Characters of a git ref kept in :func:`base_token`. Long enough to be
#: collision-free in practice (the same 8 the panel/run names already use for
#: refs) and short enough to keep a run name readable.
BASE_TOKEN_CHARS = 8

#: Shortest prefix accepted as a git ref match. Directions pin abbreviated refs
#: ("9b0739d5"); the MANIFEST records full SHAs. Below this a "prefix match" is
#: a coin flip, and a wrong base match is worse than an unknown one.
MIN_REF_CHARS = 7


def _norm(ref) -> str:
    return str(ref).strip().lower() if ref else ""


def refs_match(a, b) -> bool:
    """True iff two git refs denote the same commit, prefix-tolerantly.

    Equal after normalisation, or one is a >= :data:`MIN_REF_CHARS` prefix of
    the other. Empty/None on either side is NOT a match -- unknown is not equal.
    """
    x, y = _norm(a), _norm(b)
    if not x or not y:
        return False
    if x == y:
        return True
    short, long = (x, y) if len(x) <= len(y) else (y, x)
    return len(short) >= MIN_REF_CHARS and long.startswith(short)


def known_bases(manifest: dict | None) -> tuple[str, ...]:
    """The base keys the MANIFEST carries a per-base section for (may be empty)."""
    bases = (manifest or {}).get("bases")
    if not isinstance(bases, dict):
        return ()
    return tuple(str(k) for k in bases)


def default_base_ref(manifest: dict | None) -> str | None:
    """The cycle's DEFAULT base -- ``MANIFEST.base_ref``, or None."""
    ref = (manifest or {}).get("base_ref")
    return str(ref) if ref else None


def is_default_base(manifest: dict | None, base_ref=None) -> bool:
    """True iff ``base_ref`` names the cycle's default base.

    ``None`` means "the caller did not say", which IS the default base -- that
    is what every single-base call site means today.
    """
    if base_ref is None or not _norm(base_ref):
        return True
    return refs_match(base_ref, default_base_ref(manifest))


def match_base_key(manifest: dict | None, base_ref=None) -> str | None:
    """The ``bases`` key for ``base_ref``, or None when it does not resolve.

    Exact match wins. Otherwise a UNIQUE prefix match (either direction) is
    accepted; an AMBIGUOUS one returns None rather than guessing -- two bases
    that share a prefix is exactly when guessing costs a measurement.
    """
    bases = (manifest or {}).get("bases")
    if not isinstance(bases, dict) or not bases:
        return None
    ref = base_ref if base_ref is not None else default_base_ref(manifest)
    if not _norm(ref):
        return None
    for key in bases:
        if _norm(key) == _norm(ref):
            return str(key)
    hits = [str(k) for k in bases if refs_match(k, ref)]
    return hits[0] if len(hits) == 1 else None


def is_declared_base(manifest: dict | None, base_ref=None) -> bool:
    """Does this MANIFEST actually SAY anything about ``base_ref``?

    True for the default base (the top level is its declaration) and for any
    ref with a ``bases`` entry. False for a ref the MANIFEST has never heard of.

    This exists because rule 1 and rule 2 have a GAP between them, and a live
    dry-run walked straight into it: a LEGACY manifest (no ``bases`` key at all)
    returns its top-level artifacts verbatim for ANY ``base_ref`` -- which is
    exactly right for the single-base readers rule 1 was written for, and
    exactly wrong for a caller naming a second base. `main.py reseed --base-ref
    29dac96c` against the live c9 MANIFEST derived iBOT-named runs carrying
    I-JEPA's panel, train_seed 902206, anchor 674785 and baseline dir: two full
    trainings whose every number would have been attributed to the wrong base.

    So every WRITER that takes a base_ref gates on this first. Rule 2's "never
    inherit" only bites once a ``bases`` section exists; this makes it bite when
    one does not.
    """
    if is_default_base(manifest, base_ref):
        return True
    return match_base_key(manifest, base_ref) is not None


def require_declared_base(manifest: dict | None, base_ref=None,
                          what: str = "this operation") -> None:
    """Raise unless :func:`is_declared_base` -- the one refusal, one message."""
    if is_declared_base(manifest, base_ref):
        return
    known = list(known_bases(manifest))
    raise ValueError(
        f"{what}: MANIFEST declares no bases[{base_ref!r}] entry "
        f"(declared bases: {known or 'none — this is a legacy, top-level-only '
                                     'MANIFEST'}). A MANIFEST without an entry "
        f"for a base returns the DEFAULT base's panel, seeds and baseline dir "
        f"verbatim, so proceeding would run under {base_ref!r}'s name on "
        f"{default_base_ref(manifest)!r}'s artifacts. Declare "
        f"MANIFEST.bases[{base_ref!r}] with its own panel / train_seed / "
        f"reseed_anchor_seed first.")


def base_token(manifest: dict | None, base_ref=None) -> str | None:
    """The filesystem/run-name token that DISAMBIGUATES a non-default base.

    ``None`` for the cycle's DEFAULT base -- and that ``None`` is the whole
    back-compat contract for every namer that calls this: a single-base cycle
    (and the default base of a dual-base one) keeps the names it has always
    had, byte for byte. Only a SECOND base earns a token, so its artifacts are
    additive on disk and can never land on the first base's path.

    The token is the ref's leading :data:`BASE_TOKEN_CHARS` alphanumerics, so
    it is safe in a path, a branch name and a WANDB run name alike.
    """
    if is_default_base(manifest, base_ref):
        return None
    key = match_base_key(manifest, base_ref) or base_ref
    slug = "".join(c for c in _norm(key) if c.isalnum())
    return slug[:BASE_TOKEN_CHARS] or None


def base_artifacts(manifest: dict | None, base_ref=None) -> dict:
    """THE resolution function: every :data:`BASE_SCOPED_FIELDS` value for ONE base.

    Returns a dict with EVERY field present, its value or ``None`` for UNKNOWN,
    so a caller can never mistake "absent" for "inherited from the other base".

    ``base_ref=None`` asks for the cycle's default base, which is what every
    existing single-base call site means. A MANIFEST with no ``bases`` key
    resolves to the top-level fields verbatim -- byte-identical to today.
    """
    man = manifest if isinstance(manifest, dict) else {}
    legacy = {f: man.get(f) for f in BASE_SCOPED_FIELDS}

    bases = man.get("bases")
    if not isinstance(bases, dict) or not bases:
        return legacy  # legacy MANIFEST: unchanged, field for field

    key = match_base_key(man, base_ref)
    entry = bases.get(key) if key is not None else None
    if not isinstance(entry, dict):
        entry = {}

    # RULE 2: only the DEFAULT base inherits the top-level values. A non-default
    # base with no recorded field is UNKNOWN, never the other base's artifact.
    out = dict(legacy) if is_default_base(man, base_ref) else {
        f: None for f in BASE_SCOPED_FIELDS}
    for field in BASE_SCOPED_FIELDS:
        val = entry.get(field)
        if val is not None:
            out[field] = val
    return out


def base_artifact(manifest: dict | None, field: str, base_ref=None):
    """One field of :func:`base_artifacts` (thin accessor, no separate logic)."""
    return base_artifacts(manifest, base_ref).get(field)


def baseline_dir(manifest: dict | None, base_ref=None) -> str | None:
    """The paired-baseline output dir for a base, or None.

    Keeps the historical two-key order (``baseline_dir`` then the
    ``baseline_output_dir`` alias) that ``config.get_baseline_dir`` and
    ``lib/common.resolve_baseline_dir`` have always used -- now scoped to a base.
    """
    arts = base_artifacts(manifest, base_ref)
    for key in ("baseline_dir", "baseline_output_dir"):
        val = arts.get(key)
        if val:
            return str(val)
    return None


def panel(manifest: dict | None, base_ref=None) -> str | None:
    """The measurement-grid panel path for a base, or None.

    The panel is EVAL IDENTITY: two bases compared on different panels are not
    comparable at all, so this is per-base for the same reason baseline_dir is.
    """
    val = base_artifact(manifest, "panel", base_ref)
    return str(val) if val else None


# ---------------------------------------------------------------------------
# CAN this base actually be dispatched against?
# ---------------------------------------------------------------------------
#
# DECLARED is not DISPATCHABLE. ``MANIFEST.bases`` is a declaration: it is what
# ``_pinned_snapshot`` pins and what ``rollover`` refuses to let drift, so a
# base stays declared across a rollover whether or not this cycle can measure
# anything on it. What a TRIAL needs is narrower and physical: its own panel
# (eval identity) and its own paired baseline dir (the thing every delta is
# signed against), both actually on disk. Rule 2 is why the gap is real -- a
# non-default base never inherits the default's panel or baseline dir, so
# "declared, artifacts unrecorded" resolves to None, not to the other base's
# artifacts, and every trial dispatched against it would void.
#
# c10 is the live case: ``ab26e19a`` (iBOT) is declared, carries its own seeds
# and raw_baseline_commit, and has NO panel and NO baseline dir, because
# minting one needs a frontier checkpoint trained on its OWN objective that
# does not exist. Borrowing I-JEPA's is exactly the cross-objective artifact
# mix ``require_declared_base`` exists to refuse.


def dispatch_blocker(manifest: dict | None, base_ref=None) -> str | None:
    """WHY this base cannot be dispatched against, or ``None`` when it can.

    A string, not a bool, because the whole point is that a base holding no
    budget must be able to say why it holds none. The checks are physical --
    the recorded path must EXIST -- since a panel recorded but never minted
    voids a trial exactly as thoroughly as one never recorded.
    """
    if not is_declared_base(manifest, base_ref):
        return "MANIFEST declares no bases[...] entry for it"
    p = panel(manifest, base_ref)
    if not p:
        return "no per-base panel recorded (rule 2: it inherits none)"
    if not os.path.exists(p):
        return f"panel recorded but not on disk: {p}"
    d = baseline_dir(manifest, base_ref)
    if not d:
        return "no per-base baseline_dir recorded (rule 2: it inherits none)"
    if not os.path.isdir(d):
        return f"baseline dir recorded but not on disk: {d}"
    return None


def is_dispatchable(manifest: dict | None, base_ref=None) -> bool:
    """True iff :func:`dispatch_blocker` finds nothing in the way."""
    return dispatch_blocker(manifest, base_ref) is None


def partition_dispatchable(manifest: dict | None,
                           bases: Sequence) -> tuple[list, dict]:
    """Split ``bases`` into ``(dispatchable, {base: blocker_reason})``.

    BOTH halves are returned, for the same reason
    ``base_join.partition_records`` returns both: a caller that silently shrank
    its set would hide the very fact that matters. Order is preserved, so any
    downstream split stays deterministic and resume-stable.

    Never raises: a fault resolving one base makes that base UNDISPATCHABLE
    with the exception as its reason, rather than taking the caller down.
    """
    ok: list = []
    blocked: dict = {}
    for b in list(bases or ()):
        try:
            why = dispatch_blocker(manifest, b)
        except Exception as e:  # noqa: BLE001
            why = f"base artifacts could not be resolved: {e}"
        if why is None:
            ok.append(b)
        else:
            blocked[b] = why
    return ok, blocked


def dispatch_exclusion_note(blocked: Mapping | None) -> str:
    """One line a report prints when bases were excluded from a budget split.

    Empty when nothing was excluded, so a caller can append it unconditionally.
    """
    if not blocked:
        return ""
    parts = "; ".join(f"{str(b)[:12]}: {why}" for b, why in blocked.items())
    return (f"[{len(blocked)} declared base(s) hold NO trial budget because "
            f"they cannot be dispatched against -- {parts}. The declaration "
            f"stands (rollover still pins it); what is withheld is only the "
            f"budget, which would otherwise be reserved for trials that can "
            f"never run and would end the cycle unspent.]")
