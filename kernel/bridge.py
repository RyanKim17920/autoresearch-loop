"""The one way a v1 JSON writer also lands a typed row in the durable store.

WHY THIS MODULE EXISTS.  Commit 4 dual-wrote EVENTS and nothing else.
``Ledger.record_result`` and ``Ledger.record_receipt`` were written, tested, and
then called by exactly two things: ``kernel.ingest`` (the one-shot migration) and
the tests.  No production writer touched them, so the store's ``results`` and
``receipts`` tables only ever held what the migration put there --- measured at
the time this module was written as 154 rows in ``commit_ledger.json`` against 26
non-test rows in SQLite.  Every commit and every launch since the migration
landed in the JSON and nowhere else.

WHAT IT DOES NOT DO, AND WHY THAT IS THE WHOLE DESIGN.  It does NOT hand the
ledger a ``commit_ledger_path`` or a ``receipt_path``.  The store is opened
SQLITE-ONLY and the existing writer keeps owning its own JSON, unchanged.  That
is the mode :class:`kernel.ledger.Ledger` documents ("a ``Ledger`` handed no
``journal_path`` still records events to SQLite; it simply has no JSON half to
keep in step"), and here it is load-bearing for two measured reasons:

  1. ``Ledger._write_commit_ledger_row`` REPLACES ``commits[run_name]`` with an
     eleven-field row.  A real ``commit_full`` row carries about twenty-five ---
     ``submission_id``, ``run_card``, ``numeric_inertness``, ``provenance``,
     ``base_ref``, ``exp_id``, ``evo_status`` and the external-submission
     verification fields.  Letting the ledger write the JSON half would delete
     them.
  2. That same row sets ``score`` to ``m.score.aggregate``, but ``aggregate`` IS
     the paired delta --- ``kernel.ingest`` builds the slot as
     ``SlotScore(deltas={k: delta}, aggregate=delta)`` and stashes the raw probe
     score in ``provenance["score"]``.  Production keeps the two distinct
     (``score`` 0.6514, ``paired_delta`` 0.0042).  Letting the ledger write the
     JSON half would overwrite ``score`` with the delta.

So the JSON stays exactly as it is, and this module adds the row that was
missing.  ONE definition of the conversion, used by every writer, because the
defect being closed here is precisely what happens when a store has more writers
than it has agreed-on shapes.

UNPAIRED ROWS ARE VOID, NOT ZERO.  ``record_result`` refuses a scored row with
no denominator, and many real commit rows carry ``paired_delta: null``.  Those
become :attr:`Outcome.VOID` with no score, which is the same rule
``kernel.ingest`` applies to the same rows on the migration path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from kernel.ledger import Ledger
from kernel.model import Identity, Measurement, Outcome, SlotScore

#: ``source`` for rows written by a LIVE writer, as against the ``archive``/
#: batch sources ``kernel.ingest`` stamps on migrated rows.  It is part of the
#: trial id, so a live row and a migrated row for the same run never collide.
LIVE_SOURCE = "live"

#: The neutral slot name a single-value score carries.  Must match
#: ``kernel.ingest.AGGREGATE_SLOT``: a row written live and the same row
#: re-imported by the migration have to name their one slot identically, or a
#: reader joining the two sees two different decompositions.
AGGREGATE_SLOT = "aggregate"


def trial_id(source: str, batch: str, run_name: str) -> str:
    """``source:batch:run_name`` --- THE trial-id spelling, for every writer.

    ``kernel.ingest`` delegates to this rather than keeping its own copy.  A
    second spelling of a join key is the failure this repo has already paid for
    twice; there is one function and both paths call it.
    """
    return f"{source}:{batch}:{run_name}"


def receipt_trial_id(source: str, batch: str, lineage_id: str,
                     n: int | None) -> str:
    """The trial id for a LAUNCH receipt, which must carry the ordinal.

    THIS DELIBERATELY DIFFERS FROM THE MIGRATION'S RECEIPT KEY, AND THE
    DIFFERENCE IS THE POINT.  ``kernel.ingest._import_one_receipt`` keys on
    ``exp_id or lineage_id`` with no ordinal, so every trial of a lineage that
    never recorded a distinct ``exp_id`` collapses onto ONE row --- which is the
    lineage-keyed defect the trial-keyed table exists to end, surviving in the
    key while the schema says otherwise.  Migration could not do better: the
    ordinal it would need is what the overwrite destroyed.  A LIVE writer knows
    the ordinal at write time, so it has no such excuse.

    ``n`` of None means the caller genuinely does not know the ordinal (the
    resume-plan binder, which has a live job and a lineage but no trial number).
    Such a receipt keys on the lineage alone --- the honest address for "some
    trial of this lineage" --- rather than silently claiming to be trial 1.
    """
    tail = lineage_id if n is None else f"{lineage_id}/{int(n)}"
    return f"{source}:{batch}:{tail}"


def batch_name(batch_dir: Path | str) -> str:
    """The batch component of a trial id: the batch directory's basename."""
    return Path(batch_dir).resolve().name


def cycle_for(batch_dir: Path | str, ledger: Mapping[str, Any] | None = None,
              explicit: Any = None) -> int:
    """The cycle number for a batch.  ONE rule, shared by every live writer.

    Priority: an explicit ``--cycle`` -> the ledger's own ``cycle`` -> the
    ``batchNN`` in the batch dir name -> 0.  Zero means NOTHING NAMED ONE: the
    store partitions replay by cycle, and inventing a number there is worse than
    an explicit unknown, because a wrong cycle silently moves a row into another
    cycle's replay.

    ``batchNN -> cycle NN`` is a FALLBACK, not an identity.  Batch and cycle have
    coincided in this tree so far, but they are different things; wherever the
    caller has a real cycle it must pass it, and both callers do when they have
    one.
    """
    if explicit:
        parsed = _int(explicit)
        if parsed is not None:
            return parsed
    if isinstance(ledger, Mapping):
        recorded = ledger.get("cycle")
        if isinstance(recorded, int):
            return recorded
    match = re.search(r"batch0*(\d+)", str(Path(batch_dir).resolve().name))
    return int(match.group(1)) if match else 0


def open_store(batch_dir: Path | str, cycle: int,
               layout: Identity | None = None) -> Ledger:
    """Open the durable store for a batch, SQLITE-ONLY.

    The db sits at the state HOME above the batch dir --- the same resolution
    ``journal.Journal`` performs --- so every batch of every cycle writes one
    file.  No JSON path is handed over: see this module's docstring.
    """
    layout = layout or Identity()
    home = Path(batch_dir).resolve().parent
    return Ledger.open(home / layout.ledger_db, cycle=int(cycle))


def _float(value: Any) -> float | None:
    """A finite float, or None.  A NaN is not a number for this purpose."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def measurement_from_commit_row(
        row: Mapping[str, Any], *, batch: str, cycle: int,
        run_name: str | None = None, source: str = LIVE_SOURCE,
        slot_key: str = AGGREGATE_SLOT,
        extra: Mapping[str, Any] | None = None) -> Measurement:
    """Type one ``commit_ledger.json`` row as a :class:`Measurement`.

    The mapping is the migration's, field for field, because the same row read
    off disk later must type identically to the row written here:

      * ``paired_delta`` becomes the one-slot ``SlotScore.aggregate``;
      * the raw ``score`` goes to ``provenance["score"]``, NOT to the slot;
      * no delta means :attr:`Outcome.VOID` with no score attached;
      * a row the numeric-inertness audit flagged is :attr:`Outcome.INERT` ---
        a real measurement OF THE BASELINE, which must neither be dropped nor
        counted as mechanism evidence.
    """
    name = str(run_name or row.get("run_name") or "")
    delta = _float(row.get("paired_delta"))
    raw_score = _float(row.get("score"))
    status = str(row.get("run_type") or "")
    inertness = row.get("numeric_inertness")
    flagged = bool(inertness.get("flagged")) if isinstance(inertness, dict) \
        else False
    # ONLY MEASUREMENT-STABLE FIELDS GO IN THE ROW, AND THAT IS A CORRECTNESS
    # REQUIREMENT, NOT TIDINESS.  ``record_result`` is idempotent on an
    # IDENTICAL payload and REFUSES a different one for the same trial -- "one
    # trial has one answer".  ``commit_full`` legitimately re-commits the same
    # run (``--force``, and the documented "re-running retries the submit" path
    # after an unconfirmed external submission), and each retry carries a fresh
    # ``committed_at`` and possibly a new submission status/id.  Had those been
    # copied in here, EVERY retry would have produced a "different result" and
    # raised, so the wiring would have crashed the committer on its own
    # supported recovery path.
    #
    # The split is therefore: the store holds what was MEASURED; the JSON row
    # holds the submission/commit BOOKKEEPING and stays authoritative for it.  A
    # retry that leaves the number alone is a clean no-op; a retry that CHANGES
    # the number still refuses, which is the rule doing its job.
    provenance: dict[str, Any] = {
        "source": source, "batch": batch, "run_name": name,
        "run_type": status,
        "track": str(row.get("track") or ""),
        "score": raw_score,
    }
    stamped = row.get("provenance")
    if isinstance(stamped, dict):
        provenance["write_time_provenance"] = stamped
    for carried in ("exp_id", "receipt_bypassed"):
        if row.get(carried) is not None:
            provenance[carried] = row[carried]
    if extra:
        provenance.update(extra)

    if delta is None:
        outcome: Outcome = Outcome.VOID
        slot: SlotScore | None = None
    else:
        outcome = Outcome.INERT if (flagged or "inert" in status.lower()) \
            else Outcome.SCORED
        slot = SlotScore(deltas={slot_key: delta}, weights={slot_key: 1.0},
                         aggregate=delta, total_weight=1.0)
    seed = _int((stamped or {}).get("train_seed")
                if isinstance(stamped, dict) else None)
    return Measurement(
        trial_id=trial_id(source, batch, name),
        lineage_id=str(row.get("lineage") or row.get("lineage_id") or name),
        cycle=int(cycle), outcome=outcome,
        base_ref=str(row.get("base_ref") or ""),
        seed=seed,
        head_commit=str(row.get("head_commit") or ""),
        job_id=str(row.get("job_id") or ""),
        output_dir=str(row.get("output_dir") or ""),
        status=status, score=slot,
        # ``failed_axes`` stays EMPTY. It is structured failure attribution --
        # the field that exists so "the eval gate judged the mechanism" can be
        # told apart from "the node was busy", because only the first should
        # close a lineage. A commit row that is VOID here is void for one
        # reason, that it carried no paired delta, and that is not a failed
        # axis. Putting the run_type ("experiment") in this tuple would hand a
        # reader a fabricated axis and let it close a lineage on it.
        provenance=provenance)


def record_commit_row(batch_dir: Path | str, cycle: int, row: Mapping[str, Any],
                      *, run_name: str | None = None,
                      source: str = LIVE_SOURCE) -> Measurement:
    """Land one already-written commit row in the store.  Returns what was typed.

    Call this AFTER the JSON write, inside whatever lock the writer already
    holds, so the two projections cannot interleave differently.
    """
    batch = batch_name(batch_dir)
    measurement = measurement_from_commit_row(
        row, batch=batch, cycle=cycle, run_name=run_name, source=source)
    open_store(batch_dir, cycle).record_result(measurement)
    return measurement


def record_launch_receipt(batch_dir: Path | str, cycle: int,
                          receipt: Mapping[str, Any], *, lineage_id: str,
                          track: str, source: str = LIVE_SOURCE) -> str:
    """Land one already-written launch receipt in the store.  Returns its id."""
    n = _int(receipt.get("trial"))
    batch = batch_name(batch_dir)
    tid = receipt_trial_id(source, batch, lineage_id, n)
    open_store(batch_dir, cycle).record_receipt(
        tid, lineage_id=lineage_id, track=track, cycle=int(cycle), n=n,
        exp_id=str(receipt.get("exp_id") or "") or None,
        job_id=str(receipt.get("job_id") or "") or None,
        worktree=str(receipt.get("worktree") or "") or None,
        output_dir=str(receipt.get("output_dir") or "") or None,
        config=str(receipt.get("config") or "") or None)
    return tid
