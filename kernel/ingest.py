"""kernel.ingest — foreign evidence in, once, without touching the source.

WHAT THIS COMMIT DOES (plan section 8.2, commit 5)
--------------------------------------------------
Reads the v1 archive — eleven ``overfitX/batch*`` directories plus, optionally,
a second-store SQLite — and writes it into :class:`kernel.ledger.Ledger`.  It
replaces the four ``kgraph/ingest_*.py`` ingesters (~1,400 LOC) and the scan
half of ``tools-v2/mine_results.py``.

It exists as a separate module from ``ledger.py`` for the reason the plan gives:
``kgraph/db.py:22-32`` declares FOUR provenance sources, and a store that cannot
say which one a row came from silently joins an external leaderboard entry —
never seed-paired against anything of ours — to a same-seed experiment.  The
write API is one-directional (foreign evidence goes IN) and keeping it out of
the writer is what stops that direction reversing.

WHERE THE FOUR SOURCE NAMES LIVE, AND WHY NOT HERE
---------------------------------------------------
The plan sketches this module's interface as ``migrate_v1`` plus one
``import_<source>`` wrapper per foreign source.  Three of those wrapper names
are a task's proper noun, and ``tests/test_kernel_model.py::
test_kernel_names_no_task`` — shipped by commit 3, on the plan's own "kernel/ is
task-independent" claim — fails on any of them appearing in this directory,
docstrings included.  The guard is right and the sketch was written before it
existed: WHICH provenance labels a deployment has is task inventory, while
"every row carries the label of the store it came from" is the kernel fact.

So the labels are PARAMETERS, required and undefaulted, and the four concrete
values live with the caller (``main.py: SOURCES``) until ``adapters/`` exists at
commit 9.  Undefaulted rather than defaulted: a caller that forgets to name the
source gets a TypeError, where a default would mislabel a whole archive and the
mislabelling is exactly what the discrimination exists to prevent.

THE THREE HARD RULES, AND HOW EACH IS ENFORCED
----------------------------------------------
1. **READ-ONLY toward the archive.**  Every source is opened for reading; the
   SQLite sources are opened through ``file:...?mode=ro``, so a bug that tried
   to write one would raise rather than mutate an irreplaceable store.  Nothing
   here creates, renames or deletes a path under the archive.

2. **IDEMPOTENT.**  Running ``migrate`` twice must not double any row, and the
   three tables get there three different ways: ``results`` and ``receipts`` are
   keyed by ``trial_id`` (a primary key, so the second write is a no-op or a
   refusal — never a duplicate), and ``events`` — an append-only log with no
   natural key — is guarded by ``Ledger.cursor`` / ``record_cursor``, which
   records how many lines of each journal have been consumed.  A journal that
   has GROWN since the last run therefore imports only its new lines.

3. **PROVENANCE SURVIVES.**  Every row carries ``source`` + ``cycle`` + ``seed``
   + ``batch``.  Trial ids are namespaced ``<source>:<batch>:<run_name>`` for a
   measured reason: batch ledgers MIX carried old-cycle commits, so the same
   ``run_name`` is present in two batches with two different cycles.  Keyed by
   run name alone the second import would either overwrite the first or be
   refused as a contradiction; namespaced, both survive and any join across
   cycles has to be written on purpose.

WHAT THE DECLARED SOURCE SET CANNOT ANSWER — STATED, NOT PADDED
---------------------------------------------------------------
The panel aggregate decomposes into EIGHT slots, and that vector is what
per-slot retention would be read from.  It is not in this source set.  MEASURED
here: the eight slot values live only in each run's ``summary.json`` under the
artifact root (``final_probe_{linear,knn,fewshot}_mean_f1``,
``final_probe_seg_mean_jaccard``, ``final_probe_auc_mean``,
``final_probe_slide_mean_auc``, ``final_probe_survival_mean_cindex``,
``final_probe_robustness_mean`` — their unweighted mean reproduces
``final_probe_score`` exactly).  The batch directories carry a DIFFERENT
decomposition, the run card's twelve-leaf ``paired_delta_per_task``, whose mean
does NOT reproduce the recorded panel delta: exact on 2 of 770 run cards, worst
absolute disagreement 0.029.  The two are decompositions of the same panel over
different axes.

So migrated rows get a ONE-slot score whose single value is the recorded panel
delta, and the twelve-leaf vector is preserved verbatim in provenance under
``per_task_delta`` so commit 6 loses nothing.  Fabricating an eight-wide vector
from a twelve-wide one, or letting the aggregate disagree with the slots that
are supposed to average to it, would put a number in the store that no
measurement supports.  ``Report.slot_widths`` reports the width distribution so
the caller sees this rather than inferring it.

THE SOURCE SET IS NOW EXTENSIBLE TO IT, AND THAT IS :func:`slot_vectors`.  The
above stays true of the BATCH DIRECTORIES and is why the width distribution
matters, but the artifact tree is a second, read-only source that does carry
the vector, and reading it costs no GPU because the files are already written.
Pass ``artifact_root`` and ``slot_keys`` to :func:`migrate_v1` and every row
whose run and whose batch baseline BOTH carry a complete vector is imported
eight slots wide, its aggregate recomputed from those slots; every other row
stays one slot wide.  Without those two arguments the migration behaves exactly
as it did.  This is the step that unblocks ``policy.select.retain``, which
refuses a one-slot matrix outright rather than degrading to greedy idxmax.

THE 35-PAIRED-DELTA ASSERTION IS REFUTED — see :data:`PAIRED_DELTA_CLAIM`.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from kernel.ledger import Ledger, LedgerRefusal, UnknownEvent
from kernel.bridge import trial_id as _bridge_trial_id
from kernel.model import Measurement, Outcome, SlotScore
from kernel.stats import paired_delta

# ---------------------------------------------------------------------------
# validity vocabulary — MUST match kgraph/db.py:29-35 verbatim
# ---------------------------------------------------------------------------

#: ``delta_validity``.  Only a trial compared against a SAME-SEED baseline earns
#: ``paired_same_seed``; everything else is weaker, and the difference is the
#: whole reason a banked-gain figure can be positive while the measured
#: root-to-endpoint drift is negative.
V_PAIRED = "paired_same_seed"
V_CROSS = "cross_seed"
V_UNPAIRED = "unpaired"
V_UNKNOWN = "unknown"

#: THE CORRECTNESS CHECK THIS COMMIT WAS ASKED TO ASSERT, AND WHY IT IS NOT
#: ASSERTED AS WRITTEN.
#:
#: ``plan.md:60`` and ``:646`` require the migration to keep "the 35 valid
#: paired deltas" and cite it as the strongest available check.  It is stale by
#: its own source.  The note the figure comes from (2026-07-13, first real
#: kgraph ingest, 1,547 nodes, cycle 6 was the live cycle) records in its own
#: closing lines that 35 was the count BEFORE a provenance fix, and what
#: replaced it: "validity now 103 paired_same_seed / 6 cross / 51 unknown /
#: 1387 unpaired (was 35/6/119)".  35 counted rows whose baseline seed the
#: ingester had failed to resolve, not rows that failed to pair.
#:
#: Three independent measurements now disagree with it and with each other,
#: which is the finding: the live ``kgraph.sqlite`` holds 827 paired rows;
#: ``docs/phase4_evidence.md`` section 4.1 counts 489 conclusion-bearing paired
#: deltas via the ledger + ``summary.json`` path; and this module measures
#: :data:`PAIRED_DELTA_MEASURED` from the declared source set under the rule in
#: :func:`_validity`.  They differ because they resolve seeds differently and
#: read different source sets — not because one is a miscount.
#:
#: The assertion is therefore kept, re-pointed, and pinned to a rule that is
#: written down.  It is NOT relaxed to whatever the code happens to produce:
#: ``tests/test_kernel_ingest.py`` fails on any drift from the measured number,
#: and this constant stays here so nobody re-derives 35 from the plan and
#: "fixes" the migration to match it.
PAIRED_DELTA_CLAIM = 35

#: Same-seed paired deltas recoverable from the eleven live batch directories
#: under :func:`_validity`.  MEASURED 2026-08-29 over batch01..batch12 of the
#: live archive.  Per batch: c7 33, c8 115, c9 66, c10 83, c11 53, c12 1.
#: Batches 04/05 contribute none — c4 carries no resolvable baseline seed and
#: c5's trials ran at a different seed from its baseline (8 cross, 7 unknown),
#: which is itself the c5 finding.
PAIRED_DELTA_MEASURED = 351

#: The neutral name of the single slot a migrated score carries.  A task-level
#: caller passes its own ``slots.score_key``; the default names no task, because
#: ``kernel/`` may not.
AGGREGATE_SLOT = "aggregate"

#: Where a run's per-slot panel vector is written, relative to its artifact
#: directory.  The FILE name is a kernel fact (one JSON summary per run
#: directory); WHICH KEYS inside it are the slots is task inventory and arrives
#: as ``slot_keys``, so this module still names no task.
SUMMARY_FILE = "summary.json"

_SEED_RE = re.compile(r"(?<![0-9])(\d{5,7})(?![0-9])")
_BATCH_RE = re.compile(r"^batch(\d+)$")
#: ``c11-train-p-<stem>`` -> ``<stem>``.  See :func:`_mechanism_stem`.
_STEM_RE = re.compile(r"^c\d+[-_](?:[a-z]+[-_]p[-_])?")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class Report:
    """What a migration actually did, in the shape an assertion can read.

    Counts are separated from SKIPS on purpose.  A migration that silently drops
    rows it could not type reports a clean number and hides the interesting
    half; every refusal here lands in ``skipped`` under the reason for it, so
    "imported 817" and "declined 12 for a non-finite delta" are both visible in
    the same object.
    """

    batches: list[str] = field(default_factory=list)
    events: int = 0
    results: int = 0
    receipts: int = 0
    #: Rows the store ALREADY held.  On a second ``migrate`` this is the whole
    #: archive and ``results``/``receipts`` are zero, which is what idempotent
    #: looks like from the outside; a report that re-counted them would say it
    #: imported 2,866 rows every time it imported none.
    already_present: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    validity: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    by_cycle: dict[str, int] = field(default_factory=dict)
    slot_widths: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: RECOMPUTED-VS-RECORDED AGGREGATE AGREEMENT, counted over exactly those
    #: rows that carry both -- a multi-slot recomputation AND the delta the eval
    #: itself wrote.  This is deliberately a top-level tally and not another
    #: ``provenance`` entry: ``provenance['recorded_delta']`` is per-row and
    #: nothing reads it in aggregate, so a systematic disagreement across a whole
    #: archive looked exactly like a clean import.  A recomputation is a second
    #: implementation of the eval's own composite, and the project's recurring
    #: defect is that the second implementation is wrong and nobody is told.
    aggregate_exact: int = 0
    aggregate_disagreed: int = 0
    aggregate_max_gap: float = 0.0

    @property
    def paired_deltas(self) -> int:
        """Same-seed paired deltas imported.  THE correctness check."""
        return self.validity.get(V_PAIRED, 0)

    @property
    def aggregate_status(self) -> str:
        """Whether the recomputation reproduced the eval's own number.

        BIT-EXACT is the only clean answer and there is no tolerance knob here on
        purpose.  Naming a tolerance would be inventing a threshold, and the
        threshold that matters is not knowable from this module: v1's own
        worst-case recompute gap (2.7e-4) is harmless against a 0.01 graduation
        bar, while v2's hardcode-induced gap sat at the same order as the bar.
        So this reports WHAT WAS MEASURED -- how many rows agreed, how many did
        not, and the worst gap -- and leaves the judging to whoever set the bar.
        """
        if not (self.aggregate_exact or self.aggregate_disagreed):
            return "NOT CHECKED: no row carried both a vector and a recorded delta"
        total = self.aggregate_exact + self.aggregate_disagreed
        if not self.aggregate_disagreed:
            return f"BIT-EXACT on {self.aggregate_exact}/{total}"
        return (f"DISAGREES on {self.aggregate_disagreed}/{total} rows, worst "
                f"gap {self.aggregate_max_gap:.3e} — the recomputed panel "
                f"aggregate is not the composite the eval wrote")

    def weigh_aggregate(self, measurement: Measurement) -> None:
        """Tally one row's recomputed aggregate against the recorded delta."""
        slot = measurement.score
        if slot is None or slot.n_slots < 2:
            return
        recorded = measurement.provenance.get("recorded_delta")
        if not isinstance(recorded, (int, float)) or isinstance(recorded, bool):
            return
        gap = abs(float(slot.aggregate) - float(recorded))
        if float(slot.aggregate) == float(recorded):
            self.aggregate_exact += 1
        else:
            self.aggregate_disagreed += 1
        self.aggregate_max_gap = max(self.aggregate_max_gap, gap)

    def skip(self, why: str) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + 1

    def count(self, bucket: dict[str, int], key: Any) -> None:
        bucket[str(key)] = bucket.get(str(key), 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "batches": self.batches, "events": self.events,
            "results": self.results, "receipts": self.receipts,
            "already_present": self.already_present,
            "paired_deltas": self.paired_deltas,
            "paired_delta_claim": PAIRED_DELTA_CLAIM,
            "paired_delta_claim_status": (
                "CONFIRMED" if self.paired_deltas == PAIRED_DELTA_CLAIM
                else f"REFUTED: measured {self.paired_deltas}"),
            "validity": self.validity, "by_source": self.by_source,
            "by_cycle": self.by_cycle, "slot_widths": self.slot_widths,
            "aggregate_exact": self.aggregate_exact,
            "aggregate_disagreed": self.aggregate_disagreed,
            "aggregate_max_gap": self.aggregate_max_gap,
            "aggregate_status": self.aggregate_status,
            "skipped": self.skipped, "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# the migration
# ---------------------------------------------------------------------------


def migrate_v1(overfitx_home: Path | str, ledger: Ledger, *, source: str,
               kgraph_db: Path | str | None = None,
               foreign_sources: tuple[str, ...] = (),
               slot_key: str = AGGREGATE_SLOT,
               artifact_root: Path | str | None = None,
               slot_keys: Sequence[str] = (),
               slot_weights: Mapping[str, float] | None = None) -> Report:
    """Import the v1 archive at ``overfitx_home`` into ``ledger``.  Idempotent.

    ``source`` is the provenance label every imported row carries and every
    trial id is namespaced by.  It has no default; see the module docstring.

    Order is deliberate: journals first, then results, then receipts.  The
    journal is the only source that knows a trial's LINEAGE and its void
    status, and the commit ledger is the only one that knows the committed
    delta, so the second pass overlays the first.  Reversing them would file
    every committed row under a lineage of its own name.

    ``kgraph_db`` is read for ``foreign_sources`` only.  Rows carrying
    ``source`` are never taken from it even when present: the batch directories
    are the authoritative copy of those same runs, and importing both would put
    one run in the store twice under two ids.

    ``artifact_root`` + ``slot_keys`` OPT IN to the per-slot panel vector (see
    :func:`slot_vectors`).  Both default to absent and the migration is then
    byte-identical to what it did before: every row gets the 1-slot aggregate
    score, and ``Report.slot_widths`` says so.  Supplying them widens exactly
    those rows whose run AND whose batch baseline both carry a complete vector;
    every other row is left 1-slot wide rather than padded.

    ``slot_weights`` is the ACTIVE TASK's ``slots.weights``, keyed exactly as the
    task keys it, and it is the caller's to supply for the same reason
    ``slot_keys`` is: the kernel may not name a task, so the weight inventory
    arrives as a parameter rather than as a constant in this module.  It is
    joined per slot id by :func:`kernel.stats.paired_delta` -- weights naming
    something other than a slot id bind nothing and every slot keeps 1.0, which
    is what makes passing a group-keyed task's weights bit-identical to passing
    none.  Omitting it is NOT equivalent to declaring none for a slot-keyed task:
    it silently recomputes an unweighted mean of a weighted composite, and a
    task whose composite is a weighted sum is the case this parameter exists for.
    """
    home = Path(overfitx_home)
    report = Report()
    if not home.is_dir():
        report.notes.append(f"no archive at {home}")
        return report
    # An artifact root given against a task that declares no slot keys is a
    # SILENT no-op otherwise, and the caller would get 1-slot rows having asked
    # for 8. ``slot_vectors`` notes it; calling it with the empty tuple is how
    # that note gets written.
    vectors = (slot_vectors(artifact_root, slot_keys=slot_keys, report=report)
               if artifact_root is not None else {})
    for name, batch_dir in _batch_dirs(home):
        report.batches.append(name)
        cycle = _cycle_of(name)
        _import_journals(batch_dir, cycle, ledger, report)
        _import_results(batch_dir, name, cycle, ledger, report, source,
                        slot_key, vectors, slot_weights)
        _import_receipts(batch_dir, name, cycle, ledger, report, source)
    if kgraph_db is not None:
        import_foreign(kgraph_db, ledger, sources=foreign_sources,
                       report=report)
    return report


def import_foreign(db_path: Path | str, ledger: Ledger, *,
                   sources: tuple[str, ...],
                   report: Report | None = None) -> Report:
    """Import already-normalised nodes from a second-store SQLite, read-only.

    ONE function rather than the plan's three named wrappers, for the reason in
    the module docstring: the wrappers named tasks and the kernel may not.  A
    caller wanting one source passes a one-tuple, which is what those wrappers
    were.

    THE HALF THAT IS NOT PORTED, NAMED.  Each v1 ingester had two halves: a
    scanner that walked a foreign tree, and a normaliser that resolved seeds and
    classified validity.  Only the normalised OUTPUT is read here, because the
    migration is one-shot and that output already exists — the scanners have
    nothing left to scan that has not already been scanned.  What is lost is the
    ability to re-scan a foreign tree that was never ingested; if that is ever
    needed it is a new ingester against a live source, not a resurrection of
    these.

    The SDK-side second store is 0 bytes and always has been, so pointing this
    at the repo's own copy imports nothing.  That is reported as zero rows
    rather than as an error: phase-4 evidence section 11 measured that this
    store contributed nothing to any of the five audit passes, and a migration
    that pretended otherwise would be the first thing to claim it did.
    """
    report = report if report is not None else Report()
    path = Path(db_path)
    if not sources:
        report.notes.append(f"{path}: no foreign sources named, 0 rows")
        return report
    if not path.is_file() or path.stat().st_size == 0:
        report.notes.append(f"{path}: absent or empty, 0 rows")
        return report
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        marks = ",".join("?" * len(sources))
        rows = conn.execute(
            f"SELECT * FROM nodes WHERE source IN ({marks}) ORDER BY id",
            tuple(sources)).fetchall()
    except sqlite3.Error as exc:
        report.notes.append(f"{path}: unreadable ({exc})")
        return report
    finally:
        conn.close()
    for row in rows:
        _record(ledger, _from_node(dict(row)), report)
    return report


def slot_vectors(artifact_root: Path | str, *, slot_keys: Sequence[str],
                 report: Report | None = None
                 ) -> dict[str, dict[str, float]]:
    """``{run_name: {slot: absolute score}}`` read from the artifact tree.

    THIS IS THE PATH THAT MAKES PER-SLOT RETENTION POSSIBLE AT ALL.  The module
    docstring above records the measurement: the batch directories this
    migration reads do not contain the panel's slot decomposition, so every row
    imported from them carries a ONE-slot score, and ``policy.select.retain``
    refuses a one-slot matrix because on one slot its rule is arithmetically
    identical to greedy idxmax.  The eight values live in each run's
    ``summary.json`` under the artifact root and nowhere else.  Reading them
    costs no GPU -- the files are already on disk -- which is the whole reason
    this is the unblocking step rather than an experiment.

    ``slot_keys`` is REQUIRED and undefaulted, for the same reason ``source``
    is: which keys are the panel's slots is task inventory, and a default here
    would name a task inside ``kernel/``.  A run is indexed only when EVERY
    declared key is present and finite; a partial vector is dropped rather than
    padded, because a padded slot is a fabricated measurement and the padding
    would be invisible downstream.

    THE TWO LAYOUTS ARE BOTH READ.  ``<root>/<run>/summary.json`` is one
    artifact directory; ``<root>/<batch>/<run>/summary.json`` is the whole
    artifact store.  Neither is guessed at from the other -- both globs run.

    NAME COLLISIONS ARE NOT RESOLVED BY PICKING ONE.  The same run name exists
    under two artifact directories in the live store (32 of 2,165 measured
    2026-08-30).  Where every copy carries an identical vector the run is
    indexed, because there is no ambiguity to resolve; where the copies
    DISAGREE the run is dropped and counted under
    ``slot_vector_collision_disagrees``, since choosing between two different
    measurements of the same name is exactly the fabrication this store exists
    to refuse.
    """
    keys = tuple(str(k) for k in slot_keys)
    report = report if report is not None else Report()
    if not keys:
        report.notes.append("slot_vectors: no slot keys declared, 0 vectors")
        return {}
    root = Path(artifact_root)
    if not root.is_dir():
        report.notes.append(f"slot_vectors: no artifact root at {root}")
        return {}
    found: dict[str, list[dict[str, float]]] = {}
    seen: set[Path] = set()
    for pattern in (f"*/{SUMMARY_FILE}", f"*/*/{SUMMARY_FILE}"):
        for path in sorted(root.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            vector = _slot_vector(path, keys)
            if vector is None:
                report.skip("slot_vector_incomplete")
                continue
            found.setdefault(path.parent.name, []).append(vector)
    out: dict[str, dict[str, float]] = {}
    for run_name, copies in found.items():
        if all(copy == copies[0] for copy in copies[1:]):
            out[run_name] = copies[0]
        else:
            report.skip("slot_vector_collision_disagrees")
    return out


def _slot_vector(path: Path, keys: Sequence[str]) -> dict[str, float] | None:
    """One ``summary.json`` as a complete slot vector, or None.

    ``None`` covers unreadable, unparseable, and incomplete alike: all three
    mean "this run has no panel vector", and distinguishing them would only
    matter if the answer were ever anything but "leave the row 1-slot wide".
    """
    data = _json_file(path)
    if not isinstance(data, dict):
        return None
    vector: dict[str, float] = {}
    for key in keys:
        value = data.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return None
        vector[str(key)] = number
    return vector


def _baseline_run(batch_dir: Path) -> str | None:
    """The run name a batch's deltas were measured against.

    Read from ``panel.json``'s ``consolidated_baseline`` member, which is the
    same source :func:`_baseline_seed` trusts FIRST and for the same measured
    reason: the panel is what the cycle's deltas were actually computed
    against, and batch08's baseline commit row stamps a seed that disagrees
    with it.  ``None`` when the batch declares no such member, in which case
    its rows keep their 1-slot score.
    """
    panel = _json_file(batch_dir / "panel.json")
    members = panel.get("panel") if isinstance(panel, dict) else None
    for member in members or []:
        if isinstance(member, dict) \
                and member.get("role") == "consolidated_baseline":
            name = str(member.get("exp_id") or "").strip()
            if name:
                return name
    return None


# ---------------------------------------------------------------------------
# per-artifact importers
# ---------------------------------------------------------------------------


def _import_journals(batch_dir: Path, cycle: int, ledger: Ledger,
                     report: Report) -> None:
    """Every dispatch-journal line as one event, resuming from the cursor.

    The cursor is a LINE COUNT and not a hash of the file, because a live
    journal is appended to: keyed by content, a grown journal would re-import
    every line it already had; keyed by consumed lines, it imports exactly the
    tail.  A journal that SHRANK (a rotation) resumes from its own new length,
    which under-imports rather than duplicating — the safe direction, since a
    duplicated event silently doubles every count read off it.
    """
    for journal in sorted(batch_dir.glob("sdk_journal_cycle*.jsonl")):
        key = f"{batch_dir.name}/{journal.name}"
        lines = _lines(journal)
        seen = (_cursor_rows(ledger, key))
        for line in lines[min(seen, len(lines)):]:
            record = _json(line)
            if not isinstance(record, dict) or not record.get("event"):
                report.skip("journal_line_unparseable")
                continue
            fields = {k: v for k, v in record.items()
                      if k not in ("ts", "event", "cycle")}
            # The journal's own ``cycle`` field is kept under a second name
            # rather than dropped: where it disagrees with the file it was
            # written in, THAT is the ledger-provenance trap, and a migration
            # that silently prefers one of the two erases the evidence of it.
            if "cycle" in record:
                fields["journal_cycle"] = record["cycle"]
            try:
                ledger.record_event(str(record["event"]), cycle=cycle,
                                    ts=record.get("ts"), **fields)
            except UnknownEvent:
                report.skip("event_outside_vocabulary")
                continue
            except LedgerRefusal:
                report.skip("event_refused")
                continue
            report.events += 1
        ledger.record_cursor(key, _etag(journal), len(lines))


def _import_results(batch_dir: Path, batch: str, cycle: int, ledger: Ledger,
                    report: Report, source: str, slot_key: str,
                    vectors: Mapping[str, Mapping[str, float]],
                    slot_weights: Mapping[str, float] | None = None) -> None:
    """One :class:`Measurement` per trial, from the journal AND commit ledger.

    Neither source alone is enough, and the reason is measured: batch07's commit
    ledger carries ONE numeric ``paired_delta`` against 170 in its journal (the
    cycle recorded its deltas journal-side), while batch04 and batch05 have no
    journal at all and carry everything in the commit ledger.  A migration that
    read either one alone would silently lose a whole cycle.
    """
    commits = _commits(batch_dir)
    baseline_seed = _baseline_seed(batch_dir, commits)
    # The batch's own baseline vector is the DENOMINATOR every per-slot delta
    # in this batch is taken against.  Resolved once per batch, and absent
    # whenever the batch declares no consolidated baseline or that baseline
    # carries no complete vector -- in which case every row here stays 1-slot
    # wide, which is the honest answer and not a degradation to hide.
    base_name = _baseline_run(batch_dir)
    base_vector = vectors.get(base_name) if base_name else None
    rows = _journal_trials(batch_dir)
    for run_name, commit in commits.items():
        rows.setdefault(run_name, {})["commit"] = commit
    for run_name, parts in sorted(rows.items()):
        measurement = _measure(run_name, parts, batch_dir, batch, cycle,
                               baseline_seed, source, slot_key,
                               vectors.get(run_name), base_vector,
                               slot_weights)
        if measurement is None:
            report.skip("no_outcome")
            continue
        report.weigh_aggregate(measurement)
        _record(ledger, measurement, report)


def _import_receipts(batch_dir: Path, batch: str, cycle: int, ledger: Ledger,
                     report: Report, source: str) -> None:
    """Launch receipts, re-keyed from LINEAGE to TRIAL.

    The v1 address was ``receipts/<track>/<lineage_id>/launch_receipt.json`` —
    one file per lineage, so trial N+1 overwrote trial N and batch11 kept 26
    files against 68 ``trial_dispatched`` events.  Migration cannot recover what
    the overwrite destroyed; what it can do is stop the loss continuing, by
    keying each surviving receipt on the ``exp_id`` the file itself records.
    The count imported is therefore the count that SURVIVED, and
    ``Report.receipts`` says so rather than implying 68.

    BOTH ADDRESSES ARE READ, AND THAT IS PERMANENT, NOT A TRANSITION.  Batches
    07-12 are written in the legacy layout and are never rewritten — migration
    that could only read the current address would import nothing at all from
    the entire archive it exists to import.  The two globs differ by exactly one
    path component, so which one matched is also what says where the lineage id
    and the track are: one level up in the legacy layout, two in the current one.
    """
    seen: set[Path] = set()
    for depth, pattern in ((2, "receipts/*/*/trial*/launch_receipt.json"),
                           (1, "receipts/*/*/launch_receipt.json")):
        for path in sorted(batch_dir.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            _import_one_receipt(path, depth, batch, cycle, ledger, report,
                                source)


def _import_one_receipt(path: Path, depth: int, batch: str, cycle: int,
                        ledger: Ledger, report: Report, source: str) -> None:
    """One receipt file, from whichever of the two addresses held it.

    ``depth`` is how many directories separate the file from its lineage dir —
    1 for the legacy address, 2 for the trial-keyed one — which is the only
    thing that differs between them.
    """
    data = _json(path.read_text() if path.is_file() else "")
    if not isinstance(data, dict):
        report.skip("receipt_unparseable")
        return
    lineage_dir = path.parents[depth]
    lineage = str(data.get("lineage_id") or lineage_dir.name)
    run_name = str(data.get("exp_id") or lineage)
    trial_id = _trial_id(source, batch, run_name)
    fresh = ledger.receipt(trial_id) is None
    ledger.record_receipt(
        trial_id, lineage_id=lineage,
        track=str(data.get("track") or lineage_dir.parent.name),
        cycle=cycle, n=_int(data.get("trial")),
        exp_id=str(data.get("exp_id") or "") or None,
        job_id=str(data.get("job_id") or "") or None,
        worktree=str(data.get("worktree") or "") or None,
        output_dir=str(data.get("output_dir") or "") or None,
        config=str(data.get("config") or "") or None)
    if fresh:
        report.receipts += 1
    else:
        report.already_present += 1


# ---------------------------------------------------------------------------
# typing one trial
# ---------------------------------------------------------------------------


def _measure(run_name: str, parts: Mapping[str, Any], batch_dir: Path,
             batch: str, cycle: int, baseline_seed: int | None, source: str,
             slot_key: str,
             run_vector: Mapping[str, float] | None = None,
             base_vector: Mapping[str, float] | None = None,
             slot_weights: Mapping[str, float] | None = None
             ) -> Measurement | None:
    """Build the typed record for one run, or None when it is not a trial.

    OUTCOME IS DECIDED BY WHETHER A NUMBER EXISTS, not by the status string.
    The archive spells "no number" nineteen ways and each new spelling would
    otherwise silently become a scored row of value None.  ``INERT`` is split
    off from ``SCORED`` where the archive flagged numeric inertness: an inert
    run is a real measurement OF THE BASELINE, so it must not be dropped and
    must not count as mechanism evidence either.
    """
    trial = parts.get("trial") or {}
    commit = parts.get("commit") or {}
    delta = _float(commit.get("paired_delta"))
    if delta is None:
        delta = _float(trial.get("paired_delta"))
    score = _float(commit.get("score")) or _float(trial.get("score"))
    status = str(trial.get("status") or commit.get("run_type") or "")
    seed = _trial_seed(run_name, commit, batch_dir)
    card = _run_card(batch_dir, run_name)
    if seed is None:
        seed = _int((card or {}).get("seed"))
    provenance: dict[str, Any] = {
        "source": source, "batch": batch, "run_name": run_name,
        "run_type": str(commit.get("run_type") or ""),
        "track": str(trial.get("track") or commit.get("track") or ""),
        "mechanism_signature": _mechanism_stem(
            str(trial.get("lineage_id") or run_name)),
        "baseline_seed": baseline_seed,
        "score": score,
        "committed_at": str(commit.get("committed_at") or ""),
        "note": str(trial.get("note") or trial.get("notes") or "")[:400],
        "failure_class": str(trial.get("failure_class") or ""),
    }
    if card and isinstance(card.get("paired_delta_per_task"), dict):
        # Preserved verbatim, NOT folded into the slot vector: its mean does
        # not reproduce ``delta`` (exact on 2 of 770 cards).  See the module
        # docstring.  Commit 6 needs it; this store must not pretend it is the
        # eight-slot panel decomposition.
        provenance["per_task_delta"] = card["paired_delta_per_task"]
    if delta is None:
        outcome, slot = Outcome.VOID, None
    else:
        inert = bool((commit.get("numeric_inertness") or {}).get("flagged")) \
            if isinstance(commit.get("numeric_inertness"), dict) else False
        inert = inert or "inert" in status.lower()
        outcome = Outcome.INERT if inert else Outcome.SCORED
        slot = _slot_score(delta, slot_key, run_vector, base_vector, provenance,
                           slot_weights)
    if outcome is Outcome.VOID and not status and not trial and not commit:
        return None
    return Measurement(
        trial_id=_trial_id(source, batch, run_name),
        lineage_id=str(trial.get("lineage_id") or commit.get("lineage")
                       or run_name),
        cycle=cycle, outcome=outcome,
        base_ref=str(trial.get("base_ref") or ""),
        seed=seed, head_commit="",
        job_id=str(trial.get("job_id") or ""),
        output_dir=str(commit.get("output_dir") or ""),
        validity=_validity(delta, seed, baseline_seed),
        status=status, score=slot,
        failed_axes=(status,) if outcome is Outcome.VOID and status else (),
        provenance=provenance)


def _slot_score(delta: float, slot_key: str,
                run_vector: Mapping[str, float] | None,
                base_vector: Mapping[str, float] | None,
                provenance: dict[str, Any],
                slot_weights: Mapping[str, float] | None = None) -> SlotScore:
    """The panel vector when both sides carry one, the 1-slot aggregate else.

    THE AGGREGATE IS RECOMPUTED FROM THE SLOTS, not copied from the archive.
    ``SlotScore`` promises that its slots average to its aggregate, and a row
    whose aggregate came from one source and whose slots came from another
    would break that promise silently.  The archive's own number is preserved as
    ``provenance['recorded_delta']`` and the disagreement between the two is
    COUNTED (``Report.aggregate_*``), because a recomputation is a second
    implementation of the eval's own formula and the only thing that makes that
    safe is checking it against the number the eval actually wrote.

    THAT CHECK IMMEDIATELY REFUTED THE CLAIM THIS DOCSTRING USED TO MAKE.  It
    said the recomputed and recorded numbers "agree to within 2.7e-4 on the worst
    cycle and 5e-5 on c9-c11".  Counting them instead of asserting them, over the
    live archive's 612 multi-slot rows: they agree on ZERO of 612, worst gap
    2.306e-02 -- two orders above the quoted figure and above any graduation bar
    the project has used.  This is PRE-EXISTING and not caused by the weighting
    fix: migrating the same archive with weights supplied and with them omitted
    produces byte-identical aggregates on all 2,866 rows, and the same 612/612
    tally.  The old text was true of the per-run identity ``mean(slots) ==
    <recorded score>`` (bit-exact on 1,105 of 1,109 complete vectors, still
    measured) and was then carried over to the DELTAS, which are a different
    comparison: the recomputed delta is against ``panel.json``'s single
    ``consolidated_baseline`` run (see :func:`_baseline_run`), while the recorded
    ``paired_delta`` was written by the eval against its own reference.  Which
    reference that was is NOT established here, so the gap is reported as a
    measured quantity and not explained away.

    THE WEIGHTS COME FROM THE TASK, and this function no longer owns any
    weighting arithmetic: it calls :func:`kernel.stats.paired_delta`, the
    resolver that already existed for exactly this and that ``policy/select``
    reaches through ``weights.get(slot, 1.0)``.  The previous body hardcoded
    ``{slot: 1.0}`` and said so on purpose, because the only task that existed
    keyed its ``slots.weights`` over GROUPS (classification 3, segmentation 2,
    survival 2, robustness 1) while the vector is 8 SLOTS, so a group-keyed
    weight could not name a slot and 1.0-apiece was that task's right answer.
    That reasoning was about ONE task's key space and got written down as an
    arithmetic constant, which is the defect.  A task that keys its weights BY
    SLOT ID -- and whose eval composite is therefore a genuinely WEIGHTED sum
    rather than a mean -- gets that composite silently replaced by an unweighted
    mean of the same axes, a gap on the order of the graduation bar rather than
    of rounding.  The task spec is the only thing that knows which shape it is,
    which is why the weights arrive from it rather than from here.

    Passing the declared weights is BIT-SAFE for a group-keyed task rather than
    merely compatible with one: ``paired_delta`` joins on the delta row's own
    slot ids, so v1's four group names match nothing, every slot falls through
    to 1.0, and the aggregate is the same unweighted mean it always was.  A task
    declaring no weights at all lands in the same place.  This is checked, not
    assumed -- see ``tests/test_kernel_ingest.py``.
    """
    if run_vector and base_vector and set(run_vector) == set(base_vector):
        provenance["recorded_delta"] = delta
        return paired_delta(run_vector, base_vector, slot_weights or {})
    return SlotScore(deltas={slot_key: delta}, weights={slot_key: 1.0},
                     aggregate=delta, total_weight=1.0)


def _validity(delta: float | None, seed: int | None,
              baseline_seed: int | None) -> str:
    """THE rule the paired-delta count is measured under.  Written down here so
    the number in :data:`PAIRED_DELTA_MEASURED` is reproducible rather than
    whatever the code did on the day.

    * no delta                      -> ``unpaired``
    * both seeds known and equal    -> ``paired_same_seed``  (the only real pair)
    * both seeds known and different-> ``cross_seed``
    * either seed unknown           -> ``unknown``

    ``unknown`` is a third state and not a synonym for cross: 448 of 817 imported
    deltas land there because the archive never stamped the trial's seed, and
    calling them cross would assert something nobody measured, while calling them
    paired would manufacture the evidence this store exists to refuse.
    """
    if delta is None:
        return V_UNPAIRED
    if seed is None or baseline_seed is None:
        return V_UNKNOWN
    return V_PAIRED if int(seed) == int(baseline_seed) else V_CROSS


def _from_node(node: Mapping[str, Any]) -> Measurement:
    """One already-normalised kgraph node as a :class:`Measurement`."""
    delta = _float(node.get("paired_delta"))
    status = str(node.get("status") or "")
    outcome = (Outcome.VOID if delta is None
               else Outcome.INERT if "replicate" in status else Outcome.SCORED)
    slot = None if delta is None else SlotScore(
        deltas={AGGREGATE_SLOT: delta}, weights={AGGREGATE_SLOT: 1.0},
        aggregate=delta, total_weight=1.0)
    return Measurement(
        trial_id=str(node.get("id")),
        lineage_id=str(node.get("name") or node.get("id")),
        cycle=_int(node.get("cycle")) or 0, outcome=outcome,
        seed=_int(node.get("train_seed")),
        validity=str(node.get("delta_validity") or V_UNKNOWN),
        status=status, score=slot,
        provenance={"source": str(node.get("source") or ""),
                    "batch": str(node.get("batch") or ""),
                    "run_name": str(node.get("name") or ""),
                    "track": str(node.get("track") or ""),
                    "mechanism_signature": str(
                        node.get("mechanism_signature") or ""),
                    "move_type": str(node.get("move_type") or ""),
                    "score": _float(node.get("score")),
                    "note": str(node.get("notes") or "")[:400]})


def _record(ledger: Ledger, measurement: Measurement, report: Report) -> None:
    """Write one measurement, counting the refusal instead of raising on it.

    A refusal here is a FACT about the archive — a non-finite delta, an unpaired
    score, or the same run recorded twice with two different answers — and the
    migration's job is to report how many there are, not to stop at the first.
    """
    fresh = ledger.result(measurement.trial_id) is None
    try:
        ledger.record_result(measurement)
    except LedgerRefusal as exc:
        report.skip(f"refused:{str(exc).split(':')[0][:40]}")
        return
    if fresh:
        report.results += 1
    else:
        report.already_present += 1
    report.count(report.validity, measurement.validity)
    report.count(report.by_source,
                 measurement.provenance.get("source") or "?")
    report.count(report.by_cycle, measurement.cycle)
    report.count(report.slot_widths,
                 0 if measurement.score is None
                 else measurement.score.n_slots)


# ---------------------------------------------------------------------------
# archive readers — every one of these is read-only
# ---------------------------------------------------------------------------


def _batch_dirs(home: Path) -> Iterator[tuple[str, Path]]:
    for path in sorted(home.glob("batch*")):
        if path.is_dir() and _BATCH_RE.match(path.name):
            yield path.name, path


def _cycle_of(batch: str) -> int:
    match = _BATCH_RE.match(batch)
    return int(match.group(1)) if match else 0


def _trial_id(source: str, batch: str, run_name: str) -> str:
    """Delegates to :func:`kernel.bridge.trial_id`.

    The spelling lives in ONE place because the live writers now mint trial ids
    too: a migrated row and a live row for the same run must land on the same
    key, and a second copy of this f-string is how that stops being true.
    """
    return _bridge_trial_id(source, batch, run_name)


def _commits(batch_dir: Path) -> dict[str, dict[str, Any]]:
    data = _json_file(batch_dir / "commit_ledger.json")
    commits = data.get("commits") if isinstance(data, dict) else None
    if not isinstance(commits, dict):
        return {}
    return {str(k): v for k, v in commits.items() if isinstance(v, dict)}


def _journal_trials(batch_dir: Path) -> dict[str, dict[str, Any]]:
    """``{run_name: {"trial": the trial_completed record that counts}}``.

    THE RECORD THAT CARRIES A NUMBER WINS, and only among those does the last
    win.  115 run names carry more than one ``trial_completed``, and for 33 of
    them the LAST one carries no number while an earlier one does — every one a
    ``done`` row with a real delta followed by a ``void_foreign_resume`` emitted
    when the driver resumed.  Plain last-wins would re-type 33 real measurements
    as voids on a bookkeeping event, which is the same mistake, in miniature,
    that once wrote off 908 trials: a resume is not a result.
    """
    out: dict[str, dict[str, Any]] = {}
    for journal in sorted(batch_dir.glob("sdk_journal_cycle*.jsonl")):
        for line in _lines(journal):
            record = _json(line)
            if not isinstance(record, dict):
                continue
            if record.get("event") != "trial_completed":
                continue
            run_name = record.get("run_name")
            if not run_name:
                continue
            slot = out.setdefault(str(run_name), {})
            previous = slot.get("trial")
            if previous is not None and _scored_record(previous) \
                    and not _scored_record(record):
                continue
            slot["trial"] = record
    return out


def _scored_record(record: Mapping[str, Any]) -> bool:
    return (_float(record.get("paired_delta")) is not None
            or _float(record.get("score")) is not None)


def _run_card(batch_dir: Path, run_name: str) -> dict[str, Any] | None:
    data = _json_file(batch_dir / "run_cards" / f"{run_name}_runcard.json")
    return data if isinstance(data, dict) else None


def _baseline_seed(batch_dir: Path,
                   commits: Mapping[str, Any]) -> int | None:
    """The seed a same-seed pair in this batch must match.  Trust order:

    1. ``panel.json``'s ``consolidated_baseline`` member — the panel is what the
       cycle's deltas were actually computed against.
    2. an in-batch ``baseline_c<N>_s<seed>.json`` consolidation record.
    3. the batch's own baseline commit row's ``provenance.train_seed``.

    None when the batch carries none of the three, so its rows classify as
    ``unknown`` rather than being guessed.  Order 3 is LAST for a measured
    reason: batch08's baseline row is named ``c8-base-882147`` but stamps
    ``train_seed: 331892``, and trusting the stamp over the panel would re-type
    115 same-seed rows as cross-seed.
    """
    panel = _json_file(batch_dir / "panel.json")
    members = panel.get("panel") if isinstance(panel, dict) else None
    for member in members or []:
        if isinstance(member, dict) \
                and member.get("role") == "consolidated_baseline":
            seed = _int(member.get("seed"))
            if seed is not None:
                return seed
    for path in sorted(batch_dir.glob("baseline_c*_s*.json")):
        record = _json_file(path)
        seed = _int(record.get("seed")) if isinstance(record, dict) else None
        if seed is not None:
            return seed
        match = re.search(r"_s(\d{3,})", path.name)
        if match:
            return int(match.group(1))
    for row in commits.values():
        if isinstance(row, dict) and "base" in str(row.get("run_type") or ""):
            seed = _int((row.get("provenance") or {}).get("train_seed"))
            if seed is not None:
                return seed
    return None


def _trial_seed(run_name: str, commit: Mapping[str, Any],
                batch_dir: Path) -> int | None:
    """The seed a trial trained at: the explicit stamp, then the run card, then
    the run NAME.  The name is last and is a genuine inference — ``c4-asym-571261``
    names its seed — so it is used only when nothing stamped one, and the row it
    produces is still subject to :func:`_validity`.
    """
    seed = _int((commit.get("provenance") or {}).get("train_seed"))
    if seed is not None:
        return seed
    card = _run_card(batch_dir, run_name)
    seed = _int((card or {}).get("seed"))
    if seed is not None:
        return seed
    found = _SEED_RE.findall(run_name)
    return int(found[-1]) if found else None


def _mechanism_stem(lineage_id: str) -> str:
    """A cross-cycle mechanism key: the lineage id minus its cycle and track.

    ``c11-train-p-c6-koleo-antcollapse-w01`` -> ``c6-koleo-antcollapse-w01``,
    which is the same key the identical mechanism gets in c10.  This is the
    NORMALISED STEM, not ``mechanism_id``'s structural signature over
    (family, move_type, config_keys, code_target): it catches the replay band
    that is actually detectable — 7.4% of proposals are byte-identical and 8.9%
    reach Jaccard 0.5, with almost nothing in between — and it catches nothing
    else.  Naming the weaker key honestly is the point; an embedding checker
    over this corpus was measured to add 1.5-4.1 points and 85.5% of idea stems
    have no outcome to compare against at all.
    """
    return _STEM_RE.sub("", lineage_id.strip()) or lineage_id


# ---------------------------------------------------------------------------
# tiny readers
# ---------------------------------------------------------------------------


def _lines(path: Path) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()
    except OSError:
        return []


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _etag(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return "0:0"
    return f"{int(stat.st_mtime)}:{stat.st_size}"


def _cursor_rows(ledger: Ledger, key: str) -> int:
    cursor = ledger.cursor(key)
    return int(cursor["n_rows"]) if cursor else 0


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    # Refused at the door rather than at ``record_result``: a NaN that reached
    # the writer would be counted as a skip with no source named, and the point
    # of reading it here is to be able to say which archive row carried it.
    return None if number != number or number in (float("inf"),
                                                  float("-inf")) else number


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value == int(value):
        return int(value)
    return None
