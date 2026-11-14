"""kernel.ledger — THE durable store, in its dual-write phase.

WHAT THIS COMMIT DOES (plan section 8.2, commit 4)
--------------------------------------------------
Every ``record_*`` writes SQLite **and** the JSON artifact the v1 tree already
writes.  Readers still read the JSON.  Divergence between the two projections is
a test failure (``tests/test_kernel_ledger.py``), which is the whole safety
argument for flipping the readers at commit 6: the SQLite side is proven to hold
the same facts before anything starts trusting it.

The JSON side is not a courtesy copy.  It is the store the running system reads
today, so this module owns BOTH writes and performs them inside one transaction.
That is what makes "one writer API" true while three processes (the driver, the
``overfitx-advance`` unit, post-mortem intake) still write concurrently.

THE THREE DEFECTS THIS MODULE EXISTS TO FIX
-------------------------------------------
1. **A torn write destroyed two events.**  ``journal.Journal.append`` opened the
   file in ``"a"`` mode and wrote one newline-TERMINATED line without ever
   checking that the file already ended in one.  After a crash mid-append the
   next record was concatenated onto the torn line, so one bad shutdown cost the
   torn event *and the first good event after it*.  :meth:`Ledger.record_event`
   checks the last byte and emits a leading newline when it is not ``\\n``.  The
   behavioural contract pinned this as an xfail
   (``test_an_event_appended_after_a_torn_line_is_still_readable``); with this
   module on the write path it passes, and the marker is gone.

2. **One lock, two writers.**  ``overfitx_tools/tools/commit_full.py:281``
   degrades to "proceeding UNLOCKED" whenever the flock cannot be acquired,
   while ``overfitx_tools/tools-v2/state.py cmd_intake`` read-modify-writes the
   same ``commit_ledger.json`` and never imports the lock module at all.  Two
   writers, one of which never takes the lock, lose each other's rows.  Here the
   read-modify-write happens INSIDE ``BEGIN IMMEDIATE``: SQLite's write lock is
   the cross-process mutex, it is not optional, and it cannot degrade to
   unlocked because there is no code path that proceeds without it.  Verified
   cross-process on this filesystem — a second process attempting a concurrent
   ``BEGIN IMMEDIATE`` is refused rather than admitted.

3. **Receipts keyed by lineage, not trial.**  ``tools-v2/state.py``'s
   ``_receipt_path`` yielded ONE receipt file per lineage, so trial N+1
   overwrote trial N — measured 26 receipt files against 68 ``trial_dispatched``
   events in live ``batch11``.  :meth:`Ledger.record_receipt` uses ``trial_id``
   as the SQLite primary key, and ``Identity.receipt`` now carries the trial
   ordinal, so N trials of one lineage are N rows AND N files.  Commit 4 fixed
   the row and left the file; this commit moves the file.  The count change is
   asserted against the real archive in
   ``tests/test_receipts_are_trial_keyed.py``, not against a fixture.

WAL, MEASURED NOT ASSUMED
-------------------------
The store lives under ``OVERFITX_HOME`` on ``/admin``, which is **virtiofs**.
WAL is broken over NFS, so it was verified rather than assumed:
``PRAGMA journal_mode=WAL`` returns ``wal``, the ``-wal`` and ``-shm`` files are
created, a second connection in a second PROCESS reads committed rows, a reader
is not blocked by an open writer transaction, and a second writer is correctly
refused.  Checked on the live home, on this repo, and on ext4 ``/tmp``; all
three behave identically.  ``synchronous=FULL`` is kept (the default) rather
than dropped to NORMAL: this store is the record of what a 15-GPU-hour run
measured, and a lost commit on power failure is not a trade worth making for a
few milliseconds per event.

THE QUERY HALF (plan section 8.2, commit 5)
-------------------------------------------
``slot_matrix``, ``mechanism_outcomes``, ``void_clusters``,
``assumption_ledger`` and ``best_path`` were deferred by commit 4 with a stated
reason: they need a migrated archive to answer anything, so writing them then
would have meant writing them against an empty store, i.e. certifying the
fixture.  Commit 5 supplies the archive — ``kernel.ingest`` reads eleven live
batch directories — and the reason is honoured rather than merely retired:
``tests/test_kernel_ingest.py`` runs all five against the REAL migrated rows and
skips, loudly, when the archive is not on the machine.  No fixture stands in.

Two of them return less than their name promises, and both say so in their own
docstring rather than here: ``slot_matrix`` is one slot wide for migrated rows
because the archive's per-slot vector is not in the source set commit 5 was
pointed at, and ``assumption_ledger`` enumerates the components the archive
NAMES, which is not a decomposition of the base recipe.  Neither pads.

LAYERING
--------
Imports the stdlib and ``kernel.model``, nothing else, and names no task.  Every
path is handed in by the caller — this module resolves no filename, so the
artifact names stay declared in the task YAML (``identity.ledger_db``,
``identity.journal``, ``identity.commit_ledger``, ``identity.receipt``) rather
than hardcoded one layer below the spec that owns them.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from kernel.model import Measurement, Outcome

# ---------------------------------------------------------------------------
# The event vocabulary
# ---------------------------------------------------------------------------

#: Every event name production code appends to the dispatch journal, enumerated
#: from the tree with ``ast`` (not with a regex: nine of these are written as a
#: module constant, a conditional expression or a wrapper argument, and a
#: line-oriented grep sees none of them).
#:
#: WHY THIS IS A FROZEN SET AND NOT A CONFIGURABLE LIST
#: ---------------------------------------------------
#: A renamed event is not a test failure, it is a silently broken resume: the
#: replay stops recognising the record, the cycle re-does work it already did,
#: and nothing anywhere says so.  Freezing the names is therefore exactly the
#: thing to freeze, and it belongs with the store rather than in the task spec
#: because it is the store's SCHEMA — it is task-independent in the same way
#: ``Outcome`` is, and a task that could rename ``trial_completed`` could break
#: every reader in the system by editing YAML.
#:
#: Commits 11-12 rewrite the author and the driver.  Without this set, either
#: could quietly change what ``trial_completed`` or a ``trial_void_*`` means and
#: no contract would notice.
EVENT_VOCABULARY: frozenset[str] = frozenset({
    # cap extensions — an anchor gate conceding its own false negative
    "anchor_repair_cap_extended",
    "opus_anchor_gate_cap_extended",
    # per-base budget accounting
    "base_budget_excluded",
    "base_budget_exhausted",
    "base_budget_unknown_base_refused",
    "full_slots_cap",
    # ideation.  `idea_rejected` is the DISCARD PATH, and journaling it is the
    # point: 378 of 1,316 generated ideas never reached proposals.jsonl and
    # nothing recorded why, so "the ideator is derivative" and "the filter ate
    # it" were indistinguishable.  `lane_saturated` / `lane_reopened` are the
    # append-only kill-ledger: a lane dies WITH a named re-entry condition, and
    # it lives in the store rather than in a markdown file because a
    # prompt-sized file gets compacted by BYTES and the knowledge is what goes.
    "briefs_done",
    "idea_injected",
    "idea_rejected",
    "ideation_done",
    "ideation_refresh",
    "lane_reopened",
    "lane_saturated",
    "meta_work_routed",
    "operator_idea_answered",
    "operator_idea_unanswered",
    # exploitation of a measured parent
    "exploit_gate",
    "exploit_inheritance_refused_cross_base",
    "exploit_inheritance_rollup",
    "exploit_parent_not_seeded",
    "exploit_parent_unjoined",
    # lineage lifecycle
    "lineage_dispatched",
    "lineage_parked_skip",
    "lineage_verdict",
    "knob_claim_skipped",
    # the two kills, deliberately distinct: one is backed by a MEASUREMENT and
    # the other is not, and every void analysis reads the second as "never
    # tested".  Collapsing them would file killed-on-evidence lineages in the
    # bucket that means the opposite.
    "lineage_killed_first_trial_negative",
    "lineage_killed_void_cap",
    "verdict_guard",
    # trial lifecycle
    "trial_completed",
    "trial_deferred",
    "trial_dispatched",
    "trial_recorded",
    "trial_replicate_admitted",
    "spec_rejected",
    "spec_repaired",
    # pre-dispatch refusals, one name per axis: a single "gate_blocked" is what
    # left 178 rows in the archive with no failed_axes and made the phase-4
    # audit unanswerable.
    "trial_void_anchor",
    "trial_void_base_identity",
    "trial_void_duplicate_config",
    "trial_void_liveness",
    "trial_void_output_dir",
    "trial_void_protected_key",
    "trial_void_seed_identity",
    # the CONFIRM stage.  Added with policy/confirm.py, which is the first
    # consumer `decision.confirm` ever had: for three cycles the stage never
    # executed and there was no word for "it did", so `n_members: 0` and "the
    # second pass never ran" were indistinguishable in the journal.
    "confirm_dispatched",
    "confirm_completed",
    # cycle transitions
    "champion_promotion",
    "graduation_verdict",
    "phase",
    "reseed_completed",
    "reseed_dispatched",
    "reseed_gate_preflight_blocked",
    "reseed_gate_preflight_passed",
    "track_aborted",
    "track_stopped",
    "void_cluster_infra",
    # health / supervision
    "frozen_reap",
    "health_escalation",
    "model_escalation",
})


#: One row per imported artifact.  Created lazily by the two cursor methods
#: rather than added to :data:`_SCHEMA`, so a store written by commit 4 gains
#: the table on first import instead of needing a schema migration.
_CURSOR_SCHEMA = """CREATE TABLE IF NOT EXISTS ingest_cursor (
    source_key TEXT PRIMARY KEY,
    etag TEXT NOT NULL,
    n_rows INTEGER NOT NULL,
    ts REAL NOT NULL
)"""

#: ``config.NOISE_FLOOR``, i.e. the task spec's ``noise.floor_sigmas``.  Read
#: by :meth:`Ledger.assumption_ledger` to decide whether a challenge DISPLACED
#: a component or merely failed to.  It is a literal here and not a spec lookup
#: for the reason the whole module is task-independent: the store must be
#: readable with no task loaded.  A caller holding a ``TaskSpec`` should prefer
#: its ``noise`` block; this is the floor the archive was written under.
_NOISE_FLOOR = 0.004


class UnknownEvent(ValueError):
    """An event name outside the frozen vocabulary reached the store.

    Raised rather than logged.  An unrecognised name is a reader that will never
    fire again, and the failure mode of "log it and continue" is a resume that
    silently re-runs work — which is the expensive half of a 93.6%
    no-measurement rate, not the cheap half.
    """


class LedgerRefusal(ValueError):
    """The store refused a write because the row was not evidence.

    ``orchestrator.py:5679`` fabricates ``score=0.0`` on void rows today, and a
    0.0 that reads as "measured zero" rather than "never measured" pollutes
    every mean, every noise estimate and every parent selection downstream.
    This exception is the shape of making that unexpressible.
    """


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

#: One statement per line so a migration diff is readable.  ``events`` is
#: append-only and keyed by rowid, exactly like the JSONL it mirrors.
#: ``results`` and ``receipts`` are keyed by ``trial_id`` — defect 3 — so a
#: second trial of one lineage is a second row and not an overwrite.
_SCHEMA: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS events (
           id      INTEGER PRIMARY KEY AUTOINCREMENT,
           cycle   INTEGER NOT NULL,
           ts      REAL    NOT NULL,
           event   TEXT    NOT NULL,
           fields  TEXT    NOT NULL
       )""",
    "CREATE INDEX IF NOT EXISTS events_cycle ON events (cycle, id)",
    "CREATE INDEX IF NOT EXISTS events_name ON events (event)",
    """CREATE TABLE IF NOT EXISTS results (
           trial_id    TEXT    PRIMARY KEY,
           lineage_id  TEXT    NOT NULL,
           cycle       INTEGER NOT NULL,
           outcome     TEXT    NOT NULL,
           run_name    TEXT    NOT NULL,
           base_ref    TEXT    NOT NULL,
           seed        INTEGER,
           head_commit TEXT    NOT NULL,
           job_id      TEXT    NOT NULL,
           output_dir  TEXT    NOT NULL,
           validity    TEXT    NOT NULL,
           status      TEXT    NOT NULL,
           aggregate   REAL,
           total_weight REAL,
           ts          REAL    NOT NULL,
           measurement TEXT    NOT NULL
       )""",
    "CREATE INDEX IF NOT EXISTS results_lineage ON results (lineage_id)",
    "CREATE INDEX IF NOT EXISTS results_cycle ON results (cycle)",
    """CREATE TABLE IF NOT EXISTS receipts (
           trial_id   TEXT    PRIMARY KEY,
           lineage_id TEXT    NOT NULL,
           track      TEXT    NOT NULL,
           cycle      INTEGER NOT NULL,
           n          INTEGER,
           exp_id     TEXT,
           job_id     TEXT,
           worktree   TEXT,
           output_dir TEXT,
           config     TEXT,
           ts         REAL    NOT NULL
       )""",
    "CREATE INDEX IF NOT EXISTS receipts_lineage ON receipts (lineage_id, track)",
)

#: Applied to every connection.  ``busy_timeout`` is what turns "another process
#: holds the write lock" from an exception into a wait, and it is the reason
#: this design needs no ``filelock`` dependency.  It is FIRST on purpose: every
#: statement below it inherits the wait, and the one statement that does not
#: inherit it (``journal_mode``) is handled separately by ``_enable_wal``.
_PRAGMAS: tuple[str, ...] = (
    "PRAGMA busy_timeout=30000",
    "PRAGMA foreign_keys=ON",
)

#: How long ``_enable_wal`` will keep trying to convert the journal.  Short,
#: because the only thing it waits on is another process finishing the same
#: conversion, which takes milliseconds.
_WAL_CONVERT_TIMEOUT_S: float = 10.0
_WAL_CONVERT_POLL_S: float = 0.02


def _journal_mode(db: sqlite3.Connection) -> str:
    return str(db.execute("PRAGMA journal_mode").fetchone()[0]).lower()


def _enable_wal(db: sqlite3.Connection) -> str:
    """Convert the store to WAL, tolerating a process that got here first.

    ``busy_timeout`` does not cover this one statement.  Changing the journal
    mode needs a brief exclusive lock, and SQLite refuses the change outright
    with ``database is locked`` when another connection holds a transaction —
    the busy handler is not consulted, so a 30-second timeout buys nothing.

    That is not theoretical.  Two writers calling ``Ledger.open`` at the same
    instant raced here, and the loser died on the FIRST pragma, before any
    transaction existed — so the "one writer API" guarantee this module is
    built on never got the chance to hold.  It was not a lost row; it was a
    dead process.  ``test_two_writer_processes_lose_no_row`` caught it about
    one run in eight, which is exactly the rate that reads as an unrelated
    flake in a 5,500-test suite.

    Retrying is the whole fix, because the conversion is a property of the
    DATABASE, not of the connection: once the winner has converted, every
    later connection reads back ``wal`` and needs no conversion of its own.
    """
    deadline = time.monotonic() + _WAL_CONVERT_TIMEOUT_S
    while True:
        try:
            if _journal_mode(db) == "wal":
                return "wal"
            return str(db.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) and "busy" not in str(exc):
                raise
            if time.monotonic() >= deadline:
                # Report what the store actually is rather than raising: the
                # caller's contract is "say so if this is not WAL", and a
                # correct-but-serialised store beats a dead process.
                try:
                    return _journal_mode(db)
                except sqlite3.OperationalError:
                    return "unknown"
            time.sleep(_WAL_CONVERT_POLL_S)


# ---------------------------------------------------------------------------
# Replay — the projection a resume reads
# ---------------------------------------------------------------------------


class Replay:
    """What the store says already happened, for one cycle.

    Deliberately thin.  ``journal.Replay`` still owns the v1 resume projection
    (``trial_dispatched``, ``head_map``, ``resolved_seed_ref`` ...) and keeps
    owning it until commit 6 flips the readers; duplicating that logic here
    would create two answers to "was this trial dispatched?" while both stores
    are live, which is the exact defect this module deletes elsewhere.
    """

    __slots__ = ("cycle", "events")

    def __init__(self, cycle: int, events: Iterable[Mapping[str, Any]]):
        self.cycle = int(cycle)
        self.events: tuple[dict[str, Any], ...] = tuple(dict(e) for e in events)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.events)

    def saw(self, event: str) -> bool:
        """True if this cycle already recorded ``event`` — the resume gate."""
        return any(e.get("event") == event for e in self.events)

    def named(self, event: str) -> list[dict[str, Any]]:
        """Every record of one event name, oldest first."""
        return [e for e in self.events if e.get("event") == event]

    def count(self, event: str) -> int:
        return len(self.named(event))


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class Ledger:
    """One writer API over SQLite/WAL, dual-writing the v1 JSON artifacts.

    Every path is explicit and optional.  A ``Ledger`` handed no
    ``journal_path`` still records events to SQLite; it simply has no JSON half
    to keep in step.  That is what lets commit 6 delete the JSON writers by
    passing ``None`` rather than by editing this class.
    """

    def __init__(self, db_path: Path | str, *,
                 cycle: int = 0,
                 journal_path: Path | str | None = None,
                 commit_ledger_path: Path | str | None = None,
                 receipt_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path)
        self.cycle = int(cycle)
        self.journal_path = Path(journal_path) if journal_path else None
        self.commit_ledger_path = (
            Path(commit_ledger_path) if commit_ledger_path else None)
        #: A ``str.format`` template taking ``track``, ``lineage_id`` and ``n``.
        #: The ordinal is what makes the address trial-keyed; a template without
        #: it raises ``KeyError`` at the first write rather than quietly
        #: collapsing every trial of a lineage onto one file.
        self.receipt_path = str(receipt_path) if receipt_path else None
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None

    @classmethod
    def open(cls, db_path: Path | str, **kwargs: Any) -> "Ledger":
        """Open (creating if absent) the store at ``db_path``."""
        ledger = cls(db_path, **kwargs)
        ledger._connect()
        return ledger

    # -- connection ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._db is not None:
            return self._db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.db_path, timeout=30.0,
                             isolation_level=None, check_same_thread=False)
        db.row_factory = sqlite3.Row
        for pragma in _PRAGMAS:
            db.execute(pragma)
        mode = _enable_wal(db)
        if mode not in ("wal", "unknown"):
            # Say so rather than silently running in rollback-journal mode: WAL
            # is what makes the concurrent-reader promise true, and a store that
            # quietly lost it would fail as mysterious lock contention months
            # later.  Not fatal — correctness does not depend on WAL, only
            # concurrency does.
            db.execute("PRAGMA journal_mode=DELETE")
        for statement in _SCHEMA:
            db.execute(statement)
        self._db = db
        return db

    def connection(self) -> sqlite3.Connection:
        """The configured connection (WAL, busy_timeout, autocommit), opened on
        first use. For sibling stores that keep their own tables in this file."""
        return self._connect()

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> "Ledger":
        self._connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def journal_mode(self) -> str:
        """The journal mode actually in force — measured, never assumed."""
        return str(self._connect().execute("PRAGMA journal_mode").fetchone()[0])

    # -- writes -------------------------------------------------------------

    def record_event(self, event: str, *, cycle: int | None = None,
                     ts: float | None = None, **fields: Any) -> dict[str, Any]:
        """Append one event to BOTH stores; return the record as written.

        Refuses an event name outside :data:`EVENT_VOCABULARY`.

        The SQLite row and the JSONL line are written inside one
        ``BEGIN IMMEDIATE`` transaction, so a failure on the JSON side rolls the
        SQLite side back rather than leaving the two stores disagreeing.  The
        residual window is a failed COMMIT after a successful JSONL append,
        which loses the SQLite row only; commit 5's ingest reconciles from the
        JSONL, which is why the JSONL is written second and not first.
        """
        if event not in EVENT_VOCABULARY:
            raise UnknownEvent(
                f"{event!r} is not in the frozen journal vocabulary. Every "
                f"consumer — resume, the dashboard, the analyst, the "
                f"void-cluster trigger — reads these names; an unrecognised one "
                f"is a reader that will never fire again. Add it to "
                f"kernel.ledger.EVENT_VOCABULARY and to the behavioural "
                f"contract's frozen list in the same commit, or use one of the "
                f"{len(EVENT_VOCABULARY)} names that exist.")
        shadowed = {"ts", "event"} & set(fields)
        if shadowed:
            # Silent divergence, caught at the door: these two are COLUMNS on
            # the SQLite side and keys on the JSON side, so a field of the same
            # name would be stored once and read back twice with different
            # values -- the exact class of bug the dual-write test exists to
            # make impossible, arriving through the one route that test cannot
            # see.
            raise LedgerRefusal(
                f"{event!r} passes {sorted(shadowed)} as a field; those two "
                f"names belong to the record itself")
        record = {"ts": time.time() if ts is None else float(ts),
                  "event": event, **fields}
        line = json.dumps(record, default=str) + "\n"
        cycle = self.cycle if cycle is None else int(cycle)
        db = self._connect()
        with self._lock:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    "INSERT INTO events (cycle, ts, event, fields) "
                    "VALUES (?, ?, ?, ?)",
                    (cycle, record["ts"], event,
                     json.dumps({k: v for k, v in record.items()
                                 if k not in ("ts", "event")}, default=str)))
                if self.journal_path is not None:
                    _append_line(self.journal_path, line)
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")
        return record

    def record_result(self, measurement: Measurement) -> None:
        """Record a terminal measurement.  Refuses non-finite and unpaired.

        The refusals are layered on purpose.  :class:`kernel.model.SlotScore`
        makes a non-finite number UNCONSTRUCTIBLE — that is the type property
        the plan buys pydantic for, and it is the one that cannot be forgotten
        at a call site.  What this method adds is the refusal that a type cannot
        express: that the thing handed to the store is a ``Measurement`` at all,
        and not a dict some caller assembled with a fabricated ``score`` of 0.0.

        Idempotent on ``trial_id``.  Re-recording the identical row is a no-op
        (a resumed driver re-walking a finished trial must not double-count it);
        re-recording a DIFFERENT row for the same trial is a refusal, because
        one trial with two answers is a bug wherever it came from.
        """
        if not isinstance(measurement, Measurement):
            raise LedgerRefusal(
                f"record_result takes a kernel.model.Measurement, not "
                f"{type(measurement).__name__}. The store accepts evidence, and "
                f"evidence is a typed object whose score cannot hold a NaN and "
                f"cannot exist on a void.")
        score = measurement.score
        if measurement.outcome in (Outcome.SCORED, Outcome.INERT):
            if score is None:                       # unreachable via the model
                raise LedgerRefusal(
                    f"{measurement.trial_id}: outcome {measurement.outcome.value} "
                    f"carries no score")
            for slot, delta in score.deltas.items():
                _refuse_non_finite(delta, f"{measurement.trial_id} slot {slot!r}")
            _refuse_non_finite(score.aggregate, f"{measurement.trial_id} aggregate")
            if score.total_weight <= 0.0 or not score.deltas:
                raise LedgerRefusal(
                    f"{measurement.trial_id}: every slot is unpaired, so there "
                    f"is no denominator and no delta. Same-seed paired deltas "
                    f"are the scarcest thing this store holds -- 351 of 2,866 "
                    f"migrated rows, fewer than the rows whose seed was never "
                    f"stamped at all -- and an unpaired row does not add one.")
        row = self._result_row(measurement)
        payload = json.loads(row["measurement"])
        db = self._connect()
        with self._lock:
            db.execute("BEGIN IMMEDIATE")
            try:
                existing = db.execute(
                    "SELECT measurement FROM results WHERE trial_id = ?",
                    (measurement.trial_id,)).fetchone()
                if existing is not None:
                    if json.loads(existing["measurement"]) != payload:
                        raise LedgerRefusal(
                            f"{measurement.trial_id} is already recorded with a "
                            f"different result. One trial has one answer; "
                            f"overwriting it is how a re-run silently replaces "
                            f"the measurement a decision was made on.")
                    db.execute("COMMIT")
                    return
                db.execute(
                    "INSERT INTO results ({}) VALUES ({})".format(
                        ", ".join(row), ", ".join("?" * len(row))),
                    tuple(row.values()))
                if self.commit_ledger_path is not None:
                    self._write_commit_ledger_row(measurement, row)
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")

    def record_receipt(self, trial_id: str, *, lineage_id: str, track: str,
                       cycle: int | None = None, n: int | None = None,
                       exp_id: str | None = None, job_id: str | None = None,
                       worktree: str | None = None,
                       output_dir: str | None = None,
                       config: str | None = None) -> None:
        """Bind a dispatched trial to its job.  Keyed by TRIAL, not lineage.

        DEFECT 3, CLOSED ON BOTH SIDES HERE.  Commit 4 fixed the SQLite key and
        left the JSON half at the v1 address on purpose; this is the commit that
        moves the address.  Both projections are now keyed by trial: the row by
        ``trial_id``, the file by ``{n}`` in ``Identity.receipt``.

        ``n`` IS THEREFORE LOAD-BEARING FOR THE JSON HALF, AND ITS ABSENCE IS
        NOT SILENT.  A receipt with no ordinal has no distinct address, so it
        would land back on trial 1's file and overwrite it — the defect, exactly,
        with a trial-keyed row above it saying otherwise.  When ``n`` is None the
        SQLite row is still written (one trial, one row, always) and the JSON
        half is SKIPPED rather than aimed at a wrong address.

        The JSON write is a MERGE, matching ``_write_receipt``: a later enrich
        must never clobber a real binding with a ``None``.
        """
        if not trial_id:
            raise LedgerRefusal(
                "a receipt with no trial_id is the lineage-keyed defect "
                "wearing a new name")
        cycle = self.cycle if cycle is None else int(cycle)
        stamp = time.time()
        db = self._connect()
        with self._lock:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    "INSERT INTO receipts (trial_id, lineage_id, track, cycle, "
                    "n, exp_id, job_id, worktree, output_dir, config, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(trial_id) DO UPDATE SET "
                    "  lineage_id = excluded.lineage_id, "
                    "  track      = excluded.track, "
                    "  cycle      = excluded.cycle, "
                    "  n          = coalesce(excluded.n, receipts.n), "
                    "  exp_id     = coalesce(excluded.exp_id, receipts.exp_id), "
                    "  job_id     = coalesce(excluded.job_id, receipts.job_id), "
                    "  worktree   = coalesce(excluded.worktree, receipts.worktree), "
                    "  output_dir = coalesce(excluded.output_dir, receipts.output_dir), "
                    "  config     = coalesce(excluded.config, receipts.config), "
                    "  ts         = excluded.ts",
                    (trial_id, lineage_id, track, cycle, n, exp_id, job_id,
                     worktree, output_dir, config, stamp))
                if self.receipt_path is not None and n is not None:
                    _merge_json(
                        Path(self.receipt_path.format(
                            track=track, lineage_id=lineage_id, n=int(n))),
                        _drop_none({
                            "lineage_id": lineage_id, "track": track,
                            "trial_id": trial_id, "exp_id": exp_id,
                            "config": config, "trial": n, "job_id": job_id,
                            "worktree": worktree, "output_dir": output_dir,
                            "timestamp": stamp,
                        }))
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")

    # -- reads (only what the dual-write test needs to compare) -------------

    def replay(self, cycle: int | None = None) -> Replay:
        """The SQLite projection of one cycle's events, oldest first."""
        cycle = self.cycle if cycle is None else int(cycle)
        rows = self._connect().execute(
            "SELECT ts, event, fields FROM events WHERE cycle = ? ORDER BY id",
            (cycle,)).fetchall()
        return Replay(cycle, [
            {"ts": row["ts"], "event": row["event"], **json.loads(row["fields"])}
            for row in rows])

    def events(self, name: str) -> list[dict[str, Any]]:
        """Every event of one NAME, ACROSS EVERY CYCLE, oldest first.

        :meth:`replay` is per-cycle, which is right for resume and wrong for
        the two readers this exists for.  A saturated-lane record whose whole
        purpose is to stop the same lane being re-scouted in cycle N+3 cannot
        be answered by a query scoped to one cycle, and the rejection log is
        read the same way — "has this idea been refused before, and pointing at
        what" is a question about history, not about the cycle in hand.
        """
        rows = self._connect().execute(
            "SELECT cycle, ts, event, fields FROM events WHERE event = ? "
            "ORDER BY id", (str(name),)).fetchall()
        return [{"cycle": row["cycle"], "ts": row["ts"], "event": row["event"],
                 **json.loads(row["fields"])} for row in rows]

    def receipt(self, trial_id: str) -> dict[str, Any] | None:
        """One trial's receipt, or None.  ``supervise.wait`` reads this."""
        row = self._connect().execute(
            "SELECT * FROM receipts WHERE trial_id = ?", (trial_id,)).fetchone()
        return dict(row) if row is not None else None

    def receipts(self, lineage_id: str | None = None) -> list[dict[str, Any]]:
        """Every receipt, or every receipt of one lineage.  N trials, N rows."""
        db = self._connect()
        if lineage_id is None:
            rows = db.execute("SELECT * FROM receipts ORDER BY ts").fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM receipts WHERE lineage_id = ? ORDER BY ts",
                (lineage_id,)).fetchall()
        return [dict(row) for row in rows]

    def result(self, trial_id: str) -> dict[str, Any] | None:
        """One trial's measurement, or None.  Symmetric with :meth:`receipt`.

        ``kernel.ingest`` reads this to tell "wrote a row" from "found the row
        already there", which is the difference between a migration report that
        says what it did and one that repeats its first run's numbers forever.
        """
        row = self._connect().execute(
            "SELECT * FROM results WHERE trial_id = ?", (str(trial_id),)
        ).fetchone()
        return dict(row) if row is not None else None

    def results(self, cycle: int | None = None) -> list[dict[str, Any]]:
        """Recorded measurements for one cycle, oldest first."""
        cycle = self.cycle if cycle is None else int(cycle)
        rows = self._connect().execute(
            "SELECT * FROM results WHERE cycle = ? ORDER BY ts", (cycle,)
        ).fetchall()
        return [dict(row) for row in rows]

    # -- the query half (plan section 2, deferred by commit 4) --------------
    #
    # Commit 4 deferred these FIVE methods with a stated reason: they need a
    # migrated archive to answer anything, so writing them then would have
    # meant writing them against fixtures, and a fixture-shaped query only
    # certifies the fixture.  Commit 5 supplies the archive (``kernel.ingest``),
    # so every one of them is exercised below against REAL migrated rows.
    #
    # THE HONESTY RULE THEY ALL SHARE: a query returns what the store HOLDS and
    # says so, never a number reconstructed from an assumption about what the
    # store ought to hold.  Where the archive cannot answer, the answer is an
    # explicit absence (an empty slot map, an ``unknown`` validity, a component
    # with ``n_challenges: 0``) and never a zero standing in for a measurement.
    # That is the same rule ``record_result`` enforces on the write side, and
    # it is why 178 opaque ``gate_blocked`` rows could not be re-typed per-row.
    #
    # WHY PYTHON-SIDE AGGREGATION IS CORRECT HERE, NOT LAZINESS: the plan
    # rejected duckdb on measured grounds -- ~2k rows per cycle, low thousands
    # in all history -- so every one of these is a full scan of a small table.
    # The rows are read through ``json_extract`` on the stored ``Measurement``
    # rather than through new columns, because the typed object is stored whole
    # and a denormalising migration is exactly the thing a store this size does
    # not need.

    def slot_matrix(self, cycle: int | None = None
                    ) -> dict[str, dict[str, float]]:
        """``{run_name: {slot: delta}}`` for one cycle's scored rows.

        This is what ``policy/select.retain`` reads.  The panel decomposes every
        score into declared slots, and the counterfactual study that the plan
        cites for per-instance retention was computed over exactly this object.

        READ THE CAVEAT ON WIDTH.  The number of slots per row is whatever the
        recorded ``SlotScore`` carries, and for MIGRATED rows that is normally
        ONE -- the aggregate slot -- because the per-slot vector is not in the
        archive this store was migrated from.  ``kernel.ingest`` reports that
        count explicitly rather than padding it.  A caller that wants per-slot
        argmax must therefore check the width it actually got; a one-slot matrix
        makes "the union of per-slot argmaxes" identical to top-1 by mean, and
        silently so.  Phase-4 section 6.4 already rules that the retention rule
        stays declared-but-unrelied-on until the calibration probe exists, and
        this is the second, independent reason not to switch it on yet.
        """
        out: dict[str, dict[str, float]] = {}
        for row in self._scored(cycle):
            payload = json.loads(row["measurement"])
            score = payload.get("score") or {}
            deltas = score.get("deltas") or {}
            if deltas:
                out[row["run_name"]] = {str(k): float(v)
                                        for k, v in deltas.items()}
        return out

    def mechanisms(self) -> list[str]:
        """Every mechanism signature the store has ever recorded, sorted.

        The enumeration :meth:`mechanism_outcomes` needs to be asked over the
        whole archive rather than over a list the caller already had.
        ``agent.world`` reads this: a belief set assembled from the signatures
        somebody remembered to pass in would be exactly as blind as the
        write-only kgraph it replaces.
        """
        rows = self._connect().execute(
            "SELECT DISTINCT json_extract(measurement, "
            "'$.provenance.mechanism_signature') AS mech FROM results "
            "WHERE mech IS NOT NULL AND mech != '' ORDER BY mech").fetchall()
        return [str(row["mech"]) for row in rows]

    def mechanism_outcomes(self, sig: str) -> dict[str, Any]:
        """Every outcome ever recorded for one mechanism signature.

        The cross-cycle memory an ideator needs to not re-propose a mechanism
        that has already been measured: 6.1% of proposals are re-emissions and
        7.4% are byte-identical, so the useful check is "what happened last
        time", not a novelty score.

        VOIDS ARE COUNTED, NOT DROPPED, and that is the point.  A mechanism with
        ``n_attempts: 9, n_scored: 0`` has never been tested -- it is a
        candidate, not a dead end -- while one with ``n_scored: 9`` and a best
        delta inside the noise floor has been answered.  Collapsing the two is
        how 93.6% of dead lineages got written off as "tried".
        """
        rows = [dict(r) for r in self._connect().execute(
            "SELECT * FROM results WHERE "
            "json_extract(measurement, '$.provenance.mechanism_signature') = ? "
            "ORDER BY cycle, ts", (str(sig),)).fetchall()]
        deltas = [r["aggregate"] for r in rows
                  if r["outcome"] == Outcome.SCORED.value
                  and r["aggregate"] is not None]
        voids = [r for r in rows if r["outcome"] == Outcome.VOID.value]
        return {
            "signature": str(sig),
            "n_attempts": len(rows),
            "n_scored": sum(1 for r in rows
                            if r["outcome"] == Outcome.SCORED.value),
            "n_inert": sum(1 for r in rows
                           if r["outcome"] == Outcome.INERT.value),
            "n_void": len(voids),
            "cycles": sorted({int(r["cycle"]) for r in rows}),
            "best_delta": max(deltas) if deltas else None,
            "worst_delta": min(deltas) if deltas else None,
            "validities": sorted({str(r["validity"]) for r in rows
                                  if r["validity"]}),
            "void_statuses": sorted({str(r["status"]) for r in voids
                                     if r["status"]}),
            "trials": [r["trial_id"] for r in rows],
        }

    def void_clusters(self, cycle: int | None = None
                      ) -> dict[str, dict[str, Any]]:
        """``{status: aggregate}`` -- one row per void ROOT CAUSE, not per void.

        An AGGREGATE per cause, never the raw rows.  Dumping every row is what
        made this section a 131 KB blob that the evidence cap sliced down to a
        single cause, so 11 of 12 root causes were invisible to the failure
        brief on every invocation.

        ``n_lineages`` is carried beside ``n_voids`` because they answer
        different questions: 565 cap-kills over 285 distinct lineages is one
        cycle's void storm, and reading only the first number turns a storm into
        an apparent steady-state filter.  Exemplar notes are capped at three per
        cause -- enough to reason about the cause, not enough to become the blob
        again.

        COUNTED PER TRIAL, NOT PER EVENT, and that is not a detail.  One void
        trial writes up to two rows: the per-axis ``trial_void_liveness``, which
        names WHICH check fired, and the ``trial_completed`` whose status is the
        collapsed ``void_liveness``.  Counting rows reports the same 32 voids
        twice under two spellings of one cause -- measured on cycle 8 before
        this was fixed.  The per-axis row WINS on collision, because the axis is
        the fact the collapsed status cannot express and the whole reason 178
        opaque ``gate_blocked`` rows had to be re-typed in bulk.
        """
        by_trial: dict[str, dict[str, Any]] = {}
        for event in self.replay(cycle):
            status = _void_status_of(event)
            if status is None:
                continue
            key = str(event.get("run_name")
                      or f"{event.get('lineage_id')}#{event.get('n')}")
            axis = str(event.get("event") or "").startswith("trial_void_")
            if key in by_trial and not axis:
                continue
            by_trial[key] = {
                "cause": status.removeprefix("void_") or status,
                "lineage": str(event.get("lineage_id") or ""),
                "note": str(event.get("void_reason") or event.get("notes")
                            or event.get("note") or event.get("reason")
                            or "")[:200],
                "ts": event.get("ts"),
            }
        clusters: dict[str, dict[str, Any]] = {}
        for void in by_trial.values():
            cluster = clusters.setdefault(void["cause"], {
                "status": void["cause"], "n_voids": 0, "lineages": set(),
                "exemplars": [], "last_ts": None,
            })
            cluster["n_voids"] += 1
            if void["lineage"]:
                cluster["lineages"].add(void["lineage"])
            if void["note"] and len(cluster["exemplars"]) < 3 \
                    and void["note"] not in cluster["exemplars"]:
                cluster["exemplars"].append(void["note"])
            if isinstance(void["ts"], (int, float)):
                cluster["last_ts"] = max(cluster["last_ts"] or 0.0,
                                         float(void["ts"]))
        for cluster in clusters.values():
            lineages = cluster.pop("lineages")
            cluster["n_lineages"] = len(lineages)
            cluster["lineage_ids"] = sorted(lineages)[:10]
        return dict(sorted(clusters.items(),
                           key=lambda kv: -kv[1]["n_voids"]))

    def assumption_ledger(self, base: str) -> list[dict[str, Any]]:
        """Which parts of one base have been questioned, and with what result.

        One row per component of ``base`` -- a base_ref, the commit a cycle's
        trials were authored against -- where a COMPONENT is a mechanism
        signature the store has ever recorded against that base.  Every
        component that no trial has scored is an untested assumption, i.e. a
        candidate ablation, and naming them is the whole product of this query:
        the largest idea cluster in history (303 pooling proposals) produced
        four scored lineages, so "never questioned" is the common case and
        needs to be visible rather than inferred.

        THE HONEST LIMIT, stated because the caller cannot see it otherwise:
        components here are the mechanisms the ARCHIVE attributes to the base,
        not a decomposition of the base recipe itself.  A part of the base that
        no trial ever named does not appear at all -- absence from this list is
        not evidence of coverage.  ``status`` is therefore about the challenges
        on record: ``untested`` (nothing scored), ``held`` (scored, nothing
        cleared the floor) or ``displaced`` (a scored challenge cleared it).
        """
        rows = [dict(r) for r in self._connect().execute(
            "SELECT *, json_extract(measurement, "
            "'$.provenance.mechanism_signature') AS mech "
            "FROM results WHERE base_ref = ? ORDER BY cycle, ts",
            (str(base),)).fetchall()]
        by_component: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            component = row["mech"] or row["lineage_id"]
            by_component.setdefault(str(component), []).append(row)
        out: list[dict[str, Any]] = []
        for component, challenges in sorted(by_component.items()):
            scored = [c for c in challenges
                      if c["outcome"] == Outcome.SCORED.value
                      and c["aggregate"] is not None]
            best = max((c["aggregate"] for c in scored), default=None)
            if not scored:
                status = "untested"
            elif best is not None and best > _NOISE_FLOOR:
                status = "displaced"
            else:
                status = "held"
            out.append({
                "component": component,
                "base": str(base),
                "status": status,
                "n_challenges": len(challenges),
                "n_scored": len(scored),
                "best_delta": best,
                "evidence": [{"trial": c["trial_id"], "cycle": c["cycle"],
                              "outcome": c["outcome"], "validity": c["validity"],
                              "paired_delta": c["aggregate"]}
                             for c in challenges],
            })
        return out

    def best_path(self) -> dict[str, Any]:
        """Claimed per-cycle gain against the baseline spine that actually ran.

        The dissipation question, and the only one in this file that can be
        wrong in a way nobody notices: a cycle CLAIMS the sum of its positive
        paired deltas, while the spine records what the next cycle's baseline
        actually scored.  Measured over the archive those two disagree by
        roughly the whole claim -- which is the quantified form of "the plateau
        is a ceiling", and it is invisible unless both halves are read off the
        same store.

        ``claimed`` deliberately sums only deltas whose validity is a SAME-SEED
        pair.  A cross-seed or unknown-seed delta is not a claim about the base
        it is stacked on, and summing it here is exactly the arithmetic that
        produced a banked-gain figure of +0.0137 against a measured root-to-
        endpoint drift of -0.0009.
        """
        spine: dict[int, dict[str, Any]] = {}
        claimed: dict[int, dict[str, Any]] = {}
        for row in self._connect().execute(
                "SELECT *, json_extract(measurement, '$.provenance.run_type') "
                "AS run_type FROM results ORDER BY cycle, ts"):
            row = dict(row)
            cycle = int(row["cycle"])
            if str(row["run_type"] or "").endswith("baseline"):
                # A cycle has ONE baseline; a later one supersedes an earlier.
                spine[cycle] = {"cycle": cycle, "run": row["run_name"],
                                "seed": row["seed"],
                                "score": _base_score(row)}
                continue
            bucket = claimed.setdefault(cycle, {"sum": 0.0, "n": 0, "wins": []})
            delta = row["aggregate"]
            if delta is None or row["validity"] != "paired_same_seed":
                continue
            if delta > 0.0:
                bucket["sum"] += float(delta)
                bucket["n"] += 1
                bucket["wins"].append((row["run_name"], float(delta)))
        per_cycle: list[dict[str, Any]] = []
        previous: float | None = None
        for cycle in sorted(spine):
            step = spine[cycle]["score"]
            bucket = claimed.get(cycle, {"sum": 0.0, "n": 0, "wins": []})
            per_cycle.append({
                "cycle": cycle,
                "baseline": spine[cycle]["run"],
                "base_score": step,
                "actual_step": (None if previous is None or step is None
                                else round(step - previous, 6)),
                "claimed_paired_wins": round(bucket["sum"], 6),
                "n_paired_wins": bucket["n"],
                "wins": sorted(bucket["wins"], key=lambda w: -w[1])[:5],
            })
            if step is not None:
                previous = step
        scores = [c["base_score"] for c in per_cycle
                  if c["base_score"] is not None]
        actual = (round(scores[-1] - scores[0], 6) if len(scores) >= 2
                  else None)
        total = round(sum(c["claimed_paired_wins"] for c in per_cycle), 6)
        return {
            "spine": [{"cycle": c["cycle"], "baseline": c["baseline"],
                       "score": c["base_score"]} for c in per_cycle],
            "per_cycle": per_cycle,
            "cumulative_claimed_paired": total,
            "actual_root_to_endpoint": actual,
            "dissipation_gap": (None if actual is None
                                else round(total - actual, 6)),
        }

    # -- ingest bookkeeping ------------------------------------------------
    #
    # The cursor lives WITH the store rather than in ``kernel.ingest`` because
    # it is the thing that makes a re-run of ``ovx migrate`` a no-op, and a
    # bookkeeping table kept anywhere else would be a second writer to the file
    # this class exists to be the only writer of.  One row per imported
    # artifact: the etag proves the file has not changed, ``n_rows`` says how
    # much of it has already been consumed, so an APPENDED journal imports only
    # its new lines instead of duplicating every old one.

    def cursor(self, key: str) -> dict[str, Any] | None:
        """What an earlier import already consumed from ``key``, or None."""
        db = self._connect()
        db.execute(_CURSOR_SCHEMA)
        row = db.execute("SELECT * FROM ingest_cursor WHERE source_key = ?",
                         (str(key),)).fetchone()
        return dict(row) if row is not None else None

    def record_cursor(self, key: str, etag: str, n_rows: int) -> None:
        """Mark ``n_rows`` of ``key`` consumed at ``etag``."""
        db = self._connect()
        db.execute(_CURSOR_SCHEMA)
        with self._lock:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    "INSERT INTO ingest_cursor (source_key, etag, n_rows, ts) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(source_key) DO UPDATE SET "
                    "etag = excluded.etag, n_rows = excluded.n_rows, "
                    "ts = excluded.ts",
                    (str(key), str(etag), int(n_rows), time.time()))
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")

    def backup(self, dest: Path | str) -> Path:
        """A consistent copy taken through SQLite's own backup API.

        ``cp`` on a WAL database can copy a main file whose newest commits are
        still in the ``-wal``; the backup API cannot.
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(dest)
        try:
            self._connect().backup(target)
        finally:
            target.close()
        return dest

    # -- internals ----------------------------------------------------------

    def _scored(self, cycle: int | None) -> list[sqlite3.Row]:
        """One cycle's rows that carry a number, oldest first.

        INERT is included beside SCORED on purpose: an inert run is a real
        measurement, of the baseline, and dropping it from the matrix would
        make a no-op diff look like a missing trial rather than a null result.
        """
        cycle = self.cycle if cycle is None else int(cycle)
        return self._connect().execute(
            "SELECT * FROM results WHERE cycle = ? AND aggregate IS NOT NULL "
            "AND outcome IN (?, ?) ORDER BY ts",
            (cycle, Outcome.SCORED.value, Outcome.INERT.value)).fetchall()

    def _result_row(self, m: Measurement) -> dict[str, Any]:
        """The SQLite row for a measurement, with the JSON payload alongside.

        The typed object is stored whole in ``measurement`` and the columns are
        a denormalised index over it.  That way a query added at commit 6 never
        needs a migration to reach a field the row already carried.
        """
        return {
            "trial_id": m.trial_id,
            "lineage_id": m.lineage_id,
            "cycle": int(m.cycle),
            "outcome": m.outcome.value,
            "run_name": str(m.provenance.get("run_name") or m.trial_id),
            "base_ref": m.base_ref,
            "seed": m.seed,
            "head_commit": m.head_commit,
            "job_id": m.job_id,
            "output_dir": m.output_dir,
            "validity": m.validity,
            "status": m.status,
            "aggregate": None if m.score is None else m.score.aggregate,
            "total_weight": None if m.score is None else m.score.total_weight,
            "ts": time.time(),
            "measurement": m.model_dump_json(),
        }

    def _write_commit_ledger_row(self, m: Measurement,
                                 row: Mapping[str, Any]) -> None:
        """The v1 ``commit_ledger.json`` half of :meth:`record_result`.

        Same shape ``commit_full`` and ``cmd_intake`` write — ``{"commits":
        {run_name: {...}}}`` — because commit 6 deletes those writers and the
        rows they left must still read back identically until it does.  The
        difference is WHERE the read-modify-write happens: inside the SQLite
        write transaction, so there is no window between the load and the save
        for another process to interleave.  That is defect 2.
        """
        path = self.commit_ledger_path
        assert path is not None
        data = _load_json(path)
        if not isinstance(data, dict):
            data = {}
        commits = data.get("commits")
        if not isinstance(commits, dict):
            commits = {}
            data["commits"] = commits
        commits[row["run_name"]] = _drop_none({
            "run_name": row["run_name"],
            "run_type": m.status or "full",
            "score": None if m.score is None else m.score.aggregate,
            "paired_delta": None if m.score is None else m.score.aggregate,
            "output_dir": m.output_dir or None,
            "lineage": m.lineage_id,
            "track": m.provenance.get("track"),
            "trial_id": m.trial_id,
            "outcome": m.outcome.value,
            "committed_at": row["ts"],
            "source": "ledger",
        })
        _atomic_write_json(path, data)


# ---------------------------------------------------------------------------
# File primitives.  Small, local, and NOT imported from ``atomicio`` — kernel
# imports nothing above it, and ``atomicio`` is on the delete list anyway.
# ---------------------------------------------------------------------------


def _append_line(path: Path, line: str) -> None:
    """Append one newline-terminated line, repairing a torn predecessor.

    THIS IS DEFECT 1.  The v1 appender opened ``"a"`` and wrote ``line``.  If a
    previous append was cut short mid-write the file ends without a newline, and
    the next line is concatenated onto it — producing one unparseable record out
    of two events, so a single bad shutdown cost the torn event AND the good one
    after it.  Reading the last byte and emitting a leading ``\\n`` when it is
    not one bounds the damage to the torn event alone.

    The repair newline and the line are written in ONE ``write`` call, so the
    "single write of one newline-terminated line" property the v1 appender
    documented is preserved rather than traded away for the fix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        prefix = ""
        if end:
            handle.seek(end - 1)
            if handle.read(1) != "\n":
                prefix = "\n"
            handle.seek(0, os.SEEK_END)
        handle.write(prefix + line)
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: Path) -> Any:
    """Read a JSON file, tolerating absence and garbage.

    ``parse_constant`` is wired to raise: ``json.loads`` accepts ``NaN`` by
    default, and a store that leans on that leniency is readable by CPython and
    by nothing else.
    """
    try:
        return json.loads(path.read_text(), parse_constant=_reject_constant)
    except (OSError, ValueError):
        return None


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON via a temp file and ``os.replace`` — no torn readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    text = json.dumps(data, indent=2, default=str, allow_nan=False)
    with open(tmp, "w") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _merge_json(path: Path, updates: Mapping[str, Any]) -> None:
    """Merge-enrich a JSON object file, never clobbering with a ``None``."""
    existing = _load_json(path)
    if not isinstance(existing, dict):
        existing = {}
    existing.update(updates)
    _atomic_write_json(path, existing)


def _drop_none(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in mapping.items() if v is not None}


def _refuse_non_finite(value: float, what: str) -> None:
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise LedgerRefusal(
            f"{what} is {value!r}. JSON has no literal for NaN or Infinity, so "
            f"a store that accepted one would be unreadable by a strict parser "
            f"— and the failure would surface as 'the ledger is corrupt', "
            f"arbitrarily far from the trial that caused it.")


def _void_status_of(event: Mapping[str, Any]) -> str | None:
    """The void status one journal event reports, or None if it is not a void.

    Two shapes, deliberately both: the per-axis event names the rebuild writes
    (``trial_void_liveness``), and the v1 ``trial_completed`` whose ``status``
    field carried the reason.  Returning the STATUS rather than the event name
    is what lets the two spellings cluster together.

    A ``trial_completed`` is a void when IT CARRIES NO NUMBER — no score and no
    paired delta — which is ``Outcome.VOID``'s own definition rather than a
    keyword list over status strings.  That distinction is load-bearing: the
    live archive spells its voids nineteen different ways (``setup_failed``,
    ``smoke_bypass_fail``, ``backend_unavailable``, ``nan_loss`` …), a prefix
    list misses whichever ones it was not shown, and a missed void is silently
    a cluster that never gets reported.  The status is still what the cause is
    keyed by; it is just not what decides.
    """
    name = str(event.get("event") or "")
    if name.startswith("trial_void_"):
        return name[len("trial_void_"):]
    if name != "trial_completed":
        return None
    if isinstance(event.get("score"), (int, float)) \
            or isinstance(event.get("paired_delta"), (int, float)):
        return None
    return str(event.get("status") or "") or "unstated"


def _base_score(row: Mapping[str, Any]) -> float | None:
    """A baseline row's ABSOLUTE panel score, not its delta.

    The spine is a sequence of levels and the trials are a sequence of steps;
    reading a baseline's ``aggregate`` (which is its delta against the previous
    base, and is 0.0 by construction for a reseed) as its score is how a
    dissipation report comes out flat no matter what happened.
    """
    payload = row.get("measurement")
    if not payload:
        return None
    try:
        provenance = json.loads(payload).get("provenance") or {}
    except ValueError:
        return None
    score = provenance.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite constant {name!r} in a durable store")
