"""loop.store — ``SqliteStore``, the task-scoped store (DESIGN §2): one table per contract record over
the kernel ``Ledger`` connection. Reads filter on ``task_id``, writes refuse a foreign record, and an
untagged (v1) row raises ``ForeignTaskError`` on read instead of being silently dropped."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
import time
import types
import typing
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from kernel.ledger import Ledger
from kernel.stats import paired_delta
from loop import contracts as C
from loop.core.seeds import SeedBlock
from loop.spec import build, encode


def _q(col: str) -> str:
    return f'"{col}"'


def _is_json(tp: Any) -> bool:
    args = [a for a in typing.get_args(tp) if a is not type(None)]
    if typing.get_origin(tp) in (typing.Union, types.UnionType) and len(args) == 1:
        tp = args[0]
    return not (tp in (bool, int, float, str) or (isinstance(tp, type) and issubclass(tp, Enum)))


MEMO_MAX = 200_000        # decoded rows kept per table (cleared whole when full)


class _Table:
    def __init__(self, name: str, cls: type, key: str = "PRIMARY KEY (id)", *index: str) -> None:
        self.name, self.cls = name, cls
        self._memo: dict[tuple[Any, ...], Any] = {}
        hints = typing.get_type_hints(cls)
        self.json = {f.name: _is_json(hints[f.name]) for f in dataclasses.fields(cls)}
        cols = ", ".join([_q(n) for n in self.json] + ([key] if key else []))
        self.ddl = [f"CREATE TABLE IF NOT EXISTS {name} ({cols})"] + [
            f'CREATE INDEX IF NOT EXISTS ix_{name}_{c} ON {name}("{c}")' for c in ("task_id", *index)]

    def to_row(self, obj: Any) -> dict[str, Any]:
        return {n: (None if getattr(obj, n) is None else json.dumps(encode(getattr(obj, n)),
                                                                     sort_keys=True))
                if js else encode(getattr(obj, n)) for n, js in self.json.items()}

    def from_row(self, row: Any) -> Any:
        """The decoded record. v3 review O2: memoised on the raw row values (the records are frozen and nothing
        mutates them; the same bytes decode to the same record) -- a tick listed every node ~30 times and each
        listing re-ran json + the strict ``build`` (live 10-14 profile: 82 s of a 2-tick shadow in from_row)."""
        key = tuple(row)
        if (got := self._memo.get(key)) is not None:
            return got
        got = build(self.cls, {n: json.loads(row[n]) if js and row[n] is not None else row[n]
                               for n, js in self.json.items()}, self.name)
        if len(self._memo) >= MEMO_MAX:
            self._memo.clear()
        self._memo[key] = got
        return got


_T = {t.name: t for t in (
    _Table("nodes", C.Node, "PRIMARY KEY (id)", "status", "diff_hash"),
    _Table("runs", C.Run, "PRIMARY KEY (id), UNIQUE (node_id, seed, stage)", "node_id"),
    _Table("hypotheses", C.HypothesisRecord, "PRIMARY KEY (id)", "kind"),
    _Table("postmortems", C.Postmortem, "PRIMARY KEY (id)", "node_id"),
    _Table("handoffs", C.HandoffNote, "PRIMARY KEY (id)", "node_id"),
    _Table("reviews", C.ReviewReport),
    _Table("arm_pulls", C.ArmPull, ""),
)}
_ORDER = {"runs": "rowid", "arm_pulls": "rowid"}          # others: created_ts, rowid
_OTHER_DDL = [
    "CREATE TABLE IF NOT EXISTS hypothesis_links (task_id TEXT NOT NULL, hypothesis_id TEXT, "
    "node_id TEXT, sign INTEGER, auto INTEGER, PRIMARY KEY (hypothesis_id, node_id))",
    "CREATE TABLE IF NOT EXISTS state (task_id TEXT NOT NULL, key TEXT, value TEXT, "
    "PRIMARY KEY (task_id, key))",
    "CREATE TABLE IF NOT EXISTS diagnostics (run_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, node_id TEXT, "
    "seed INTEGER, stage TEXT, ts REAL, version INTEGER, features TEXT)",
    "CREATE INDEX IF NOT EXISTS ix_diagnostics_node ON diagnostics(node_id)",
    "CREATE TABLE IF NOT EXISTS log (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, "
    "ts REAL, event TEXT, fields TEXT)",
    # loop.core.seeds (v3): seed pools; additive, the old tables are untouched
    "CREATE TABLE IF NOT EXISTS seed_blocks (id TEXT PRIMARY KEY, task_id TEXT NOT NULL, pool TEXT, seeds TEXT, "
    "node_ids TEXT, purpose TEXT, drawn_ts REAL, closed_ts REAL)",
    # loop.core.meta (v3): versions of the TUNABLE config / the prompts; additive
    "CREATE TABLE IF NOT EXISTS config_versions (task_id TEXT NOT NULL, kind TEXT NOT NULL, v INTEGER NOT NULL, "
    "parent INTEGER, ts REAL, diff TEXT, reason TEXT, backtest TEXT, state TEXT, PRIMARY KEY (task_id, kind, v))",
]
CONFIG_VERSION_KINDS = ("config", "prompt")


class StoreClosed(sqlite3.ProgrammingError):
    """A read on a store after :meth:`SqliteStore.close` (a clear error instead of sqlite's ProgrammingError /
    a SIGSEGV on a connection closed under the query)."""


class SqliteStore:
    """The store of ONE task id over a SQLite file."""

    def __init__(self, db_path: str | Path, task_id: str, *,
                 weights: Mapping[str, float] | None = None, clock: Any = time.time) -> None:
        if not task_id:
            raise ValueError("SqliteStore needs a non-empty task_id")
        self.task_id, self._weights, self._clock = task_id, dict(weights or {}), clock
        self._ledger = Ledger(db_path)
        self._db = self._ledger.connection()
        self._lock = threading.RLock()
        # v3-c6a (ideate waited 436 s on this lock: 8 session tool threads + render threads + the tick shared it):
        # READS go through a per-thread connection on the WAL file (autocommit: every committed write is visible);
        # WRITES keep the one connection + lock, whose wait is timed (lock_stats)
        self._path, self._local, self._closed = Path(db_path), threading.local(), False
        self._wal = str(self._db.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
        self._stats = {"lock_wait_s": 0.0, "lock_wait_max_s": 0.0, "lock_waits": 0, "reads": 0, "locked_reads": 0}
        self._slock = threading.Lock()      # the read counters (bumped from every reader thread, outside the write lock)
        with self._lock:
            for stmt in [s for t in _T.values() for s in t.ddl] + _OTHER_DDL:
                self._db.execute(stmt)
            for t in _T.values():                 # a field added later: its column, at its default
                have = {r[1] for r in self._db.execute(f"PRAGMA table_info({t.name})")}
                for f in dataclasses.fields(t.cls):
                    if f.name not in have and isinstance(f.default, (str, int, float, tuple)):
                        d = json.dumps(f.default) if isinstance(f.default, tuple) else f.default
                        self._db.execute(f"ALTER TABLE {t.name} ADD COLUMN {_q(f.name)} DEFAULT "
                                         + ("'" + d.replace("'", "''") + "'" if isinstance(d, str) else repr(d)))
            if "verdicts" not in {r[1] for r in self._db.execute("PRAGMA table_info(seed_blocks)")}:
                self._db.execute("ALTER TABLE seed_blocks ADD COLUMN verdicts TEXT")   # v3 fix A: frozen verdicts

    def close(self) -> None:
        with self._lock:          # never under a query on another thread (a closed-under sqlite3 query SIGSEGVs)
            self._closed = True
            mine = getattr(self._local, "db", None)
            if mine is not None:      # this thread's reader; another thread's may be mid-query: dropped, never closed
                mine.close()           # under it (it is closed when its thread / this store is collected)
                self._local.db = None
            self._ledger.close()

    # -- v3-c6a: per-thread read connections + timed write lock -------------------------------------------------
    @contextmanager
    def _locked(self) -> Any:
        """The write lock, its wait timed (``lock_stats``)."""
        t0 = time.monotonic()
        self._lock.acquire()
        w = time.monotonic() - t0
        with self._slock:
            st = self._stats
            st["lock_wait_s"] += w
            st["lock_waits"] += 1
            st["lock_wait_max_s"] = max(st["lock_wait_max_s"], w)
        try:
            yield
        finally:
            self._lock.release()

    def lock_stats(self, reset_max: bool = False) -> dict[str, float]:
        """Cumulative: seconds waited for the write lock (total / max since the last ``reset_max``), lock
        acquisitions, reads on the per-thread connections, reads that had to take the lock (an open transaction /
        no WAL)."""
        with self._slock:
            out = dict(self._stats)
            if reset_max:
                self._stats["lock_wait_max_s"] = 0.0
        return out

    def _reader(self) -> sqlite3.Connection | None:
        """This thread's read connection (WAL only; None: the shared connection under the lock)."""
        if not self._wal or self._closed:
            return None
        c = getattr(self._local, "db", None)
        if c is None:
            c = sqlite3.connect(self._path, timeout=30.0, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA busy_timeout=30000")
            c.execute("PRAGMA query_only=ON")
            self._local.db = c
        return c

    def _read(self, sql: str, args: Any = ()) -> list[Any]:
        """Rows of a SELECT: on this thread's read connection, unless a transaction is open on the shared one (its
        own writes are visible only there: ``factorial.atomic``) or the file is not WAL."""
        if self._closed:
            raise StoreClosed(f"SqliteStore({self._path.name}, {self.task_id!r}) is closed: a read after close()")
        c = None if self._db.in_transaction else self._reader()
        if c is None:
            with self._slock:
                self._stats["locked_reads"] += 1
            with self._locked():
                if self._closed:
                    raise StoreClosed(f"SqliteStore({self._path.name}, {self.task_id!r}) is closed: a read after close()")
                return self._db.execute(sql, args).fetchall()
        with self._slock:
            self._stats["reads"] += 1
        return c.execute(sql, args).fetchall()

    # -- plumbing -------------------------------------------------------------
    def _check(self, record: Any) -> None:
        C.require_task(self.task_id, [record], where="store:write")

    def _insert(self, table: str, record: Any) -> None:
        self._check(record)
        row = _T[table].to_row(record)
        with self._locked():
            self._db.execute(f"INSERT INTO {table} ({', '.join(map(_q, row))}) "
                             f"VALUES ({', '.join('?' for _ in row)})", list(row.values()))

    def _select(self, table: str, **eq: Any) -> list[Any]:
        """Rows of this task AND untagged rows: an untagged row is refused, never skipped."""
        sql, args = f"SELECT * FROM {table} WHERE task_id IN (?, '')", [self.task_id]
        for col, val in eq.items():
            if val is None:
                continue
            vals = [v.value if isinstance(v, Enum) else v for v in
                    (val if isinstance(val, (list, tuple, set, frozenset)) else [val])]
            sql += f' AND "{col}" IN ({", ".join("?" for _ in vals)})'
            args += vals
        rows = self._read(f"{sql} ORDER BY {_ORDER.get(table, 'created_ts, rowid')}", args)
        return C.require_task(self.task_id, [_T[table].from_row(r) for r in rows],
                              where=f"store:{table}")

    def _get(self, table: str, id_: str) -> Any:
        rows = self._select(table, id=id_)
        return rows[0] if rows else None

    def _update(self, table: str, id_: str, fields: Mapping[str, Any]) -> Any:
        with self._locked():
            cur = self._get(table, id_)
            if cur is None:
                raise KeyError(f"{table}: no row {id_!r} for task {self.task_id!r}")
            if {"id", "task_id"} & set(fields):
                raise ValueError(f"{table}: cannot update id / task_id")
            row = _T[table].to_row(dataclasses.replace(cur, **fields))
            self._db.execute(f"UPDATE {table} SET {', '.join(_q(c) + '=?' for c in row)} "
                             f"WHERE id=? AND task_id=?", [*row.values(), id_, self.task_id])
        return self._get(table, id_)

    @staticmethod
    def _list(vals: Iterable[Any] | None) -> Any:
        return None if vals is None else list(vals)

    # -- nodes / runs / hypotheses / artifacts -------------------------------------
    def put_node(self, node: C.Node) -> None:
        self._insert("nodes", node)

    def update_node(self, node_id: str, **fields: Any) -> C.Node:
        return self._update("nodes", node_id, {"updated_ts": self._clock(), **fields})

    def get_node(self, node_id: str) -> C.Node | None:
        return self._get("nodes", node_id)

    def nodes(self, *, status: Iterable[C.NodeStatus] | None = None,
              island: str | None = None) -> list[C.Node]:
        status = self._list(status)
        return [] if status == [] else self._select("nodes", status=status, island=island)

    def find_by_diff_hash(self, diff_hash: str) -> C.Node | None:
        rows = self._select("nodes", diff_hash=diff_hash)
        return rows[0] if rows else None

    def ensure_root(self, commit: str) -> C.Node:
        """Create the baseline node if absent (idempotent); return it."""
        with self._locked():
            root = self.get_node(C.ROOT_NODE_ID)
            if root is None:
                now = self._clock()
                root = C.Node(id=C.ROOT_NODE_ID, task_id=self.task_id, commit=commit, parent_id=None,
                              strategy=C.Strategy.BASELINE, operator=None, family="baseline",
                              mechanism_tag="baseline", diff_hash="", status=C.NodeStatus.BASELINE,
                              created_ts=now, updated_ts=now)
                self.put_node(root)
            return root

    def put_run(self, run: C.Run) -> None:
        self._insert("runs", run)                 # sqlite3.IntegrityError on a double launch

    def update_run(self, run_id: str, **fields: Any) -> C.Run:
        return self._update("runs", run_id, fields)

    def get_run(self, run_id: str) -> C.Run | None:
        return self._get("runs", run_id)

    def runs(self, *, node_id: str | None = None, status: Iterable[C.RunStatus] | None = None,
             stage: C.Stage | None = None) -> list[C.Run]:
        status = self._list(status)
        return [] if status == [] else self._select("runs", node_id=node_id, status=status,
                                                    stage=stage)

    def put_hypothesis(self, h: C.HypothesisRecord) -> None:
        self._insert("hypotheses", h)

    def update_hypothesis(self, hypothesis_id: str, **fields: Any) -> C.HypothesisRecord:
        return self._update("hypotheses", hypothesis_id, fields)

    def get_hypothesis(self, hypothesis_id: str) -> C.HypothesisRecord | None:
        return self._get("hypotheses", hypothesis_id)

    def hypotheses(self, *, kind: C.HypothesisKind | None = None) -> list[C.HypothesisRecord]:
        return self._select("hypotheses", kind=kind)

    def link(self, hypothesis_id: str, node_id: str, *, sign: int, auto: bool) -> None:
        if sign not in (-1, 1):
            raise ValueError(f"link sign must be +1/-1, got {sign!r}")
        with self._locked():
            self._db.execute("INSERT OR REPLACE INTO hypothesis_links VALUES (?,?,?,?,?)",
                             (self.task_id, hypothesis_id, node_id, sign, int(auto)))

    def links(self, hypothesis_id: str, *, explicit_only: bool = False) -> list[tuple[str, int]]:
        """(node id, sign) in link order; ``explicit_only`` keeps the ``auto=False`` rows (a node's own
        hypothesis, a review probe's node) and drops the derived auto-links."""
        rows = self._read("SELECT node_id, sign FROM hypothesis_links WHERE task_id=? "
                          "AND hypothesis_id=?" + (" AND auto=0" if explicit_only else "")
                          + " ORDER BY rowid", (self.task_id, hypothesis_id))
        return [(r["node_id"], int(r["sign"])) for r in rows]

    def put_postmortem(self, pm: C.Postmortem) -> None:
        self._insert("postmortems", pm)

    def postmortems(self, *, node_id: str | None = None) -> list[C.Postmortem]:
        return self._select("postmortems", node_id=node_id)

    def put_handoff(self, note: C.HandoffNote) -> None:
        self._insert("handoffs", note)

    def handoffs(self, *, node_id: str | None = None) -> list[C.HandoffNote]:
        return self._select("handoffs", node_id=node_id)

    def put_review(self, report: C.ReviewReport) -> None:
        self._insert("reviews", report)

    def reviews(self) -> list[C.ReviewReport]:
        return self._select("reviews")

    def record_pull(self, pull: C.ArmPull) -> None:
        self._insert("arm_pulls", pull)

    def pulls(self) -> list[C.ArmPull]:
        return self._select("arm_pulls")

    def drop_pulls(self, node_id: str) -> None:
        """Forget a node's bandit credit (a verdict corrected after the fact is paid again at its new close)."""
        with self._locked():
            self._db.execute("DELETE FROM arm_pulls WHERE task_id=? AND node_id=?", (self.task_id, node_id))

    # -- run diagnostics (loop.diagnostics) -------------------------------------
    def put_diagnostics(self, run_id: str, *, node_id: str, seed: int, stage: Any, features: Mapping[str, float],
                        version: int = 1, ts: float | None = None) -> None:
        """One run's extracted checkpoint / metrics features (idempotent: a re-extraction replaces the row)."""
        with self._locked():
            self._db.execute("INSERT OR REPLACE INTO diagnostics VALUES (?,?,?,?,?,?,?,?)",
                             (run_id, self.task_id, node_id, int(seed), getattr(stage, "value", stage),
                              self._clock() if ts is None else ts, int(version),
                              json.dumps(dict(features), sort_keys=True)))

    def diagnostics(self, node_id: str | None = None) -> list[dict[str, Any]]:
        """Diagnosed runs of this task (a node's, or all): {run_id, node_id, seed, stage, ts, version, features}."""
        sql, args = "SELECT * FROM diagnostics WHERE task_id=?", [self.task_id]
        if node_id is not None:
            sql, args = sql + " AND node_id=?", [*args, node_id]
        rows = self._read(sql + " ORDER BY ts, run_id", args)
        return [{**{k: r[k] for k in ("run_id", "node_id", "seed", "stage", "ts", "version")},
                 "features": json.loads(r["features"] or "{}")} for r in rows]

    def diagnosed_ids(self) -> set[str]:
        return {r[0] for r in self._read("SELECT run_id FROM diagnostics WHERE task_id=?", (self.task_id,))}

    # -- seed blocks (loop.core.seeds) -------------------------------------------
    def put_seed_block(self, block: SeedBlock) -> None:
        with self._locked():
            self._db.execute("INSERT INTO seed_blocks (id, task_id, pool, seeds, node_ids, purpose, drawn_ts, "
                             "closed_ts, verdicts) VALUES (?,?,?,?,?,?,?,?,?)",
                             (block.id, self.task_id, block.pool, json.dumps(list(block.seeds)),
                              json.dumps(list(block.node_ids)), block.purpose, block.drawn_ts, block.closed_ts,
                              None if block.verdicts is None else json.dumps([dict(v) for v in block.verdicts])))

    def update_seed_block(self, block_id: str, **fields: Any) -> SeedBlock:
        """Extend (``seeds``) or close (``closed_ts``) a block, or freeze its ``verdicts`` (written once)."""
        cur = next((b for b in self.seed_blocks() if b.id == block_id), None)
        if cur is None:
            raise KeyError(f"seed_blocks: no block {block_id!r} for task {self.task_id!r}")
        new = dataclasses.replace(cur, **fields)
        with self._locked():
            self._db.execute("UPDATE seed_blocks SET seeds=?, node_ids=?, closed_ts=?, verdicts=? "
                             "WHERE id=? AND task_id=?",
                             (json.dumps(list(new.seeds)), json.dumps(list(new.node_ids)), new.closed_ts,
                              None if new.verdicts is None else json.dumps([dict(v) for v in new.verdicts]),
                              block_id, self.task_id))
        return new

    def seed_blocks(self) -> list[SeedBlock]:
        rows = self._read("SELECT * FROM seed_blocks WHERE task_id=? ORDER BY drawn_ts, id", (self.task_id,))
        return [SeedBlock(r["id"], r["pool"], tuple(json.loads(r["seeds"])), tuple(json.loads(r["node_ids"])),
                          r["purpose"], r["drawn_ts"], r["closed_ts"],
                          None if r["verdicts"] is None else tuple(json.loads(r["verdicts"]))) for r in rows]

    # -- config versions (loop.core.meta) ---------------------------------------
    def put_config_version(self, kind: str, v: int, *, parent: int | None, ts: float, diff: Mapping[str, Any],
                           reason: str, backtest: Mapping[str, Any] | None, state: str) -> None:
        """A new version (``kind`` config | prompt); an existing (kind, v) raises sqlite3.IntegrityError."""
        if kind not in CONFIG_VERSION_KINDS:
            raise ValueError(f"config_versions: kind {kind!r} not in {CONFIG_VERSION_KINDS}")
        with self._locked():
            self._db.execute("INSERT INTO config_versions VALUES (?,?,?,?,?,?,?,?,?)",
                             (self.task_id, kind, int(v), parent, ts, json.dumps(encode(diff), sort_keys=True),
                              reason, json.dumps(encode(backtest or {}), sort_keys=True, default=str), state))

    def set_config_version_state(self, kind: str, v: int, state: str) -> None:
        with self._locked():
            n = self._db.execute("UPDATE config_versions SET state=? WHERE task_id=? AND kind=? AND v=?",
                                 (state, self.task_id, kind, int(v))).rowcount
        if not n:
            raise KeyError(f"config_versions: no {kind} v{v} for task {self.task_id!r}")

    def config_versions(self, kind: str | None = None) -> list[dict[str, Any]]:
        """{kind, v, parent, ts, diff, reason, backtest, state} in version order."""
        rows = self._read("SELECT * FROM config_versions WHERE task_id=? AND (? IS NULL OR kind=?) "
                          "ORDER BY kind, v", (self.task_id, kind, kind))
        return [{"kind": r["kind"], "v": r["v"], "parent": r["parent"], "ts": r["ts"],
                 "diff": json.loads(r["diff"] or "{}"), "reason": r["reason"],
                 "backtest": json.loads(r["backtest"] or "{}"), "state": r["state"]} for r in rows]

    # -- state and log ----------------------------------------------------------
    def get_state(self, key: str, default: Any = None) -> Any:
        rows = self._read("SELECT value FROM state WHERE task_id=? AND key=?", (self.task_id, key))
        row = rows[0] if rows else None
        return default if row is None else json.loads(row["value"])

    def set_state(self, key: str, value: Any) -> None:
        with self._locked():
            self._db.execute("INSERT OR REPLACE INTO state VALUES (?,?,?)",
                             (self.task_id, key, json.dumps(encode(value), sort_keys=True)))

    def update_state(self, key: str, fn: Any, default: Any = None) -> Any:
        """Read-modify-write of one state key under the write lock (atomic against every other writer of this
        store): ``fn(current) -> (new value | None, result)``; the new value is written unless None; returns
        ``result``."""
        with self._locked():
            row = self._db.execute("SELECT value FROM state WHERE task_id=? AND key=?", (self.task_id, key)).fetchone()
            new, result = fn(default if row is None else json.loads(row[0]))
            if new is not None:
                self._db.execute("INSERT OR REPLACE INTO state VALUES (?,?,?)",
                                 (self.task_id, key, json.dumps(encode(new), sort_keys=True)))
        return result

    # -- the competition board (loop.board; created on first use, so a task without a board has no table) --------
    _BOARD_DDL = ("CREATE TABLE IF NOT EXISTS board_runs (task_id TEXT NOT NULL, run_id TEXT NOT NULL, updated TEXT, "
                  "fetched_ts REAL, row TEXT, source TEXT, PRIMARY KEY (task_id, run_id))")

    def _board_table(self) -> bool:
        if not getattr(self, "_board_ok", False):
            with self._locked():
                self._db.execute(self._BOARD_DDL)
            self._board_ok = True
        return True

    def put_board_runs(self, rows: Iterable[Mapping[str, Any]], ts: float | None = None) -> int:
        """Upsert board runs ({run_id, updated_at, ...}); a row's stored source is kept. Returns rows written."""
        self._board_table()
        ts, n = self._clock() if ts is None else ts, 0
        with self._locked():
            for r in rows:
                rid = str(r.get("run_id") or "")
                if not rid:
                    continue
                self._db.execute(
                    "INSERT INTO board_runs (task_id, run_id, updated, fetched_ts, row, source) VALUES (?,?,?,?,?,NULL) "
                    "ON CONFLICT(task_id, run_id) DO UPDATE SET updated=excluded.updated, "
                    "fetched_ts=excluded.fetched_ts, row=excluded.row",
                    (self.task_id, rid, str(r.get("updated_at") or ""), ts, json.dumps(r, sort_keys=True, default=str)))
                n += 1
        return n

    def put_board_source(self, run_id: str, source: Mapping[str, Any]) -> None:
        self._board_table()
        with self._locked():
            self._db.execute("UPDATE board_runs SET source=? WHERE task_id=? AND run_id=?",
                             (json.dumps(source, sort_keys=True, default=str), self.task_id, run_id))

    def board_runs(self) -> list[dict[str, Any]]:
        """Every stored board run: its row plus ``_source`` (the stored source digest, or None)."""
        self._board_table()
        rows = self._read("SELECT row, source FROM board_runs WHERE task_id=? ORDER BY run_id", (self.task_id,))
        return [{**json.loads(r["row"]), "_source": json.loads(r["source"]) if r["source"] else None} for r in rows]

    # -- the operator's manual runs (loop.localruns; EXTERNAL evidence, never nodes / runs; created on first use) --
    _LOCAL_DDL = ("CREATE TABLE IF NOT EXISTS local_runs (task_id TEXT NOT NULL, name TEXT NOT NULL, fp TEXT, "
                  "ts REAL, row TEXT, PRIMARY KEY (task_id, name))")

    def _local_table(self) -> None:
        if not getattr(self, "_local_ok", False):
            with self._locked():
                self._db.execute(self._LOCAL_DDL)
            self._local_ok = True

    def put_local_run(self, name: str, fp: str, row: Mapping[str, Any], ts: float | None = None) -> None:
        """Upsert one imported manual run (``fp``: the fingerprint of its files, the import's change test)."""
        self._local_table()
        with self._locked():
            self._db.execute(
                "INSERT INTO local_runs (task_id, name, fp, ts, row) VALUES (?,?,?,?,?) ON CONFLICT(task_id, name) "
                "DO UPDATE SET fp=excluded.fp, ts=excluded.ts, row=excluded.row",
                (self.task_id, name, fp, self._clock() if ts is None else ts,
                 json.dumps(dict(row), sort_keys=True, default=str)))

    def local_runs(self) -> list[dict[str, Any]]:
        """Every imported manual run: its row plus ``_fp`` / ``_ts``."""
        self._local_table()
        rows = self._read("SELECT name, fp, ts, row FROM local_runs WHERE task_id=? ORDER BY name", (self.task_id,))
        return [{**json.loads(r["row"]), "name": r["name"], "_fp": r["fp"], "_ts": r["ts"]} for r in rows]

    def log(self, event: str, **fields: Any) -> None:
        with self._locked():
            self._db.execute("INSERT INTO log (task_id, ts, event, fields) VALUES (?,?,?,?)",
                             (self.task_id, self._clock(), event,
                              json.dumps(encode(fields), sort_keys=True, default=str)))

    def events(self, event: str | None = None) -> list[dict[str, Any]]:
        rows = self._read("SELECT ts, event, fields FROM log WHERE task_id=? AND (? IS NULL OR event=?) ORDER BY id",
                          (self.task_id, event, event))
        return [{"ts": r["ts"], "event": r["event"], **json.loads(r["fields"] or "{}")} for r in rows]

    def table_dump(self, *, max_rows: int = 500) -> str:
        """The raw ledger for prompts: one line per node with its mean per-axis and panel
        paired delta vs the same-seed root run."""
        def by_seed(node_id: str) -> dict[int, C.Run]:
            return {r.seed: r for r in reversed(self.runs(node_id=node_id,
                                                          status=[C.RunStatus.DONE]))}
        root, nodes = by_seed(C.ROOT_NODE_ID), self.nodes()
        axes = sorted({a for r in root.values() for a in r.scores})
        lines = [f"# task {self.task_id} | axes: {' '.join(axes)}",
                 "# id | parent | island | family | tag | levers | status | n | d(axes...) | panel"]
        for node in nodes[:max_rows]:
            mine = by_seed(node.id)
            pairs = [(r, root[s]) for s, r in mine.items() if s in root and node.id != C.ROOT_NODE_ID]
            deltas = [paired_delta(r.scores, b.scores, self._weights) for r, b in pairs]
            panel = [r.aggregate - b.aggregate if None not in (r.aggregate, b.aggregate)
                     else d.aggregate for (r, b), d in zip(pairs, deltas) if d.deltas]

            def mean(xs: list[float]) -> str:
                return f"{sum(xs) / len(xs):+.4f}" if xs else "."
            levers = ",".join(f"{'+' if d > 0 else '-'}{k}"
                              for k, d in sorted(node.levers.directions.items()) if d) or "-"
            lines.append(" | ".join([
                node.id, node.parent_id or "-", node.island, node.family, node.mechanism_tag,
                levers, node.status.value, str(len(mine)),
                " ".join(mean([d.deltas[a] for d in deltas if a in d.deltas]) for a in axes) or ".",
                mean(panel)]))
        if len(nodes) > max_rows:
            lines.append(f"# ... {len(nodes) - max_rows} more nodes omitted")
        return "\n".join(lines)
