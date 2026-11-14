"""loop.prune — CHECKPOINT PRUNING. Every run leaves a ~1.35 GB ``latest.pt`` (model + EMA + fitted readout
buffers) in its output dir; nothing else needs one except the eval-only lane (loop.evalonly), which resumes
MODELS' checkpoints. A rate-limited pass (a background thread of the driver: :class:`Pruner`) deletes checkpoint files only (``PATTERNS``:
never summary.json, metrics or logs), of finished (done / void / crash) runs only:

* eval-only runs' checkpoints (resume shims): always -- they are reproducible;
* runs of a node that is live (queued / screening / promoting / confirming / evaluating) or closed (its last
  ``node_transition``) less than ``grace_h`` ago: never;
* MODELS -- root, the epoch bases, the incumbent, CONFIRMED nodes, nodes in / queued for the held-out audit (or
  passed it), banked winner-cross parents, and parents of live nodes: ALL their checkpoints stay;
* v3: a derived SCREENING node (it stays SCREENING for life; the store status is never updated) SETTLES -- its
  checkpoints go like a closed node's -- once no open block / wave / port / queue entry, no RUNNING / SUBMITTING
  run and nothing submitted within ``grace_h`` holds it; eval-only runs' checkpoints go as above;
* a run whose output has a pending diagnostics extraction, and every referenced eval-only source, stay;
* every other closed node (screened out, rejected, vetoed, crashed, void, not confirmed, audit failed,
  eval passed) past the grace period: all of its checkpoints.

``dry_run`` logs what would go (``prune`` event with bytes) and deletes nothing."""

from __future__ import annotations

import errno
import json
import re
import threading
from pathlib import Path
from typing import Any, Callable

from loop import diagnostics as DG
from loop.contracts import ACTIVE_STATUSES, K_INFLIGHT, ROOT_NODE_ID, NodeStatus, Run, RunStatus
from loop.evalonly import K_EO, STAGE_ORDER, is_eval_run
from loop.labless import K_AUDIT
from loop.spec import LoopContext

K_PRUNE = "prune"                                  # {"ts": last pass}
PATTERNS = ("latest.pt", "*.pt.tmp", "step_*.pt", "resume_shim.pt")
LIVE = frozenset({NodeStatus.QUEUED, NodeStatus.EVALUATING, *ACTIVE_STATUSES})
_ATTEMPT = re.compile(r"_a(\d+)$")


def run_dirs(run: Run) -> list[Path]:
    """Every attempt's output dir of ``run`` (``<stage>_s<seed>_a<k>``, k = 0..attempt)."""
    if not run.output_dir:
        return []
    out = Path(run.output_dir)
    if not (m := _ATTEMPT.search(out.name)):
        return [out]
    return [out.with_name(f"{out.name[:m.start()]}_a{k}") for k in range(max(run.attempt, int(m.group(1))) + 1)]


def checkpoints(run: Run) -> list[Path]:
    return [p for d in run_dirs(run) if d.is_dir() for pat in PATTERNS for p in sorted(d.glob(pat)) if p.is_file()]


def eval_refs(ctx: LoopContext, store: Any) -> set[str]:
    """Run ids an eval-only plan names as its source (lane state + every plan file ever launched)."""
    eo = store.get_state(K_EO) or {}
    refs = {r["source"] for e in (eo.get("nodes") or {}).values() for r in (e.get("runs") or {}).values()}
    refs |= set(((eo.get("null") or {}).get("runs") or {}).values())
    for f in (Path(ctx.state_dir) / "evalonly").glob("*/*.json") if ctx.state_dir else ():
        try:                                         # every plan ever launched (survives lost lane state)
            refs.add(json.loads(f.read_text()).get("source_run") or "")
        except (OSError, ValueError, AttributeError):
            continue
    return refs


def classify(ctx: LoopContext, store: Any, snapshot: Any, now: float) -> tuple[set[str], set[str], set[str]]:
    """(protected node ids, model node ids, referenced source run ids)."""
    grace = 3600.0 * ctx.cfg.prune.grace_h
    nodes = store.nodes()
    closed = {e["node_id"]: e["ts"] for e in store.events("node_transition")}   # the last transition = the close
    # v3 (loop.tick): statuses are derived, never stored -- a node stays QUEUED in the store for life. Its snapshot
    # names the derived-closed nodes (``closed``: not live whatever the store says) and what open nodes / blocks /
    # waves reference (``keep``: always protected)
    done = set(getattr(snapshot, "closed", ()) or ())
    # v3 (10-19 root cause): a derived SCREENING node stays SCREENING for life (it is never FAILED / VOID / CRASHED
    # unless its test closes it), so "derived-open" kept 2357 nodes / 1.9 TB forever. A screening node SETTLES -- its
    # checkpoints are reproducible (a test retrains from the commit) -- once nothing holds it: no open block / wave /
    # port / queue entry (``snapshot.keep`` excludes screening nodes that are not referenced), no RUNNING / SUBMITTING
    # run, nothing submitted within ``grace_h``
    screening, live = set(getattr(snapshot, "screening", ()) or ()), set()
    hold = set(getattr(snapshot, "keep", ()) or ())
    if screening:
        busy = {r.node_id for r in store.runs() if r.status in (RunStatus.RUNNING, RunStatus.SUBMITTING)
                or now - float(r.submitted_ts or 0.0) < grace}
        by_id = {n.id: n for n in nodes}
        settled = {n for n in screening - hold - busy if n in by_id and now - float(by_id[n].updated_ts or 0.0) >= grace
                   and now - closed.get(n, 0.0) >= grace}
        live = screening - settled
        done |= settled
        hold |= live | {by_id[n].parent_id for n in live if n in by_id and by_id[n].parent_id} \
            | {by_id[n].second_parent_id for n in live if n in by_id and by_id[n].second_parent_id}
    protected = {n.id for n in nodes if n.status in LIVE and n.id not in done
                 or (n.status is not NodeStatus.BASELINE and now - closed.get(n.id, n.updated_ts or 0.0) < grace)}
    protected |= hold
    audit = store.get_state(K_AUDIT) or {}
    # never pruned at all: root, the epoch bases (BASELINE), CONFIRMED / audit nodes, the incumbent, banked parents
    hard = {ROOT_NODE_ID, *(n.id for n in nodes if n.status in (NodeStatus.CONFIRMED, NodeStatus.BASELINE)),
            *audit.get("passed", []), *audit.get("queue", []), (audit.get("current") or {}).get("node", "")}
    if snapshot is not None:
        hard |= {snapshot.incumbent_id, *getattr(snapshot, "banked", ())}
    # parents of live nodes (store-live and not derived-closed, or a live screening node): never pruned either
    parents = {p for n in nodes if (n.status in LIVE and n.id not in done or n.id in live)
               for p in (n.parent_id, n.second_parent_id) if p}
    protected |= hard | parents
    models = hard | parents
    refs = eval_refs(ctx, store)
    return protected, models - {""}, refs - {""}


def plan(ctx: LoopContext, store: Any, snapshot: Any, now: float) -> tuple[list[tuple[Path, int]], int]:
    """(checkpoint files to delete with their sizes, checkpoint files kept)."""
    protected, models, refs = classify(ctx, store, snapshot, now)
    reserve: dict[str, set[str]] = {}
    for nid in models:
        done = sorted((r for r in store.runs(node_id=nid, status=[RunStatus.DONE]) if not is_eval_run(r)
                       and (Path(r.output_dir) / "latest.pt").is_file()), key=lambda r: (STAGE_ORDER.get(r.stage, 9), r.seed))
        reserve[nid] = {r.id for r in done[:ctx.cfg.prune.reserve_per_model]}
    drop, kept = [], 0
    for run in store.runs(status=[RunStatus.DONE, RunStatus.VOID, RunStatus.CRASH]):
        files = checkpoints(run)
        if not files:
            continue
        keep = not is_eval_run(run) and (run.node_id in protected or run.id in refs
                                         or run.id in reserve.get(run.node_id, ())
                                         or (run.status is RunStatus.DONE and DG.pending(ctx, run, now)))
        if keep:
            kept += len(files)
            continue
        drop += [(p, p.stat().st_size) for p in files]
    return drop, kept


def smoke_plan(ctx: LoopContext, now: float) -> list[tuple[Path, int]]:
    """Smoke checkpoints (``<state>/smoke/<wt>/out/**/latest.pt``, left by killed or failed smoke jobs) older than
    ``prune.smoke_max_age_h`` (default 6), with their sizes."""
    if not ctx.state_dir:
        return []
    age = 3600.0 * float(getattr(ctx.cfg.prune, "smoke_max_age_h", 6.0))
    out = []
    for p in sorted((Path(ctx.state_dir) / "smoke").glob("*/out/**/latest.pt")):
        try:
            st = p.stat()
        except OSError:
            continue
        if p.is_file() and now - st.st_mtime >= age:
            out.append((p, st.st_size))
    return out


def _newest_mtime(d: Path) -> float:
    """The newest mtime of ``d`` and its direct children (a smoke writes its out/ files there)."""
    ts = [d.stat().st_mtime]
    for c in d.iterdir():
        try:
            ts.append(c.stat().st_mtime)
        except OSError:
            continue
    return max(ts)


def _size(d: Path) -> int:
    n = 0
    for root, _, files in __import__("os").walk(d):
        for f in files:
            try:
                n += (Path(root) / f).lstat().st_size
            except OSError:
                continue
    return n


def smoke_dir_plan(ctx: LoopContext, store: Any, now: float) -> list[tuple[Path, int]]:
    """v3-c8a B6 (10-14: 232 ``loop-req-v3-*`` smoke dirs, ~14 MB each, 3.2 GB, mostly ``out/labless_source``
    copies; :func:`smoke_plan` deleted only their latest.pt): WHOLE smoke dirs (``<state>/smoke/loop-<request>[-<step>]``)
    of research requests no longer in flight (``K_INFLIGHT``, queued re-issues included) whose newest entry is older
    than ``prune.smoke_dir_max_age_h`` (default 24), with their sizes. Other smoke dirs are left alone."""
    if not ctx.state_dir:
        return []
    root = Path(ctx.state_dir) / "smoke"
    if not root.is_dir():
        return []
    age = 3600.0 * float(getattr(ctx.cfg.prune, "smoke_dir_max_age_h", 24.0))
    live = {str(e.get("id") or "") for e in store.get_state(K_INFLIGHT) or []} - {""}
    out = []
    for d in sorted(root.iterdir()):
        name = d.name
        if not name.startswith("loop-") or not d.is_dir() or d.is_symlink():
            continue
        rid = name[len("loop-"):]
        if any(rid == r or rid.startswith(r + "-") for r in live):
            continue                                  # an in-flight request's smoke: never
        try:
            if now - _newest_mtime(d) < age:
                continue
        except OSError:
            continue
        out.append((d, _size(d)))
    return out


TMP_MAX_AGE_S = 3600.0


def tmp_plan(ctx: LoopContext, now: float) -> list[Path]:
    """v3 review N1: the git-plumbing scratch files (``<state>/tmp/idx-*``, ``blob-*``: loop.adapters commit_files) a killed
    build left behind, older than an hour."""
    d = Path(getattr(ctx, "state_dir", "") or "") / "tmp"
    if not getattr(ctx, "state_dir", "") or not d.is_dir():
        return []
    out = []
    for f in d.iterdir():
        try:
            if f.is_file() and f.name.startswith(("idx-", "blob-")) and now - f.stat().st_mtime > TMP_MAX_AGE_S:
                out.append(f)
        except OSError:
            continue
    return out


def worktree_nodes(ctx: LoopContext, store: Any, snapshot: Any, now: float) -> list[str]:
    """v3-c8a B6 (10-14: 411 node worktrees under the adapter's root, ``prune git: 0 node worktrees`` every pass): the
    v3 view keeps every non-closed node's CHECKPOINTS (a SCREENING node stays SCREENING for life), but a node worktree
    is only a checkout its next launch re-creates from the tagged commit. One goes when its node has no active run,
    is not QUEUED / in an open block / wave / port (``snapshot.queued`` / ``snapshot.open``), is neither root nor the
    incumbent, and launched nothing (nor was created) in the last ``prune.grace_h``."""
    grace = 3600.0 * ctx.cfg.prune.grace_h
    hold = set(getattr(snapshot, "queued", ()) or ()) | set(getattr(snapshot, "open", ()) or ()) | {
        ROOT_NODE_ID, getattr(snapshot, "incumbent_id", "") or ""}
    last: dict[str, float] = {}
    active: set[str] = set()
    for r in store.runs():
        last[r.node_id] = max(last.get(r.node_id, 0.0), float(r.submitted_ts or 0.0))
        if r.status in (RunStatus.RUNNING, RunStatus.SUBMITTING):
            active.add(r.node_id)
    return sorted(n.id for n in store.nodes() if n.id not in hold and n.id not in active
                  and now - max(last.get(n.id, 0.0), float(n.created_ts or 0.0)) >= grace)


def git_plan(ctx: LoopContext, store: Any, snapshot: Any, now: float) -> tuple[dict[str, str], list[str], list[str]]:
    """(node id -> commit, every node: kept reachable; research requests whose branch may go -- not in flight,
    started more than ``grace_h`` ago, their node (if any) neither protected nor a model; node ids whose
    worktree may go -- neither protected nor a model, no active run)."""
    protected, models, _ = classify(ctx, store, snapshot, now)
    keep, nodes = protected | models, store.nodes()
    inflight = {e.get("id") for e in store.get_state(K_INFLIGHT) or []}
    node_of = {m.group(1): n.id for n in nodes if (m := re.search(r"\brequest=(\S+)", n.notes or ""))}
    grace = 3600.0 * ctx.cfg.prune.grace_h
    requests = sorted({e["request_id"] for e in store.events("research_started")
                       if e.get("request_id") and e["request_id"] not in inflight and now - e["ts"] >= grace
                       and node_of.get(e["request_id"]) not in keep})
    active = {r.node_id for r in store.runs(status=[RunStatus.RUNNING, RunStatus.SUBMITTING])}
    wts = worktree_nodes(ctx, store, snapshot, now) if hasattr(snapshot, "queued") else \
        sorted(n.id for n in nodes if n.id not in keep and n.id not in active and n.id != ROOT_NODE_ID)
    return {n.id: n.commit for n in nodes if n.commit}, requests, wts


def stray_keep(ctx: LoopContext, store: Any, snapshot: Any, now: float) -> set[str]:
    """Node ids whose STRAY worktrees (eval-only ``<node>_<c10>``, ``w###.x``; adapter.tidy ``keep``) stay: the
    incumbent and every node with a RUNNING / SUBMITTING run. A protected node's eval worktrees are NOT kept
    (c5: they held 97 of them, ~1.5 GB, and are rebuilt from the commit on demand; the node's own worktree is
    governed by :func:`git_plan`, not by this)."""
    inc = getattr(snapshot, "incumbent_id", "") if snapshot is not None else ""
    return {inc} - {""} | {r.node_id for r in store.runs(status=[RunStatus.RUNNING, RunStatus.SUBMITTING])}


class Pruner:
    """The rate-limited pass. ``background`` (the driver): the pass -- plan, unlinks, git tidy (minutes: heartbeats
    were 11-14 min apart while it ran in the tick) -- runs in a daemon thread; the next tick returns its lines.
    The adapter's worktree lock keeps tidy from racing a researcher's worktree add; :meth:`stop` ends a pass
    between two git commands (never inside one: a killed ``branch -D`` would leave a ref lock)."""

    def __init__(self, ctx: LoopContext, store: Any, clock: Callable[[], float], adapter: Any = None,
                 background: bool = False) -> None:
        self.ctx, self.store, self.clock, self.adapter, self.background = ctx, store, clock, adapter, background
        self._thread: threading.Thread | None = None
        self._lines: list[str] = []
        self._halt = False

    def tick(self, snapshot: Any) -> list[str]:
        cfg, now = self.ctx.cfg.prune, self.clock()
        lines, self._lines = self._lines, []                # a finished background pass reports here
        if self._thread is not None and self._thread.is_alive():
            return lines
        if not cfg.enabled or now - float((self.store.get_state(K_PRUNE) or {}).get("ts", 0.0)) < 60.0 * cfg.every_min:
            return lines
        self.store.set_state(K_PRUNE, {"ts": now})
        if not self.background:
            return lines + self._pass(snapshot, now)
        self._thread = threading.Thread(target=self._bg, args=(snapshot, now), name="loop-prune", daemon=True)
        self._thread.start()
        return lines

    def _bg(self, snapshot: Any, now: float) -> None:
        try:
            self._lines = self._pass(snapshot, now)
        except Exception as exc:  # noqa: BLE001 - housekeeping never stops the loop
            self.store.log("prune_error", error=repr(exc)[:500])

    def stop(self, timeout: float = 60.0) -> None:
        """Ask a running background pass to end after its current git command and wait <= ``timeout``."""
        self._halt = True
        if self._thread is not None:
            self._thread.join(timeout)

    def _pass(self, snapshot: Any, now: float) -> list[str]:
        cfg = self.ctx.cfg.prune
        try:
            drop, kept = plan(self.ctx, self.store, snapshot, now)
            drop += smoke_plan(self.ctx, now)
            dirs = smoke_dir_plan(self.ctx, self.store, now)
        except Exception as exc:  # noqa: BLE001 - a failed pass is logged and retried next period
            self.store.log("prune_error", error=repr(exc)[:500])
            return [f"prune failed: {repr(exc)[:300]}"]
        gone = sum(s for _, s in dirs)
        if dirs and not cfg.dry_run:
            import shutil
            for d, _ in dirs:
                if self._halt:
                    break
                shutil.rmtree(d, ignore_errors=True)
            gone = sum(s for d, s in dirs if not d.exists())
        stale = tmp_plan(self.ctx, now)
        if stale and not cfg.dry_run:
            for f in stale:
                try:
                    f.unlink()
                except OSError:
                    pass
        smoke_line = ([f"prune tmp{' (dry run)' if cfg.dry_run else ''}: {len(stale)} stale idx-/blob- scratch file(s)"]
                      if stale else []) + ([f"prune smoke{' (dry run)' if cfg.dry_run else ''}: {len(dirs)} finished request smoke dir(s), "
                       f"{sum(s for _, s in dirs) / 1e9:.2f} GB" + ("" if cfg.dry_run else f" ({gone / 1e9:.2f} GB freed)")]
                      if dirs else [])
        if dirs:
            self.store.log("prune_smoke", dry_run=cfg.dry_run, dirs=len(dirs), bytes=sum(s for _, s in dirs))
        if drop and not cfg.dry_run:                # work launched since the plan (an eval-only plan resuming one of
            try:                                    # these checkpoints) is re-read once more right before the unlinks
                held = {d for rid in eval_refs(self.ctx, self.store) if (r := self.store.get_run(rid)) is not None
                        for d in run_dirs(r)}
                drop = [(p, s) for p, s in drop if p.parent not in held]
            except Exception as exc:  # noqa: BLE001 - a failed re-check skips the deletions this pass
                self.store.log("prune_error", error=repr(exc)[:500])
                drop = []
        freed, failed = 0, 0
        meta = self._audit_meta(snapshot) if drop and not cfg.dry_run else {}
        for path, size in drop if not cfg.dry_run else ():
            if self._halt:
                break
            try:
                self._audit(path, size, meta.get(path), now)    # the audit line lands BEFORE the unlink; no line, no delete
                path.unlink()
                freed += size
            except OSError:
                failed += 1
        total = sum(s for _, s in drop)
        self.store.log("prune", dry_run=cfg.dry_run, files=len(drop), bytes=total, freed=freed, failed=failed, kept=kept)
        return [f"prune{' (dry run)' if cfg.dry_run else ''}: {len(drop)} checkpoints, {total / 1e9:.1f} GB "
                + (f"would be freed; {kept} kept" if cfg.dry_run else f"freed {freed / 1e9:.1f} GB ({failed} failed); {kept} kept"),
                *smoke_line, *self._git(snapshot, now)]

    def _audit_meta(self, snapshot: Any) -> dict[Path, tuple[str, str, str]]:
        """{checkpoint file: (node_id, run_id, reason)} for the audit trail (files in no run: smoke checkpoints)."""
        out: dict[Path, tuple[str, str, str]] = {}
        closed, screening = getattr(snapshot, "closed", ()), getattr(snapshot, "screening", ())
        try:
            runs = list(self.store.runs(status=[RunStatus.DONE, RunStatus.VOID, RunStatus.CRASH]))
        except Exception as exc:  # noqa: BLE001 - the reason/ids are best effort; the path/bytes line is the trail
            self.store.log("prune_error", error=repr(exc)[:300])
            return out
        for run in runs:
            why = ("eval-only run" if is_eval_run(run) else "derived-closed" if run.node_id in closed else
                   "settled screening" if run.node_id in screening else "closed (store status)")
            for f in checkpoints(run):
                out[f] = (run.node_id, run.id, why)
        return out

    def _audit(self, path: Path, size: int, meta: tuple[str, str, str] | None, now: float) -> None:
        """Append one JSONL line (ts, path, bytes, node_id, run_id, reason) to ``<state_dir>/prune_deleted.jsonl``;
        raises OSError when it cannot (the caller then skips this file's deletion)."""
        if not self.ctx.state_dir:
            raise OSError("no state_dir for the prune audit trail")
        node_id, run_id, why = meta or ("", "", "smoke checkpoint")
        line = json.dumps({"ts": now, "path": str(path), "bytes": size, "node_id": node_id, "run_id": run_id,
                           "reason": why})
        try:
            Path(self.ctx.state_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(self.ctx.state_dir) / "prune_deleted.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        except OSError as exc:
            if exc.errno not in (errno.ENOSPC, errno.EDQUOT):
                raise
            # a FULL disk is the one case where the pass must still free space: the delete is logged to the store instead
            self.store.log("prune_audit_unwritable", path=str(path), bytes=size, node_id=node_id, run_id=run_id,
                           reason=why, error=repr(exc)[:200])

    def _git(self, snapshot: Any, now: float) -> list[str]:
        """Research branches / node worktrees of long-closed work (adapter.tidy); node commits stay tagged."""
        if not self.ctx.cfg.prune.git or not hasattr(self.adapter, "tidy"):
            return []
        try:
            n = self.adapter.tidy(*git_plan(self.ctx, self.store, snapshot, now), dry_run=self.ctx.cfg.prune.dry_run,
                                  halt=lambda: self._halt, keep=stray_keep(self.ctx, self.store, snapshot, now))
        except Exception as exc:  # noqa: BLE001 - housekeeping never stops the loop
            self.store.log("prune_git_error", error=repr(exc)[:500])
            return [f"prune git failed: {repr(exc)[:300]}"]
        self.store.log("prune_git", dry_run=self.ctx.cfg.prune.dry_run, **n)
        return [f"prune git{' (dry run)' if self.ctx.cfg.prune.dry_run else ''}: {n['branches']} research branches, "
                f"{n['worktrees']} node worktrees, {n.get('stray', 0)} stray worktrees ({n.get('stray_removed', 0)} removed, "
                f"{n.get('stray_kept', 0)} kept); {n['tagged']} node commits tagged"]
