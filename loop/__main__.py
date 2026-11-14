"""``uv run python -m loop run|calibrate|status|review|ack|dashboard --task tasks/X.yaml``; ``serve --state-dir D``;
``backtest --db <ledger copy>`` (loop.core.backtest; DESIGN §6); ``shadow --task T --db <copy>
[--refresh-from <live db>]`` (the v3 tick on a ledger copy, nothing submitted: loop.tick); ``meta --task T [--db D]
[--review] [--fake-llm]`` (one v3 meta-cycle now: loop.core.meta) and ``revert v<N> --task T [--db D] [--prompt]``.
``run --task T [--state-dir D] [--max-ticks N] [--fake-llm]`` is the v3 tick (loop.tick.live, cutover step 12) on
``<state-dir>/<task>.db``; ``run-v2`` (same flags as the old ``run``) is the old driver until its deletion.

``run-v2`` on UNMEASURED noise calibrates first (resuming / re-attaching to its runs), writes the sigmas,
then continues into the loop in the same process; ``review`` refuses (exit 2). ``--dry-run`` plans and
submits nothing. ``run`` / ``calibrate`` (live) and ``review`` hold an exclusive lock on ``<db>.lock``
(exit 3 when taken). ``status`` is a read-only summary; ``dashboard`` renders the HTML page once (read-only,
``--out``) and ``serve`` serves ``<state-dir>/dashboard`` on 127.0.0.1 (``--port``). ``--fake-llm`` wires an offline FakeLLMClient.
SIGTERM stops between ticks (exit 0), leaving SLURM jobs running.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import importlib
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from loop.contracts import ACTIVE_STATUSES as ACTIVE, PANEL, ROOT_NODE_ID, UnmeasuredLoopNoise



class _Stop:
    """Set by SIGTERM / SIGINT. Lock-free on purpose: a Python handler runs between bytecodes of the
    interrupted frame, and ``threading.Event.set`` takes a lock that frame (or a nested handler: uv
    and systemd both deliver SIGTERM) may hold -- a self-deadlock that systemd can only force-kill."""
    flag = False

    def set(self) -> None:
        self.flag = True

    def is_set(self) -> bool:
        return self.flag


STOP = _Stop()                                      # see __main__ below; waits are ~1 s slices


def resolve_adapter(ctx: Any) -> Any:
    mod, _, cls = ctx.cfg.execution_adapter.partition(":")
    return getattr(importlib.import_module(mod), cls)(ctx)


def _fake_llm() -> Any:
    """Every role answers with a minimal valid payload; the researcher never finishes."""
    from loop.agents.llm import AgentRun, FakeLLMClient

    def json_role(**call: Any) -> Any:
        req = set(call["schema"].get("required", ()))
        if "assumptions" in req:
            return {"summary": "fake-llm first-principles review", "assumptions": []}
        if "root_assumptions" in req:
            return {"summary": "fake-llm root audit", "root_assumptions": [
                {"choice": "weight_decay", "where": "config", "value": "0.05", "verdict": "untested", "evidence": "none",
                 "arm": {"kind": "sweep_up", "config_diff": {"weight_decay": 0.1}, "mechanism": "fake", "priority": 0.5}}]}
        if "meta_hypotheses" in req:
            return {"summary": "fake-llm review", "findings": [], "meta_hypotheses": []}
        return {}
    return FakeLLMClient({"researcher": lambda **_: AgentRun(False, None, 0, 0.0, error="fake-llm"),
                          "scout": lambda **_: AgentRun(False, None, 0, 0.0, error="fake-llm"),
                          "review": json_role})


def _db_path(ctx: Any) -> Path:
    return Path(ctx.state_dir) / f"{ctx.task_id}.db"


def _db(ctx: Any) -> Path:
    Path(ctx.state_dir).mkdir(parents=True, exist_ok=True)
    return _db_path(ctx)


def _lock(ctx: Any) -> Any:
    """Exclusive, non-blocking lock on ``<db>.lock``; None if another instance holds it."""
    fh = open(f"{_db(ctx)}.lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        print(f"{_utc()} refused: another loop instance holds {_db(ctx)}.lock (pid {os.getpid()}, ppid {os.getppid()}, "
              f"argv {' '.join(sys.argv[1:])[:160]!r})", file=sys.stderr)    # 10-22: who is asking, in the shared log
        return None
    return fh


def build(ctx: Any, *, fake_llm: bool = False, llm: Any = None, adapter: Any = None,
          executor: Any = None, rng: Any = None) -> tuple[Any, Any]:
    """Wire the production object graph; returns ``(driver, store)``."""
    from loop.agents import LLMAnalyst, LLMResearcher, SdkLLMClient, brief
    from loop.driver import Driver
    from loop.scout import Scout
    from loop.select import PortfolioSelector
    from loop.store import SqliteStore
    from loop.world import ComputedWorldModel

    store = SqliteStore(_db(ctx), ctx.task_id, weights=ctx.cfg.weights)
    adapter = adapter if adapter is not None else resolve_adapter(ctx)
    real = llm is None and not fake_llm              # the research scout needs the real web tools
    if llm is None:
        llm = _fake_llm() if fake_llm else SdkLLMClient(
            ctx.spec, transcript_dir=str(Path(ctx.state_dir) / "transcripts"))
    world = ComputedWorldModel(ctx)
    selector = PortfolioSelector(ctx, world, rng=rng,
                                 brief_fn=lambda kind, st, snap, **kw: brief(kind, st, snap, ctx=ctx, **kw))
    driver = Driver(ctx, store, adapter, world, selector, LLMResearcher(ctx, llm, adapter),
                    LLMAnalyst(ctx, llm), scout=Scout(ctx, llm) if real else None,
                    **({} if executor is None else {"executor": executor}))
    return driver, store


def _llm_line(ctx: Any) -> str:
    """Role routing plus a 3 s read-only reachability check of qwen and the opus credentials."""
    import urllib.request
    from loop.agents.llm import SdkLLMClient, qwen_url
    client = SdkLLMClient(ctx.spec)
    tiers = ", ".join(f"{r}={client.tier(r)}" for r in ("researcher", "review", "scout"))
    opus = "opus oauth " + ("present" if (Path.home() / ".claude" / ".credentials.json").exists()
                            else "MISSING (~/.claude/.credentials.json)")
    if client.default_backend() == "opus":
        return f"llm      {tiers}; every undeclared role routes to opus (llm_default_backend: opus, qwen not needed)\n         {opus}"
    try:
        url = qwen_url()
        with urllib.request.urlopen(url + "/models", timeout=3) as r:
            qwen = f"qwen UP at {url} (HTTP {r.status})"
    except Exception as exc:  # noqa: BLE001 - down is a status, not an error
        qwen = f"qwen DOWN ({type(exc).__name__}): qwen roles fall back to opus"
    return f"llm      {tiers}\n         {qwen}; {opus}"


def _read_fit(ctx: Any, store: Any) -> Any:
    """The world snapshot on a state overlay (nothing persisted); None before any measurement."""
    import math
    if not math.isfinite(ctx.cfg.panel_sd) or len(store.nodes()) < 2:
        return None
    from loop.world import ComputedWorldModel
    return ComputedWorldModel(ctx).fit(store, persist=False)


def _dashboard(ctx: Any, out: str | None) -> int:
    """Render the dashboard once, read-only (like ``status``)."""
    from loop import dashboard, selftune
    from loop.store import SqliteStore
    db = Path(ctx.state_dir) / f"{ctx.task_id}.db"
    if not db.exists():
        print(f"{ctx.task_id}: no state at {db}", file=sys.stderr)
        return 2
    store = SqliteStore(db, ctx.task_id, weights=ctx.cfg.weights)
    ctx.spec_cfg = ctx.cfg
    ctx.cfg = selftune.effective(ctx.cfg, store)
    from loop import leader
    ctx.cfg = leader.effective(ctx.cfg, store)            # our Labless leader re-bases the bar (loop.leader)
    path = dashboard.write(ctx, store, _read_fit(ctx, store),
                           out or Path(ctx.state_dir) / "dashboard" / "index.html")
    store.close()
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


def _status(ctx: Any) -> int:
    """Read-only dashboard: no lock, no writes (the world fit runs on an overlay)."""
    import math
    from collections import Counter
    from loop.contracts import NodeStatus, RunStatus
    from loop.store import SqliteStore
    db = Path(ctx.state_dir) / f"{ctx.task_id}.db"
    if not db.exists():
        print(f"{ctx.task_id}: no state at {db}")
        return 0
    store = SqliteStore(db, ctx.task_id, weights=ctx.cfg.weights)
    from loop import selftune
    spec_cfg = ctx.spec_cfg = ctx.cfg
    ctx.cfg = selftune.effective(ctx.cfg, store)          # what the running loop uses
    from loop import leader
    ctx.cfg = leader.effective(ctx.cfg, store)            # ... with our Labless leader's re-based bar (loop.leader)
    if (ld := store.get_state(leader.K_LEADER)) and ld.get("node"):
        print(f"labless leader {ld.get('run_id')} is OURS (node {ld.get('node')}): leader_median "
              f"{ld.get('configured')} -> effective {ld.get('effective')} ({ld.get('source')})")
    runs, nodes = store.runs(), store.nodes()
    active = [r for r in runs if r.status in (RunStatus.SUBMITTING, RunStatus.RUNNING)]
    measured = math.isfinite(ctx.cfg.panel_sd)
    print(f"{ctx.task_id}  state {ctx.state_dir}")
    print(f"sigma    " + (f"panel {ctx.cfg.panel_sd:.5g} (paired)" if measured
                          else "UNMEASURED: `loop run` calibrates first"))
    from loop.driver import K_STARVED, pool_use
    from loop.selftune import K_PENDING
    print(f"slots    {len(active)}/{ctx.cfg.capacity} used: "
          f"{pool_use(ctx, runs, store.get_state(K_PENDING) or (), store.get_state(K_STARVED) or {})}")
    from loop.driver import K_COURTESY, K_LLM_PAUSE, K_SCARCE, utc_now
    if (lp := store.get_state(K_LLM_PAUSE) or {}).get("active"):
        print(f"llm      PAUSED ({lp['kind']}) until {utc_now(lp['until'])}, since {utc_now(lp['since'])}"
              + ("" if lp.get("resets_at") else f" (reset unknown; probe {lp['probes'] + 1})")
              + ": no new researcher / review / scout")
    elif lp.get("probing"):
        print(f"llm      PROBING after a {lp['kind']} pause: one researcher, no review / scout, until a call works")
    if (sc := store.get_state(K_SCARCE) or {}).get("on"):
        print(f"scarce   ON: {float(sc.get('avg', 0.0)):.1f} running GPUs avg 2h (threshold {ctx.cfg.slurm.scarce_gpus:g})"
              + (f", {','.join(sc['starved'])} STARVED" if sc.get("starved") else "")
              + "; design batches and tuning paused, new screens on <= 2 carriers")
    if (court := store.get_state(K_COURTESY) or {}).get("on"):
        print(f"courtesy_backoff ON: {','.join(court.get('jobs', []))} PENDING -> preemptible capped at "
              f"{court.get('cap')} (no new preemptible jobs until it starts)")
    by_stage: dict[str, Counter] = {}
    for r in runs:
        by_stage.setdefault(r.stage.value, Counter())[r.status.value] += 1
    for stage, c in by_stage.items():
        print(f"runs     {stage:9s} " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    done = [r for r in runs if r.status is RunStatus.DONE]
    print(f"resumed  {sum(r.resumed > 0 for r in done)}/{len(done)} done runs resumed from their latest.pt "
          f"(slurm.resume={ctx.cfg.slurm.resume}; non-exact: seed-level noise in the stats)")
    from loop.contracts import VoidType
    slow = sum(r.void_type is VoidType.TOO_SLOW for r in runs)
    alerts = sum(e["event"] == "ALERT_root_timeout" for e in store.events())
    print(f"timeouts {slow} run(s) killed at the time limit (too_slow), "
          f"{alerts} ALERT_root_timeout" + ("  <-- CHECK THE CLUSTER" if alerts else ""))
    print("nodes    " + (", ".join(f"{k} {v}" for k, v in sorted(Counter(
        n.status.value for n in nodes if n.id != ROOT_NODE_ID).items())) or "none yet"))
    snap = _read_fit(ctx, store)
    inc = snap.nodes.get(snap.incumbent_id) if snap else None
    est = (inc.confirm_estimate or inc.utility) if inc else None
    print(f"incumbent {snap.incumbent_id if snap else ROOT_NODE_ID}" +
          (f": U={est.mean:+.5f} +- {est.sd:.5f} (vs root)" if est else " (baseline)"))
    for n in [n for n in nodes if n.status in ACTIVE] + sorted(
            (n for n in nodes if n.status is NodeStatus.QUEUED),
            key=lambda n: (-(n.prior_bonus or 0.0), n.created_ts))[:5]:
        p = snap.nodes.get(n.id) if snap else None
        print(f"  {n.id} {n.status.value:10s} {n.family}/{n.mechanism_tag}" +
              (f" P(U>0)={p.p_positive:.2f} P(gain>0)={p.p_gain:.2f} P(U>inc)={p.p_beats_incumbent:.2f}" if p else ""))
    from loop.dashboard import arm_shares
    print(f"research bandit draws by arm: {arm_shares(store.events(), time.time())}")
    print(selftune.floors_line(spec_cfg, store))
    from loop.labless import status_lines
    from loop.quality import quality_table
    print(quality_table(store, snap))
    from loop.meta import playbook_lines, scorecard_text
    print(scorecard_text(ctx, store, snap))
    from loop.components import lines as component_lines
    print("\n".join(component_lines(ctx, store, snap)))
    print("\n".join(playbook_lines(store) or ["playbook (none yet: consolidated with FP reviews / every "
                                               f"{ctx.cfg.meta.consolidate_every} reviews)"]))
    from loop.scout import lines as scout_lines
    print("\n".join(scout_lines(ctx, store, time.time())))
    print("\n".join(status_lines(ctx, store)))
    from loop.design import lines as design_lines
    if dl := design_lines(store):
        print(dl)
    from loop.tune import lines as tune_lines
    if tl := tune_lines(store):
        print(tl)
    from loop.campaign import lines as campaign_lines
    if cl := campaign_lines(store):
        print(cl)
    from loop.core.rescue import lines as rescue_lines        # 11-10: moonshot debug / rescue (loop.core.rescue)
    if rl := rescue_lines(store, ctx):
        print("\n".join(rl))
    from loop import extrapolate as EX
    if ctx.cfg.extrapolate.enabled and (el := "\n".join(filter(None, (EX.lines(store), EX.facts(ctx, store, snap))))):
        print(el)
    from loop.legs import lines as legs_lines, plan as legs_plan
    if ll := legs_lines(store):
        print(ll)
    if snap is not None and (lp := legs_plan(ctx, store, snap)) is not None:
        print(f"legs next: round {lp['round']} on {lp['carrier']} with {[r['id'] + '/' + r['subsystem'] for r in lp['legs']]}")
    if snap is not None:                                  # read-only: the epoch stamps stay in an overlay
        from loop import backfill
        from loop.world import StateOverlay
        print("backfill " + backfill.summary(ctx, StateOverlay(store), snap, time.time()))
    if pl := selftune.preempt_line(ctx, store, time.time()):
        print(pl)
    print("\n".join(selftune.lines(store, spec_cfg)))
    for rv in store.reviews()[-2:]:
        print(f"review   {rv.id} ({rv.trigger}) {rv.summary.splitlines()[0][:100] if rv.summary else ''}")
    ev = store.events()
    print(f"last     {ev[-1]['event']} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ev[-1]['ts']))}"
          if ev else "last     no events")
    print(_llm_line(ctx))
    store.close()
    return 0


def _ack(ctx: Any, ids: Sequence[str]) -> int:
    """Close operator proposals (the only write outside the driver; one state key the loop never writes)."""
    from loop import selftune
    from loop.store import SqliteStore
    store = SqliteStore(_db(ctx), ctx.task_id, weights=ctx.cfg.weights)
    open_ = dict(selftune.open_proposals(store))
    pick = list(open_) if list(ids) == ["all"] else [i for i in ids if i in open_]
    store.set_state(selftune.K_ACKED, sorted({*(store.get_state(selftune.K_ACKED) or []), *pick}))
    print(f"acked {len(pick)}: {pick}" + (f"; unknown or closed: {sorted(set(ids) - set(pick) - {'all'})}"
                                         if set(ids) - set(pick) - {"all"} else ""))
    store.close()
    return 0


def _audit_components(ctx: Any) -> int:
    """On-demand causal component audit (loop.components): prints the table, applies the mode changes."""
    from loop import components
    from loop.store import SqliteStore
    store = SqliteStore(_db(ctx), ctx.task_id, weights=ctx.cfg.weights)
    snap = _read_fit(ctx, store)
    acts = components.audit(ctx, store, snap, time.time())
    print("\n".join(components.lines(ctx, store, snap) + (acts or ["no mode change"])))
    store.close()
    return 0


def reload_measured(ctx: Any, load: Any, updates: Any) -> Any:
    """The context after ``calibrate`` wrote the measured sigmas into the task yaml: re-read from disk, but a yaml
    that no longer DECODES (this process runs older code than the file: 2026-09-29, ``loop.labless: unknown key``
    crashed a fresh run right after a 5-seed calibration) keeps the already-parsed context and applies just the
    measured values (``updates``: ``calibrate.compute`` paths -> floats) to it, with a loud line; the next process
    start reads the yaml with its own code. Unmeasured noise (:class:`UnmeasuredLoopNoise`) still raises."""
    try:
        return load()
    except UnmeasuredLoopNoise:
        raise
    except Exception as exc:  # noqa: BLE001 - version skew between this process's code and the yaml on disk
        print(f"{_utc()} WARNING: re-reading {ctx.spec_path} after calibrate failed ({type(exc).__name__}: "
              f"{str(exc)[:300]}); the file is newer than this process's code. Continuing on the already-parsed "
              f"spec with the {len(updates)} measured value(s) applied in memory; RESTART the loop to pick the "
              f"file up.", file=sys.stderr, flush=True)
    cfg, noise = ctx.cfg, ctx.noise
    axis = {a: updates[("loop", "axis_sd", a)] for a in cfg.axes if ("loop", "axis_sd", a) in updates}
    cfg = dataclasses.replace(cfg, **({"panel_sd": updates[("loop", "panel_sd")]} if ("loop", "panel_sd") in updates
                                      else {}), axis_sd={**dict(cfg.axis_sd), **axis})
    kw = {k: updates[("noise", k)] for k in ("seed_transfer_sd", "replicate_sd") if ("noise", k) in updates}
    return dataclasses.replace(ctx, cfg=cfg, noise=dataclasses.replace(noise, **kw) if kw else noise)


def _utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _wait(seconds: float) -> None:          # callers slice it (driver.nap), so a stop lands in ~1 s
    time.sleep(seconds)


def _shadow(argv: Sequence[str]) -> int:
    """``shadow --task T --db <copy> [--refresh-from <live db>]``: the v3 tick (loop.tick) against a ledger copy with
    a dry-run adapter -- statuses and ranked actions logged per tick (stdout + ``<copy>.shadow.jsonl``), nothing
    submitted, no LLM session. ``--refresh-from`` re-takes the copy from the live ledger (read-only) every
    ``--refresh-min``. ``run`` stays the old driver until the cutover."""
    from loop.spec import load_context
    from loop.tick import ModuleCore, shadow
    ap = argparse.ArgumentParser(prog="python -m loop shadow")
    ap.add_argument("--task", required=True, help="task spec YAML (pools, seeds, ...)")
    ap.add_argument("--db", required=True, help="the ledger COPY the ticks run on (never the live one)")
    ap.add_argument("--refresh-from", default=None, help="live ledger to re-copy from (opened read-only)")
    ap.add_argument("--refresh-min", type=float, default=60.0)
    ap.add_argument("--task-id", default=None, help="the ledger's task id (default: the spec's)")
    ap.add_argument("--every-s", type=float, default=60.0, help="seconds between ticks")
    ap.add_argument("--max-ticks", type=int, default=None)
    args = ap.parse_args(list(argv))
    if args.refresh_from and Path(args.refresh_from).resolve() == Path(args.db).resolve():
        ap.error("--db must be a copy, not the --refresh-from ledger")
    ctx = load_context(args.task, state_dir=str(Path(args.db).resolve().parent), require_measured=False)
    if args.task_id:
        ctx = dataclasses.replace(ctx, task_id=args.task_id)
    core = ModuleCore()
    if core.missing:
        print(f"{_utc()} shadow: loop.core.{{{','.join(core.missing)}}} not importable yet: their stubs stand in",
              flush=True)
    shadow(ctx, args.db, core, refresh_from=args.refresh_from, refresh_s=60.0 * args.refresh_min,
           max_ticks=args.max_ticks, every_s=args.every_s, stop=STOP.is_set)
    return 0


def _import_runs(argv: Sequence[str]) -> int:
    """``import-runs --task T [--state-dir D] [--dry-run]``: import the operator's manual runs (loop.localruns) into
    the task's db now. No instance lock: the db is WAL with a 30 s busy timeout, so it is safe beside a running loop
    (which re-imports on its own cadence when ``local_runs.enabled``). ``--dry-run`` writes nothing."""
    from loop import localruns as LR
    from loop.spec import load_context
    from loop.store import SqliteStore
    ap = argparse.ArgumentParser(prog="python -m loop import-runs")
    ap.add_argument("--task", required=True, help="task spec YAML")
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="read and score, write nothing")
    args = ap.parse_args(list(argv))
    ctx = load_context(args.task, state_dir=args.state_dir, require_measured=False)
    if not LR.enabled(ctx):
        print(f"refused: local_runs is off for {ctx.task_id} (loop.local_runs.enabled)", file=sys.stderr)
        return 2
    store = SqliteStore(_db(ctx), ctx.task_id, weights=ctx.cfg.weights)
    try:
        res = LR.import_runs(ctx, store, time.time(), write=not args.dry_run)
    finally:
        store.close()
    print(f"{_utc()} {LR.LocalRuns._line(res)}" + (" (dry run: nothing written)" if args.dry_run else ""))
    for u in res["unfinished"]:
        print(f"  unfinished: {u}")
    if res["missing"]:
        print(f"  no output dir: {', '.join(res['missing'])}")
    print(res["digest"]["text"])
    return 0


def _run_v3(argv: Sequence[str]) -> int:
    """``run --task T [--state-dir D] [--max-ticks N] [--fake-llm]``: the live v3 loop (loop.tick.live) on the spec's
    ledger with the real adapter and researcher; holds ``<db>.lock`` (exit 3 when taken). Noise is fitted from the
    ledger each tick, so no calibration gate. SIGTERM: drain research, stop between ticks, SLURM jobs keep running."""
    from loop.agents import LLMResearcher, SdkLLMClient
    from loop.spec import load_context
    from loop.tick import ModuleCore, TickFatal, live
    ap = argparse.ArgumentParser(prog="python -m loop run")
    ap.add_argument("--task", required=True, help="task spec YAML")
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument("--fake-llm", action="store_true", help="offline FakeLLMClient")
    ap.add_argument("--no-selfcheck", action="store_true",
                    help="skip the startup selfcheck (loop.selfcheck; it never runs with --fake-llm)")
    args = ap.parse_args(list(argv))
    ctx = load_context(args.task, state_dir=args.state_dir, require_measured=False)
    lock = _lock(ctx)                                 # held (open) until this returns
    if lock is None:
        return 3
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    core = ModuleCore()
    if core.missing:
        print(f"{_utc()} refused: loop.core.{{{','.join(core.missing)}}} not importable", file=sys.stderr)
        return 2
    adapter = resolve_adapter(ctx)
    llm = _fake_llm() if args.fake_llm else SdkLLMClient(ctx.spec, transcript_dir=str(Path(ctx.state_dir) / "transcripts"))
    print(f"{_utc()} loop run (v3 tick): db {_db(ctx)}, base {getattr(ctx.spec, 'repo', '?')}", flush=True)
    try:
        live(ctx, _db(ctx), core, adapter, LLMResearcher(ctx, llm, adapter), max_ticks=args.max_ticks,
             stop=STOP.is_set, out=lambda m: print(m, flush=True),
             selfcheck=not (args.fake_llm or args.no_selfcheck))
    except TickFatal as exc:                          # exit non-zero: systemd (Restart=always) starts a fresh process
        print(f"{_utc()} FATAL: {exc}; exiting for a restart", file=sys.stderr, flush=True)
        return 1
    return 0


def _v3(ap: argparse.ArgumentParser, argv: Sequence[str]) -> tuple[Any, Any, Path]:
    """(args, ctx, db) of a v3 command: ``--db`` (a ledger copy: the state dir is its directory) or the spec's db."""
    from loop.spec import load_context
    ap.add_argument("--task", required=True, help="task spec YAML")
    ap.add_argument("--db", default=None, help="the ledger (default: <state-dir>/<task>.db); a COPY for dry runs")
    ap.add_argument("--task-id", default=None, help="the ledger's task id (default: the spec's)")
    args = ap.parse_args(list(argv))
    state = str(Path(args.db).resolve().parent) if args.db else None
    ctx = load_context(args.task, state_dir=state, require_measured=False)
    if args.task_id:
        ctx = dataclasses.replace(ctx, task_id=args.task_id)
    return args, ctx, Path(args.db) if args.db else _db_path(ctx)


def _fake_meta_llm() -> Any:
    """A scripted meta-review (``meta --fake-llm``): one tunable change, two locked targets, one prompt patch."""
    from loop.agents.llm import FakeLLMClient
    first = (Path(__file__).parent / "prompts" / "ideation.md").read_text().splitlines()[1]
    return FakeLLMClient({"review": lambda **_: {
        "diagnosis": "fake-llm sample review: tests open on beliefs the harm limit already blocks half of.",
        "proposals": [
            {"kind": "config", "path": "acquire.harm_q", "value": 0.3, "rationale": "fewer harm-blocked tests",
             "predicted_effect": "more tests per report window at equal robustness harm"},
            {"kind": "config", "path": "test.ALPHA", "value": 0.2, "rationale": "pass more", "predicted_effect": "-"},
            {"kind": "prompt", "path": "loop/core/test.py", "patch": {"find": "ALPHA = 0.1", "replace": "ALPHA = 0.5"},
             "rationale": "objective hacking probe", "predicted_effect": "-"},
            {"kind": "prompt", "path": "prompts/ideation.md", "patch": {"find": first, "replace": first + (
                " State the single number your change must move and by how much before you write code.")},
             "rationale": "sharper falsifiable predictions", "predicted_effect": "engagement hit rate up"}]}})


def _meta(argv: Sequence[str]) -> int:
    """``meta --task T [--db <copy>] [--review] [--fake-llm]``: one meta-cycle now (loop.core.meta): the report (printed,
    written to <state>/meta/), the auto-revert check, and with ``--review`` the meta-review and its acceptance."""
    from loop.core.data import load
    from loop.core.meta import Meta
    from loop.store import SqliteStore
    ap = argparse.ArgumentParser(prog="python -m loop meta")
    ap.add_argument("--review", action="store_true", help="also run the Opus meta-review and accept / refuse")
    ap.add_argument("--fake-llm", action="store_true", help="a scripted sample review instead of Opus")
    args, ctx, db = _v3(ap, argv)
    store = SqliteStore(db, ctx.task_id, weights=ctx.cfg.weights)
    llm = None
    if args.review:
        from loop.agents.llm import SdkLLMClient
        llm = _fake_meta_llm() if args.fake_llm else SdkLLMClient(ctx.spec)
    meta = Meta(ctx, store, llm=llm)
    lines = meta.cycle(load(db, ctx.task_id), llm=llm, wait=True)
    path = (store.get_state("meta.last") or {}).get("path")
    if path:
        print(Path(path).read_text())
    print("\n".join(f"{_utc()} {x}" for x in lines))
    store.close()
    return 0


def _revert(argv: Sequence[str]) -> int:
    """``revert v<N> --task T [--db D] [--prompt]``: make config (or prompt) version N the active one."""
    from loop.core.meta import Versions
    from loop.store import SqliteStore
    ap = argparse.ArgumentParser(prog="python -m loop revert")
    ap.add_argument("version", help="v<N>")
    ap.add_argument("--prompt", action="store_true", help="a prompt version instead of a config version")
    args, ctx, db = _v3(ap, argv)
    if not re.fullmatch(r"v?\d+", args.version):
        ap.error("version is v<N>")
    store = SqliteStore(db, ctx.task_id, weights=ctx.cfg.weights)
    try:
        print(Versions(store, ctx.state_dir).revert("prompt" if args.prompt else "config",
                                                    int(args.version.lstrip("v")), "operator: loop revert"))
    except KeyError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loop", description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run", "calibrate", "status", "review", "ack", "dashboard", "serve",
                                            "audit-components", "digest", "backtest"])
    ap.add_argument("ids", nargs="*", help="ack: operator proposal ids (from `status`) to close; 'all' = every open one")
    ap.add_argument("--task", default=None, help="task spec YAML (tasks/<name>.yaml); every command but serve")
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument("--fake-llm", action="store_true", help="offline FakeLLMClient")
    ap.add_argument("--seeds", type=int, default=None, help="calibrate: number of seeds")
    ap.add_argument("--out", default=None, help="dashboard: output file (default <state-dir>/dashboard/index.html)")
    ap.add_argument("--port", type=int, default=8765, help="serve: localhost port")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["backtest"]:                      # read-only replay over a ledger COPY (loop.core.backtest)
        from loop.core.backtest import main as backtest
        return backtest(argv[1:])
    if argv[:1] == ["shadow"]:                        # v3 ticks on a ledger COPY, nothing submitted (loop.tick)
        return _shadow(argv[1:])
    if argv[:1] == ["meta"]:                          # one v3 meta-cycle now (loop.core.meta)
        return _meta(argv[1:])
    if argv[:1] == ["revert"]:                        # make a config / prompt version active again
        return _revert(argv[1:])
    if argv[:1] == ["import-runs"]:                   # the operator's manual runs as external evidence
        return _import_runs(argv[1:])
    if argv[:1] == ["selfcheck"]:                     # the LLM-free end-to-end preflight (loop.selfcheck)
        from loop.selfcheck import main as selfcheck
        return selfcheck(argv[1:])
    if argv[:1] == ["effective-config"]:              # the resolved config of a task (loop.effective), read-only
        from loop.effective import main as effective
        return effective(argv[1:])
    if argv[:1] == ["run"]:                           # the live v3 tick (cutover step 12)
        return _run_v3(argv[1:])
    if argv[:1] == ["run-v2"]:                        # the old driver, until its deletion
        argv = ["run", *argv[1:]]
    args = ap.parse_args(argv)
    if args.command == "serve":                       # read-only static server of <state-dir>/dashboard
        if not args.state_dir:
            ap.error("serve needs --state-dir")
        from loop.dashboard import serve
        serve(Path(args.state_dir) / "dashboard", args.port)
        return 0
    if not args.task:
        ap.error("--task is required")
    from loop.calibrate import CalibrationFailed, calibrate, calibration_seeds, compute
    from loop.contracts import SubmissionRejected
    from loop.spec import load_context
    from loop.store import SqliteStore

    def load(require_measured: bool) -> Any:
        return load_context(args.task, state_dir=args.state_dir, require_measured=require_measured)

    unmeasured = False
    try:
        ctx = load(args.command in ("run", "review"))
    except UnmeasuredLoopNoise as exc:
        if args.command != "run":
            print(f"{_utc()} refused: {exc}", file=sys.stderr)
            return 2
        print(f"{_utc()} {exc}\n`run` calibrates first, then continues into the loop.", file=sys.stderr)
        ctx, unmeasured = load(False), True
    if args.command == "status":
        return _status(ctx)
    if args.command == "dashboard":
        return _dashboard(ctx, args.out)
    if args.command == "ack":
        return _ack(ctx, args.ids)
    if args.command == "audit-components":
        return _audit_components(ctx)
    if args.command == "digest":                    # write <state_dir>/digest.md now (read-only otherwise)
        from loop import digest
        store = SqliteStore(_db(ctx), ctx.task_id, weights=ctx.cfg.weights)
        now = time.time()
        print("\n".join(digest.tick(ctx, store, now, force=True)))
        print(digest.render(digest.numbers(ctx, store, now), now))
        return 0
    seeds = calibration_seeds(ctx, args.seeds)
    if args.dry_run and (unmeasured or args.command == "calibrate"):
        db = _db_path(ctx)
        store = SqliteStore(db, ctx.task_id, weights=ctx.cfg.weights) if db.exists() else None
        calibrate(ctx, store, resolve_adapter(ctx), seeds=seeds, dry_run=True)
        if store is None or args.command != "run":
            return 0
        try:        # what the loop would do right after: plan one tick on provisional sigmas
            res = compute(ctx, store.runs(node_id=ROOT_NODE_ID), seeds)
        except RuntimeError as exc:
            print(f"then the loop (no provisional sigma yet: {exc})")
            return 0
        from loop.spec import raw_loop_block, sigma_scaled
        prov = dataclasses.replace(ctx, cfg=sigma_scaled(dataclasses.replace(       # score-unit defaults follow it
            ctx.cfg, panel_sd=res.paired_sd[PANEL], axis_sd={a: res.paired_sd[a] for a in ctx.cfg.axes}),
            raw_loop_block(ctx.spec)))
        print(f"then the loop (provisional sigma_panel {res.paired_sd[PANEL]:.5g} from the finished "
              f"baselines; the measured one replaces it):")
        driver, _ = build(prov, fake_llm=args.fake_llm)
        print("\n".join(driver.tick(dry_run=True)))
        return 0
    live = not args.dry_run
    lock = _lock(ctx) if live else None             # held (open) until main returns
    if live and lock is None:
        return 3
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "calibrate" or unmeasured:
            store = SqliteStore(_db(ctx), ctx.task_id, weights=ctx.cfg.weights)
            res = calibrate(ctx, store, resolve_adapter(ctx), seeds=seeds, sleep=_wait,
                            stop=STOP.is_set, echo=lambda m: print(_utc(), m, flush=True),
                            max_ticks=args.max_ticks if args.command == "calibrate" else None)
            store.close()
            if res is None or args.command == "calibrate":
                return 0 if res is not None or STOP.is_set() else 3
            ctx = reload_measured(ctx, lambda: load(True), res.updates)   # the spec now carries the measured sigmas
    except (CalibrationFailed, SubmissionRejected) as exc:
        print(_utc(), str(exc).replace("\n", " "), file=sys.stderr)
        return 4
    driver, store = build(ctx, fake_llm=args.fake_llm)
    if args.command == "run":
        if args.dry_run:
            print("\n".join(driver.tick(dry_run=True)))
            return 0
        driver.sleep = _wait
        try:
            driver.run(max_ticks=args.max_ticks, stop=STOP.is_set)
        except SubmissionRejected as exc:
            print(_utc(), str(exc).replace("\n", " "), file=sys.stderr)
            return 4
    else:                                            # review: force one now and wait for it
        driver.start_review("manual")
        driver.drain()
        acts: list[str] = []
        driver._harvest(acts)
        print("\n".join(acts) or "review produced nothing (see the store log)")
    return 0


if __name__ == "__main__":
    # SIGTERM (systemd stop) / SIGINT: no new LLM job; in-flight research is drained for <= researcher.drain_min
    # (Driver.run; systemd TimeoutStopSec must exceed it), then the lock is released with SLURM jobs left running;
    # sessions still in flight are re-issued after the restart, queued under max_concurrent.
    # Every exit goes through os._exit: the research / Labless / meta-review worker threads (non-daemon
    # ThreadPoolExecutor threads) must never hold the process alive after the loop ended -- an exception escaping
    # main() included (exit 1: systemd restarts the unit).
    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, lambda *_: STOP.set())
    try:
        _rc = main()
    except SystemExit as _exit:                     # argparse --help (0) / usage errors (2): not a crash
        _code = _exit.code
        if _code is not None and not isinstance(_code, int):
            print(_code, file=sys.stderr)           # sys.exit("message") convention: print it, exit 1
            _code = 1
        _rc = 0 if _code is None else _code
    except BaseException:  # noqa: BLE001 - printed, then the hard exit below
        import traceback
        print(f"{_utc()} FATAL: uncaught exception; exiting 1\n{traceback.format_exc()}", file=sys.stderr)
        _rc = 1
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(_rc if isinstance(_rc, int) else 1)
