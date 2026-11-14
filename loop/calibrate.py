"""loop.calibrate — noise-calibration mode (docs/loop/DESIGN.md §7): root x calibration seeds x 2
replicates, then the measured sds are written back into the task YAML, line by line."""

from __future__ import annotations

import datetime as _dt
import itertools
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from loop import seeds as S
from loop.contracts import PANEL, ROOT_NODE_ID, Run, RunStatus, Stage
from loop.spec import LoopContext
from loop.world.noise import CAL_KEY

_KEY = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_.\-]+):(?P<rest>.*)$")
_STAGES = (Stage.BASELINE, Stage.CALIBRATE)


@dataclass(frozen=True)
class CalibrationResult:
    seed_sd: Mapping[str, float]
    replicate_sd: Mapping[str, float]
    paired_sd: Mapping[str, float]
    run_ids: tuple[str, ...]
    updates: Mapping[tuple[str, ...], float] = field(default_factory=dict)
    written: bool = False
    #: per-run sds after pooling with ``loop.noise_prior`` (= seed_sd without a prior) and the prior's dfs
    pooled_sd: Mapping[str, float] = field(default_factory=dict)
    prior_df: float = 0.0


def calibration_seeds(ctx: LoopContext, k: int | None = None) -> tuple[int, ...]:
    """``seeds.calibration``, else screen + promote + confirm_pool[:4]; ``k`` takes the
    first k of that list extended by the confirm pool."""
    sp = ctx.cfg.seeds
    seeds = sp.calibration_set
    if k is not None:
        seeds = tuple(dict.fromkeys(seeds + tuple(sp.confirm_pool)))[:k]
    if len(seeds) < 2:
        raise ValueError(f"calibration needs >= 2 seeds, got {list(seeds)}")
    return tuple(int(s) for s in seeds)


def compute(ctx: LoopContext, runs: Sequence[Run], seeds: Sequence[int]) -> CalibrationResult:
    """The three sds per target from DONE root runs (pure). ``seed_sd`` uses EVERY finished
    baseline (already paid for, even at a seed outside ``seeds``); replicates only ``seeds``."""
    done = {(r.stage, int(r.seed)): r for r in runs if r.status is RunStatus.DONE and r.scores}
    rep1 = [r for (st, _), r in sorted(done.items(), key=lambda kv: kv[0][1]) if st is Stage.BASELINE]
    pairs = [(done[(Stage.BASELINE, s)], done[(Stage.CALIBRATE, s)]) for s in seeds
             if (Stage.BASELINE, s) in done and (Stage.CALIBRATE, s) in done]
    if len(rep1) < 2:
        raise RuntimeError(f"calibration: only {len(rep1)} replicate-1 run(s) finished; need >= 2")

    def value(run: Run, t: str) -> float:
        return float(run.scores[t]) if t != PANEL else \
            sum(ctx.cfg.weights[a] * float(run.scores[a]) for a in ctx.cfg.axes)
    targets = (PANEL, *ctx.cfg.axes)
    if any(not math.isfinite(value(r, t)) for r in done.values() for t in targets):
        raise RuntimeError("calibration: non-finite scores; not a measurement")
    seed_sd = {t: statistics.stdev([value(r, t) for r in rep1]) for t in targets}
    pooled, prior_df = pool_prior(ctx, seed_sd, len(rep1) - 1)
    rep_sd = {}
    for t in targets if pairs else ():
        diffs = [value(b, t) - value(a, t) for a, b in pairs]
        rep_sd[t] = (statistics.stdev(diffs) if len(diffs) >= 2 else abs(diffs[0])) / math.sqrt(2)
    paired = {t: math.sqrt(2) * v for t, v in pooled.items()}
    updates = {("noise", "seed_transfer_sd"): pooled[PANEL], ("loop", "panel_sd"): paired[PANEL],
               **{("loop", "axis_sd", a): paired[a] for a in ctx.cfg.axes},
               **({("noise", "replicate_sd"): rep_sd[PANEL]} if PANEL in rep_sd else {})}
    # a zero sd (e.g. bit-identical replicates) is not a measurement: leave that line alone
    return CalibrationResult(seed_sd, rep_sd, paired,
                             tuple(r.id for r in rep1) + tuple(b.id for _, b in pairs),
                             {k: v for k, v in updates.items() if math.isfinite(v) and v > 0},
                             pooled_sd=pooled, prior_df=prior_df)


def pool_prior(ctx: LoopContext, seed_sd: Mapping[str, float], df: int) -> tuple[dict[str, float], float]:
    """(per-run sds pooled with the external prior ``loop.noise_prior``, its dfs): per target
    ``sqrt((df s² + df0 s0²) / (df + df0))`` (s0: the prior's panel_sd / axis_sd); no prior (df0 0): ``seed_sd``."""
    npc = getattr(ctx.cfg, "noise_prior", None)
    df0 = float(getattr(npc, "df", 0.0) or 0.0)
    if df0 <= 0:
        return dict(seed_sd), 0.0
    prior = {PANEL: float(npc.panel_sd), **{a: float(v) for a, v in npc.axis_sd.items()}}
    return {t: (math.sqrt((df * s * s + df0 * prior[t] ** 2) / (df + df0)) if t in prior else s)
            for t, s in seed_sd.items()}, df0


def rewrite_yaml(text: str, updates: Mapping[tuple[str, ...], float], comment: str) -> str:
    """Replace the scalar at each dotted path, touching only those lines. Handles block
    mappings and one level of inline flow mapping (``axis_sd: {a: UNMEASURED}``). A path
    the spec does not declare raises KeyError."""
    lines, stack, todo = text.splitlines(keepends=True), [], dict(updates)
    for i, line in enumerate(lines):
        body = line.rstrip("\n")
        m = _KEY.match(body)
        if not m or body.lstrip()[:1] in ("#", "-"):
            continue
        indent = len(m.group("indent"))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = tuple(k for _, k in stack) + (m.group("key"),)
        value = m.group("rest").split(" #", 1)[0].strip()
        head = f"{m.group('indent')}{m.group('key')}: "
        tail = f"  # {comment}" + ("\n" if line.endswith("\n") else "")
        children = {k[-1]: v for k, v in todo.items() if k[:-1] == path}
        if path in todo:
            lines[i] = f"{head}{todo.pop(path):.6g}{tail}"
        elif value.startswith("{") and children:
            flow = yaml.safe_load(value) or {}
            for key, v in children.items():
                flow[key] = f"{v:.6g}"
                todo.pop(path + (key,))
            lines[i] = head + "{" + ", ".join(f"{k}: {v}" for k, v in flow.items()) + "}" + tail
        if not value:
            stack.append((indent, m.group("key")))
    if todo:
        raise KeyError(f"calibrate: the spec does not declare {['.'.join(k) for k in todo]}")
    return "".join(lines)


class CalibrationFailed(RuntimeError):
    """Too few runs finished; the message is one line naming why each failed."""


def _failure_reasons(runs: Sequence[Run]) -> str:
    out = []
    for r in runs:
        if r.status is RunStatus.DONE:
            continue
        tail = (r.log_tail or "").strip().splitlines()
        vt = f"/{r.void_type.value}" if r.void_type else ""
        out.append(f"{r.id}: {r.status.value}{vt}" + (f" ({tail[-1][:120]})" if tail else ""))
    return "; ".join(out[:16]) + (f"; +{len(out) - 16} more" if len(out) > 16 else "")


def _unneeded(runs: Sequence[Run], wanted: Sequence[tuple[Stage, int]]) -> list[Run]:
    """Replicate rows at a seed calibration no longer wants (e.g. cancelled after the seed
    set shrank): never retried, never waited for."""
    return [r for r in runs if r.stage is Stage.CALIBRATE and (r.stage, int(r.seed)) not in wanted
            and r.status is not RunStatus.DONE]


def calibrate(ctx: LoopContext, store: Any, adapter: Any, *, seeds: Sequence[int] | None = None,
              dry_run: bool = False, write: bool = True, max_ticks: int | None = None,
              clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
              echo: Callable[[str], Any] = print,
              stop: Callable[[], bool] = lambda: False) -> CalibrationResult | None:
    """Run (or resume: existing rows are reused, running jobs re-attached) the calibration runs,
    then write the measured sds. ``dry_run`` prints the plan (against ``store`` when given) and
    touches nothing; ``stop()`` true ends it between ticks (None, resumable)."""
    from loop.driver import Courtesy, RunBook, hb_line, nap   # shares submit / poll / resume with the driver

    seeds = tuple(seeds) if seeds else calibration_seeds(ctx)
    wanted = [(st, int(s)) for st in _STAGES for s in seeds]
    if dry_run:
        echo(f"calibrate {ctx.task_id}: root x seeds {list(seeds)} x 2 replicates = "
             f"{len(wanted)} runs, {ctx.cfg.capacity} at a time")
        command = getattr(adapter, "command", None)       # the exact launch, when the adapter has one
        rows = store.runs(node_id=ROOT_NODE_ID) if store is not None else []
        have = {(r.stage, int(r.seed)): r for r in rows}
        # loop.calibrate.pools: spread -- the plan shows the pool each new run would take (the fewest calibration runs
        # so far, declaration order breaking ties; live, a pool with no free slot is skipped). Without this the
        # printed launch had no pool and showed the first (primary) for all 10 runs, though the live spread
        # alternates primary / idle.
        spread = getattr(getattr(ctx.cfg, "calibrate", None), "pools", "auto") == "spread"
        names = [p.name for p in ctx.cfg.pools]
        held = Counter(r.pool or names[0] for r in rows if r.stage in _STAGES)
        for stage, seed in wanted:
            run = have.get((stage, seed)) or Run(
                id=Run.make_id(ROOT_NODE_ID, stage, seed), task_id=ctx.task_id,
                node_id=ROOT_NODE_ID, seed=seed, stage=stage, status=RunStatus.SUBMITTING,
                pool=min(names, key=lambda n: (held[n], names.index(n))) if spread and names else None)
            if (stage, seed) not in have:
                held[run.pool or (names[0] if names else "")] += 1
                echo(f"  would run {run.id}" + (f" on {run.pool}" if run.pool else "")
                     + (f": {command(run)}" if command else ""))
            elif run.status in (RunStatus.RUNNING, RunStatus.SUBMITTING):
                echo(f"  would attach {run.id} to job {run.job_id or '(by receipt)'} ({run.status.value}; "
                     f"not resubmitted)")
            else:
                echo(f"  {run.id}: {run.status.value}")
        for r in _unneeded(rows, wanted):
            echo(f"  not needed: {r.id} (job {r.job_id}, {r.status.value}): outside the calibration "
                 f"seeds, never retried")
        extra = [r.id for r in rows if r.stage is Stage.BASELINE and r.status is RunStatus.DONE
                 and int(r.seed) not in seeds]
        if extra:
            echo(f"  seed_sd also uses the finished baselines {extra}")
        echo(f"  then rewrite noise.*_sd, loop.panel_sd, loop.axis_sd.* in {ctx.spec_path}")
        return None
    root = store.ensure_root(adapter.root_commit())
    book = RunBook(ctx, store, adapter, clock=clock)
    courtesy = Courtesy(ctx, store, adapter, book, clock)      # the preemptible cap applies here too
    cc = getattr(ctx.cfg, "calibrate", None)
    if getattr(cc, "pools", "auto") == "spread":
        book.pool = _spread(ctx, store, book)                  # type: ignore[method-assign]
    max_conc = int(getattr(cc, "max_concurrent", 0) or 0) or ctx.cfg.capacity
    next_hb = clock()
    for ticks in itertools.count(1):
        if stop():
            echo("calibrate: stop requested; re-run to resume")
            return None
        book.resolve_submitting()
        for r in book.poll():
            echo(f"run {r.id} -> {r.status.value}")
        for r in _unneeded(store.runs(node_id=ROOT_NODE_ID), wanted):
            if S.is_retryable(r, book.infra_retry):
                store.update_run(r.id, attempt=book.infra_retry + 1)
                store.log("calibration_run_not_needed", run_id=r.id, status=r.status.value)
                echo(f"run {r.id} ({r.status.value}) not needed: never retried")
        have = {(r.stage, int(r.seed)): r for r in store.runs(node_id=ROOT_NODE_ID)}
        court: list[str] = []
        courtesy.step(court)
        for line in court:
            echo(line)
        free = min(ctx.cfg.capacity - len(book.active()), max_conc - len(book.active()), book.free())
        for key in wanted:
            if free > 0 and key not in have:
                book.launch(root, *key)
            elif free > 0 and S.is_retryable(have[key], book.infra_retry):
                book.retry(have[key], root)
            else:
                continue
            free -= 1
        runs = [r for r in store.runs(node_id=ROOT_NODE_ID) if (r.stage, int(r.seed)) in wanted]
        pending = [r for r in runs if r.status in (RunStatus.SUBMITTING, RunStatus.RUNNING)
                   or S.is_retryable(r, book.infra_retry)]
        if len(runs) == len(wanted) and not pending:
            break
        if max_ticks is not None and ticks >= max_ticks:
            echo(f"calibrate: stopped after {ticks} ticks, {len(pending)} run(s) pending; re-run to resume")
            return None
        if ctx.cfg.heartbeat.log_every_min > 0 and clock() >= next_hb:
            echo(hb_line(ctx, store, "calibrate"))
            next_hb = clock() + 60 * ctx.cfg.heartbeat.log_every_min
        nap(ctx.cfg.heartbeat.poll_s, stop, sleep)       # a SIGTERM lands within ~1 s
    try:
        res = compute(ctx, store.runs(node_id=ROOT_NODE_ID), seeds)
    except RuntimeError as exc:
        raise CalibrationFailed(f"calibrate failed: {exc}; run failures: "
                                f"{_failure_reasons(runs) or 'none recorded'}") from exc
    for t in res.seed_sd:
        echo(f"{t}: seed_sd={res.seed_sd[t]:.6g} replicate_sd={res.replicate_sd.get(t, math.nan):.6g}"
             f" paired_sd={res.paired_sd[t]:.6g}")
    store.log("calibrated", seeds=list(seeds), run_ids=list(res.run_ids),
              seed_sd=dict(res.seed_sd), replicate_sd=dict(res.replicate_sd))
    n_base = sum(r.startswith(f"{ROOT_NODE_ID}:{Stage.BASELINE.value}:") for r in res.run_ids)
    # the dfs behind the written sigmas: the calibration's own + the external prior's (loop.noise_prior)
    store.set_state(CAL_KEY, {"df": n_base - 1 + int(round(res.prior_df)), "run_ids": list(res.run_ids),
                              **({"prior_df": res.prior_df} if res.prior_df else {})})
    if not write:
        return res
    path = Path(ctx.spec_path)
    path.write_text(rewrite_yaml(path.read_text(), res.updates,
                                 f"measured {_dt.date.fromtimestamp(clock()).isoformat()} by loop "
                                 f"calibrate, runs={','.join(res.run_ids)}"))
    echo(f"wrote {len(res.updates)} measured value(s) into {path}")
    return CalibrationResult(res.seed_sd, res.replicate_sd, res.paired_sd, res.run_ids,
                             res.updates, written=True, pooled_sd=res.pooled_sd, prior_df=res.prior_df)


def _spread(ctx: LoopContext, store: Any, book: Any) -> Callable[..., str]:
    """``loop.calibrate.pools: spread``: a calibration run goes to the open pool (primary or preemptible) holding the
    FEWEST calibration runs so far (declaration order breaks ties), so the replicate pairs see every pool's nodes and
    load, as production runs do; '' = no pool has a free slot."""
    names = [p.name for p in ctx.cfg.pools]
    first = names[0]

    def pool(node: Any = None, stage: Any = None, seed: Any = None) -> str:
        open_ = set(book._open(False)) | set(book._open(True))
        if not open_:
            return ""
        held = {n: 0 for n in names}
        for r in store.runs(node_id=ROOT_NODE_ID):
            if r.stage in _STAGES:
                held[r.pool or first] = held.get(r.pool or first, 0) + 1
        return min((n for n in names if n in open_), key=lambda n: (held.get(n, 0), names.index(n)))
    return pool
