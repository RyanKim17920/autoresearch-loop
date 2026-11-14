"""loop.resume — resume-on-requeue (``slurm.resume``): a preempted run continues from its own latest.pt instead of
step 0. The adapter does the work (a config copy per resumable run, armed with ``train.resume`` once the run saved
its first checkpoint; the receipt counts ``resumed`` starts so stats can exclude / inspect them); this module only
says which runs are resumable. Paired deltas lean on same-seed near-determinism (replicate sd ~0.0005) and train.py's
resume is not bit-exact (the data loader restarts its permutation and the RNG streams are reseeded). Under "on"
(the default, operator 2026-10-01) every run is resumable and the statistics say so: a run with ``resumed`` > 0
gives its paired deltas seed-level noise and stays out of the noise fit (loop.world.noise.resume_var). Under
"calibrate" it stays off until a one-time calibration passes: root re-run (stage AUDIT, outside the world model and
the held-out audit) at a seed whose BASELINE finished, force-interrupted at ``FRAC`` of training and resumed; on iff
|final_score - the uninterrupted run's| <= ``slurm.resume_tol``, else it stays off and the reason is logged."""

from __future__ import annotations

from typing import Any

from loop import seeds as S
from loop.contracts import ROOT_NODE_ID, JobHandle, Run, RunStatus, Stage
from loop.spec import LoopContext

K_RESUME = "resume"       # {"status": running|passed|failed, "run", "ref", "interrupted", "delta", "why"}
FRAC = 0.5                # the calibration run is interrupted once its last train log is past this sample fraction


def tick(ctx: LoopContext, store: Any, book: Any, adapter: Any) -> list[str]:
    """Set the adapter's resumable runs; start / interrupt / judge the calibration (one step per tick)."""
    mode, st = ctx.cfg.slurm.resume, dict(store.get_state(K_RESUME) or {})
    if not hasattr(adapter, "interrupt"):
        return []
    adapter.resume_ok = mode == "on" or (mode == "calibrate" and st.get("status") == "passed")
    adapter.resume_runs = {st["run"]} if st.get("run") and mode != "off" else set()
    if mode != "calibrate" or st.get("status") in ("passed", "failed"):
        return []
    if not st:
        return _start(ctx, store, book, adapter)
    run, ref = store.get_run(st["run"]), store.get_run(st["ref"])
    if run is None or ref is None:
        return _verdict(store, st, False, "calibration rows missing")
    handle = JobHandle(run.id, run.job_id, run.output_dir, run.submitted_ts or 0.0)
    if run.status is RunStatus.RUNNING and not st.get("interrupted"):
        rs = adapter.resume_state(handle)
        if not (rs["armed"] and rs["fraction"] >= FRAC):
            return []
        try:
            job = adapter.interrupt(handle)
        except Exception as exc:  # noqa: BLE001 - tried again next tick
            store.log("resume_calibration_error", error=repr(exc)[:300])
            return []
        st["interrupted"] = rs["fraction"]
        store.set_state(K_RESUME, st)
        store.log("resume_calibration", status="interrupted", run=run.id, fraction=rs["fraction"], job_id=job)
        return [f"resume calibration: interrupted {run.id} at {rs['fraction']:.0%}; resuming as job {job}"]
    if run.status in (RunStatus.VOID, RunStatus.CRASH) and not S.is_retryable(run, book.infra_retry):
        return _verdict(store, st, False, f"calibration run {run.status.value}: {(run.log_tail or '')[-200:]}")
    if run.status is not RunStatus.DONE or run.aggregate is None:
        return []
    if not adapter.resume_state(handle)["resumed"]:
        return _verdict(store, st, False, "the calibration run never resumed (inconclusive)")
    st["delta"] = float(run.aggregate) - float(ref.aggregate)
    ok = abs(st["delta"]) <= ctx.cfg.slurm.resume_tol
    return _verdict(store, st, ok, f"|delta final_score| {abs(st['delta']):.5f} {'<=' if ok else '>'} "
                                   f"resume_tol {ctx.cfg.slurm.resume_tol}")


def _start(ctx: LoopContext, store: Any, book: Any, adapter: Any) -> list[str]:
    root, pre = store.get_node(ROOT_NODE_ID), next((p.name for p in ctx.cfg.pools if p.preemptible), "")
    skip = set(map(int, ctx.cfg.seeds.audit)) | {r.seed for r in store.runs(node_id=ROOT_NODE_ID, stage=Stage.AUDIT)}
    refs = sorted((r for r in store.runs(node_id=ROOT_NODE_ID, stage=Stage.BASELINE, status=[RunStatus.DONE])
                   if r.aggregate is not None and r.seed not in skip), key=lambda r: r.seed)
    if root is None or not refs or not pre or not book._open(True):
        return []
    run = Run(id=Run.make_id(ROOT_NODE_ID, Stage.AUDIT, refs[0].seed), task_id=ctx.task_id, node_id=ROOT_NODE_ID,
              seed=refs[0].seed, stage=Stage.AUDIT, status=RunStatus.SUBMITTING, submitted_ts=book.clock(), pool=pre)
    st = {"status": "running", "run": run.id, "ref": refs[0].id}
    store.set_state(K_RESUME, st)
    store.put_run(run)
    adapter.resume_runs = {run.id}
    store.log("resume_calibration", **st)
    run = book._submit(run, root)
    return [f"resume calibration: {run.id} -> {run.status.value} (vs {refs[0].id}; interrupted at {FRAC:.0%})"]


def _verdict(store: Any, st: dict, ok: bool, why: str) -> list[str]:
    st.update(status="passed" if ok else "failed", why=why)
    store.set_state(K_RESUME, st)
    store.log("resume_calibration", **st)
    return [f"resume calibration {st['status']}: {why}" + ("; resume-on-requeue ON" if ok else
                                                            "; runs keep restarting from step 0")]
