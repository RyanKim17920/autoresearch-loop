"""v3-c5 fixes A4 (a TEST block's dead pairs: infra re-drawn, member crashes capped), R4 (ideation renders briefs
off the tick thread) and A3 (brief kinds weighted by their measured yield this epoch)."""

from __future__ import annotations

import dataclasses

from loop import tick as TK
from loop.contracts import RunStatus, VoidType

from .test_tick_core import _module_tick, _Verdicts


# ------------------------------------------------------------------------------------------------ A4
def _open_block(ctx, tmp_path, cap=2):
    tick, store, t = _module_tick(ctx, tmp_path, _Verdicts(["open"]))
    tick.ctx = dataclasses.replace(tick.ctx, cfg=dataclasses.replace(
        tick.ctx.cfg, test=dataclasses.replace(tick.ctx.cfg.test, max_candidate_crashes=cap)))
    tick.step()
    [b] = [b for b in store.seed_blocks() if b.id.startswith("test-")]
    return tick, store, t, b


def _block(store, bid):
    return next(x for x in store.seed_blocks() if x.id == bid)


PROBE_TAIL = "subprocess.CalledProcessError: Command '['python', '/wt/n1/probe.py', 'x.json']' returned non-zero"


def test_infra_deaths_are_redrawn_without_limit(ctx, tmp_path):
    tick, store, t, b = _open_block(ctx, tmp_path)
    cand = b.node_ids[0]
    for i in range(3):                                  # three infra deaths in a row: always a new seed
        cur = _block(store, b.id)
        vt = VoidType.CANCELLED if i == 1 else VoidType.INFRA
        store.update_run(f"{cand}:confirm:{cur.seeds[0]}", status=RunStatus.VOID, void_type=vt, attempt=99)
        t[0] += 60
        tick.step()
        after = _block(store, b.id)
        assert after.open and cur.seeds[0] not in after.seeds and len(after.seeds) == 5
    assert not store.events("test_verdict")
    tick.close()
    store.close()


def test_candidate_crashes_close_the_block_fail_unscorable_at_the_cap(ctx, tmp_path):
    tick, store, t, b = _open_block(ctx, tmp_path, cap=2)
    cand, ctrl = b.node_ids
    rid = f"{cand}:confirm:{b.seeds[0]}"
    store.update_run(rid, status=RunStatus.CRASH, log_tail=PROBE_TAIL)
    t[0] += 60
    tick.step()
    mid = _block(store, b.id)
    assert mid.open and b.seeds[0] in mid.seeds                       # v3-c5b S3: a probe crash is retried once
    assert store.get_run(rid).attempt == 1 and store.get_run(rid).status is not RunStatus.CRASH
    assert [e["run_id"] for e in store.events("test_probe_retry")] == [rid]
    store.update_run(rid, status=RunStatus.CRASH, log_tail=PROBE_TAIL)  # the same-seed retry crashes too
    t[0] += 60
    tick.step()
    mid = _block(store, b.id)
    assert mid.open and b.seeds[0] not in mid.seeds                   # the first counted crash: re-drawn
    store.update_run(f"{cand}:confirm:{mid.seeds[-1]}", status=RunStatus.VOID, void_type=VoidType.EVAL)
    t[0] += 60
    rep = tick.step()
    after = _block(store, b.id)
    assert not after.open and after.closed_ts == t[0]
    [rec] = after.verdicts
    assert rec["node"] == cand and rec["vs"] == ctrl and rec["kind"] == "fail" and "unscorable" in rec["reason"]
    assert "crash:probe" in rec["reason"] and "void:eval" in rec["reason"]
    [ev] = [e for e in store.events("test_verdict") if e.get("unscorable")]
    assert ev["node_id"] == cand and len(ev["crashes"]) == 2
    assert any("FAIL (unscorable" in x for x in rep.lines)
    # the candidate's running runs of the block are cancelled; the control's are left
    left = [r for r in store.runs(node_id=cand, status=[RunStatus.RUNNING]) if r.seed in after.seeds]
    assert not left
    assert store.runs(node_id=ctrl, status=[RunStatus.RUNNING])
    tick.close()
    store.close()


def test_control_crashes_close_the_block_with_no_verdict(ctx, tmp_path):
    tick, store, t, b = _open_block(ctx, tmp_path, cap=2)
    cand, ctrl = b.node_ids
    store.update_run(f"{ctrl}:remint:{b.seeds[0]}", status=RunStatus.CRASH, log_tail="torch.OutOfMemoryError: CUDA out of memory")
    store.update_run(f"{ctrl}:remint:{b.seeds[1]}", status=RunStatus.CRASH, log_tail="ValueError: boom")
    t[0] += 60
    rep = tick.step()
    after = _block(store, b.id)
    assert not after.open and after.verdicts == ()                   # no verdict: the candidate is re-testable
    [ev] = store.events("test_unscorable_control")
    assert ev["control"] == ctrl and ev["candidates"] == [cand]
    assert "crash:oom" in ev["crash_class"] and "crash:ValueError" in ev["crash_class"]
    assert not store.events("test_verdict")
    assert any("unscorable (control)" in x for x in rep.lines)
    tick.close()
    store.close()


def test_crash_class():
    R = TK.Run
    mk = lambda **k: R(id="a", task_id="t", node_id="n", seed=1, stage=TK.TEST_STAGE, **k)   # noqa: E731
    assert TK.crash_class(mk(status=RunStatus.CRASH, log_tail=PROBE_TAIL)) == "crash:probe"
    assert TK.crash_class(mk(status=RunStatus.CRASH, log_tail="x\nKeyError: 'a'\n")) == "crash:KeyError"
    assert TK.crash_class(mk(status=RunStatus.VOID, void_type=VoidType.TOO_SLOW)) == "void:too_slow"
    assert TK.member_fault(mk(status=RunStatus.VOID, void_type=VoidType.TOO_SLOW))
    assert not TK.test_fault(mk(status=RunStatus.VOID, void_type=VoidType.TOO_SLOW))        # v3-c5b S3: infra
    assert TK.crash_class(mk(status=RunStatus.CRASH, log_tail="FileNotFoundError: /r/latest.pt")) == "crash:checkpoint"
    assert TK.crash_class(mk(status=RunStatus.CRASH, log_tail="python state/evalonly/resume_shim.py")) == "crash:shim"
    assert not TK.test_fault(mk(status=RunStatus.CRASH, log_tail="FileNotFoundError: /r/latest.pt"))
    assert not TK.test_fault(mk(status=RunStatus.CRASH, log_tail="x\nKeyError: 'a'\n"), eval_run=True)
    assert TK.test_fault(mk(status=RunStatus.CRASH, log_tail=PROBE_TAIL), eval_run=True)
    assert not TK.test_fault(mk(status=RunStatus.CRASH, log_tail=PROBE_TAIL), pending_retry=True)
    assert not TK.member_fault(mk(status=RunStatus.VOID, void_type=VoidType.INFRA, attempt=9))
    assert not TK.member_fault(mk(status=RunStatus.VOID, void_type=VoidType.CANCELLED))


def test_crashes_on_replaced_seeds_close_the_block_without_a_current_death(ctx, tmp_path):
    """v3-c5b M4 (live test-n2673): the member tally runs every tick over every seed the block ever drew -- 2 counted
    crashes on since-replaced seeds close it even when no current seed is dead."""
    tick, store, t, b = _open_block(ctx, tmp_path, cap=2)
    cand, _ = b.node_ids
    tick.store.set_state(TK.K_PROBE_RETRY, [])
    for s in b.seeds[:2]:                                # replaced under the old rule (no retry): legacy crashes
        store.update_run(f"{cand}:confirm:{s}", status=RunStatus.CRASH, log_tail="ValueError: boom")
    store.update_seed_block(b.id, seeds=(*b.seeds[2:], 9001, 9002))
    store.log("seeds_drawn", purpose=f"confirm {cand}", seeds=[9001, 9002], block=b.id, why="replace dead pairs")
    t[0] += 60
    tick.step()
    after = _block(store, b.id)
    assert not after.open and after.verdicts[0]["kind"] == "fail" and "unscorable" in after.verdicts[0]["reason"]
    tick.close()
    store.close()


def test_control_unscorable_backs_the_candidate_off(ctx, tmp_path):
    tick, store, t, b = _open_block(ctx, tmp_path, cap=2)
    cand, ctrl = b.node_ids
    store.update_run(f"{ctrl}:remint:{b.seeds[0]}", status=RunStatus.CRASH, log_tail="ValueError: a")
    store.update_run(f"{ctrl}:remint:{b.seeds[1]}", status=RunStatus.CRASH, log_tail="ValueError: b")
    t[0] += 60
    tick.step()
    [ev] = store.events("test_unscorable_control")
    assert ev["backoff_until"] == t[0] + 12 * 3600.0
    assert tick._test_backoff(t[0] + 3600) == {cand} and tick._test_backoff(t[0] + 13 * 3600) == frozenset()
    tick.close()
    store.close()
