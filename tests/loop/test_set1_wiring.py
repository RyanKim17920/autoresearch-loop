"""10-21 first-tick fixes: the real Tick wiring of the set-1 hooks (a fake Tick hid ``self.A``), the pending gate counting
runs launched since the poll (51 screens went to a full idle pool in one tick), stale-pending moving a run only once, and
the heartbeat's pool sanity ALERT."""

from __future__ import annotations

import dataclasses
import time
from types import SimpleNamespace

from loop import runner as R
from loop import tick as TK
from loop.contracts import NodeStatus, Run, RunStatus, Stage
from loop.core import data as CD
from loop.spec import PoolCfg

from .test_runner import _node, _pools
from .test_set1_harness import _Held, _stale_book


def _real_tick(ctx, store, tmp_path):
    return TK.Tick(ctx, store, _Held(), TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)


def test_the_real_tick_ages_out_old_queued_nodes_through_its_core(ctx, store, tmp_path):
    """Not a fake Tick: ``Tick._age_out_queued`` reaches acquire through ``Tick.core`` (it read ``self.A``: AttributeError)."""
    _node(store, "n1")
    _node(store, "n2")
    from loop.core.acquire import PROGRAM_TAG
    now = time.time() + 200 * 3600.0                          # n1 never ran in 200 h; n2 is an operator-program arm (exempt)
    store.update_node("n1", created_ts=1000.0)
    store.update_node("n2", created_ts=1000.0, notes=PROGRAM_TAG)
    tick = _real_tick(ctx, store, tmp_path)
    data = CD.load(tmp_path / "loop.db", ctx.task_id)
    statuses = {nid: tick.core.ST.Status.QUEUED for nid in data.nodes}
    out = tick._age_out_queued(data, statuses, now)
    assert len(out) == 1 and "n1" in out[0]
    assert store.get_node("n1").status is NodeStatus.REJECTED and store.get_node("n2").status is NodeStatus.QUEUED
    assert tick._age_out_queued(data, statuses, now) == []          # idempotent
    tick.close()


def test_a_tick_without_an_acquire_module_ages_nothing(ctx, store, tmp_path):
    tick = TK.Tick(ctx, store, _Held(), SimpleNamespace(), db_path=tmp_path / "loop.db", clock=time.time)
    assert tick._age_out_queued(None, {}, time.time()) == []
    tick.close()


def test_the_real_tick_heartbeat_runs_with_the_new_pool_fields(ctx, store, tmp_path):
    tick = _real_tick(ctx, store, tmp_path)
    assert " pools=" in tick.heartbeat(1, 0.5)
    tick.close()


# ---------------------------------------------------------------- the burst past the pending gate
def test_runs_launched_since_the_poll_count_as_pending_against_the_gate(ctx, store):
    book = R.RunBook(_pools(ctx, primary=0, idle=56, max_pending_per_pool=2), store, _Held(), clock=lambda: 1000.0)
    node = _node(store, "n1")
    book.poll()
    book.pool_states = {"primary": (0, 0), "idle": (2, 1)}          # live 10-21 22:27: 2 running, 1 pending at the poll
    n = 0
    while n < 80 and (pool := book.pool(node, Stage.SCREEN, 100_000 + n)):
        book.launch(node, Stage.SCREEN, 100_000 + n)
        n += 1
    assert n <= 4, f"{n} launches in one tick past a gate of 2 pending"          # was 56 - 3 = 53
    assert book.limits()["idle"] - book.used()["idle"] <= 0


def test_the_gate_still_lets_a_pool_that_keeps_up_fill_in_one_tick(ctx, store):
    book = R.RunBook(_pools(ctx, primary=0, idle=10, max_pending_per_pool=2), store, _Held(), clock=lambda: 1000.0)
    node = _node(store, "n1")
    book.pool_states = {"primary": (0, 0), "idle": (0, 0)}
    store.set_state(R.K_STARVED, {})
    n = 0
    while n < 30 and book.pool(node, Stage.SCREEN, 100_000 + n):
        book.launch(node, Stage.SCREEN, 100_000 + n)
        n += 1
    assert 1 <= n <= 10                                                 # bounded by the pool, and by the gate once pending


# ---------------------------------------------------------------- stale pending: one move per run, no ping-pong
def test_stale_pending_moves_a_run_once_then_cancels_it_back_into_the_queue(ctx, store):
    book, adapter, t = _stale_book(ctx, store, primary=2, idle=2, stale_pending_min=90)
    node = _node(store, "n1")
    run = book.launch(node, Stage.BASELINE, 100_000)               # HIGH: primary
    assert run.pool == "primary"
    book.poll()
    t[0] += 60 * 100
    assert len(book.stale_pending()) == 1 and store.get_run(run.id).pool == "idle"
    assert run.id in set(store.get_state(R.K_STALE_MOVED))
    book.poll()
    t[0] += 60 * 100                                               # stale again, now at idle, primary has a free slot
    out = book.stale_pending()
    assert adapter.moved == [(run.id, "idle")]                     # NOT moved back to primary
    assert "scancelled, back in the queue" in out[0] and store.get_run(run.id).status is RunStatus.VOID


# ---------------------------------------------------------------- heartbeat sanity guard
def test_the_heartbeat_alerts_once_when_slots_held_pass_twice_the_slots_in_force(ctx, store, tmp_path, capsys):
    tick = _real_tick(ctx, store, tmp_path)
    tick.book.active = lambda: [object()] * 58
    tick.book.limits = lambda gate=True: {"primary": 0, "idle": 5}
    tick._pool_sanity()
    tick._pool_sanity()
    err = capsys.readouterr().err
    assert err.count("ALERT: pool runs held 58 > 2x the 5") == 1
    tick.book.active = lambda: [object()] * 7
    tick._pool_sanity()
    tick.book.active = lambda: [object()] * 58
    tick._pool_sanity()                                                 # re-armed after a healthy reading
    assert capsys.readouterr().err.count("ALERT") == 1
    tick.close()


def test_the_pool_alert_ignores_smoke_slots_and_the_smoke_gate(ctx, store, tmp_path, capsys):
    """11-09 00:16Z false alarm: ``primary:7r/0p 11/8, idle:0r/9p 12/3`` -- 16 training runs, the rest smoke jobs /
    the smoke reserve, against limits the smoke-pending gate had cut: no burst past the cap, so no ALERT."""
    tick = _real_tick(ctx, store, tmp_path)
    tick.book.used = lambda: {"primary": 11, "idle": 12}                # runs + 7 smoke slots
    tick.book.active = lambda: [object()] * 16
    tick.book.limits = lambda gate=True: {"primary": 8, "idle": 3} if not gate else {"primary": 7, "idle": 0}
    tick._pool_sanity()
    assert "ALERT" not in capsys.readouterr().err
    tick.close()


# ---------------------------------------------------------------- 10-22: a quarantine does not outlive the code that earned it
def _boom() -> None:
    raise AttributeError("'Tick' object has no attribute 'A'")


def test_a_quarantine_records_the_code_revision_and_a_new_revision_releases_it(ctx, store, tmp_path, monkeypatch):
    """Live 10-21 23:06: ``queue.age_out`` was quarantined by a bug a deploy had already fixed, and stayed disabled
    (the entry left only by hand). A restart on a different revision releases it; the same revision does not."""
    monkeypatch.setattr(TK, "code_rev", lambda: "aaaaaaaaaaaa")
    tick = _real_tick(ctx, store, tmp_path)
    acts: list[str] = []
    for _ in range(TK.QUARANTINE_AFTER):
        assert tick.guarded("queue.age_out", "age out", _boom, acts, default=[]) == []
    assert tick.quarantined("queue.age_out") and store.get_state(TK.K_QUARANTINE)["queue.age_out"]["rev"] == "aaaaaaaaaaaa"
    assert tick.guarded("queue.age_out", "age out", lambda: ["ran"], acts, default=[]) == []     # still skipped
    assert tick.release_stale_quarantine(acts, rev="aaaaaaaaaaaa") == []                          # same code: stays
    assert tick.quarantined("queue.age_out")
    assert tick.release_stale_quarantine(acts, rev="bbbbbbbbbbbb") == ["queue.age_out"]           # new code: released
    assert not tick.quarantined("queue.age_out") and "queue.age_out" not in (store.get_state(TK.K_ERRORS) or {})
    assert store.events("quarantine_released") and any("quarantine released" in a for a in acts)
    assert tick.guarded("queue.age_out", "age out", lambda: ["ran"], acts, default=[]) == ["ran"]
    tick.close()


def test_a_revless_legacy_quarantine_is_stale_and_released_on_the_first_readable_start(ctx, store, tmp_path, monkeypatch):
    """10-22: an entry written before the ``rev`` field persisted across every deploy; it is released at the first start
    that can read its own revision (and nothing is released when it cannot)."""
    monkeypatch.setattr(TK, "code_rev", lambda: "bbbbbbbbbbbb")
    store.set_state(TK.K_QUARANTINE, {"old.site": {"reason": "x", "ts": 1.0}, "same.site": {"reason": "y", "ts": 1.0,
                                                                                              "rev": "bbbbbbbbbbbb"}})
    store.ensure_root("c0")
    tick = _real_tick(ctx, store, tmp_path)
    assert tick.release_stale_quarantine([], rev="") == [] and tick.quarantined("old.site")      # no git: operator only
    acts: list[str] = []
    tick._start(acts, time.time())                                    # the real first start of the process
    assert not tick.quarantined("old.site") and tick.quarantined("same.site")
    assert any("quarantine released" in a and "old.site" in a for a in acts)
    tick.close()


def test_an_unreadable_revision_quarantine_is_never_auto_released(ctx, store, tmp_path):
    tick = _real_tick(ctx, store, tmp_path)
    store.set_state(TK.K_QUARANTINE, {"new.site": {"reason": "x", "ts": 1.0, "rev": "aaaaaaaaaaaa"}})
    assert tick.release_stale_quarantine([], rev="") == [] and tick.quarantined("new.site")     # no git here
    tick.close()


def test_a_fresh_process_releases_a_stale_quarantine_on_its_first_step(ctx, store, tmp_path, monkeypatch):
    monkeypatch.setattr(TK, "code_rev", lambda: "bbbbbbbbbbbb")
    store.set_state(TK.K_QUARANTINE, {"queue.age_out": {"reason": "x", "ts": 1.0, "rev": "aaaaaaaaaaaa"}})
    store.ensure_root("c0")
    tick = _real_tick(ctx, store, tmp_path)
    acts: list[str] = []
    tick._start(acts, time.time())
    assert not tick.quarantined("queue.age_out") and any("quarantine released" in a for a in acts)
    tick.close()


def test_code_rev_is_a_short_hash_or_empty():
    assert TK.code_rev() == "" or (len(TK.code_rev()) == 12 and TK.code_rev().isalnum())


# ---------------------------------------------------------------- 10-22: smokes go first by GATING, never by Nice
def test_no_new_submit_to_a_pool_while_one_of_our_smokes_is_pending_there(ctx, store):
    book = R.RunBook(_pools(ctx, primary=0, idle=10, max_pending_per_pool=2), store, _Held(), clock=lambda: 1000.0)
    node = _node(store, "n1")
    book.pool_states = {"primary": (0, 0), "idle": (0, 0)}
    store.set_state(R.K_STARVED, {})
    assert book.pool(node, Stage.SCREEN, 100_000) == "idle"
    book.smoke_pending = {"idle": 1}
    assert book.limits()["idle"] - book.used()["idle"] <= 0 and not book.pool(node, Stage.SCREEN, 100_001)
    assert book.limits(gate=False)["idle"] == 10                   # only the gated view holds (headroom sizing unchanged)
    book.smoke_pending = {}
    assert book.pool(node, Stage.SCREEN, 100_002) == "idle"


def test_the_real_tick_poll_reads_the_adapters_pending_smokes_and_holds_the_pool(ctx, store, tmp_path):
    adapter = _Held()
    adapter.smokes_pending = lambda: {"idle": 2}
    tick = TK.Tick(_pools(ctx, primary=0, idle=10), store, adapter, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)
    tick.book.poll()
    assert tick.book.smoke_pending == {"idle": 2}
    assert tick.book.limits()["idle"] == tick.book.used()["idle"]
    adapter.smokes_pending = lambda: (_ for _ in ()).throw(OSError("squeue down"))
    tick.book.poll()
    assert tick.book.smoke_pending == {} and tick.book.limits()["idle"] > tick.book.used()["idle"]   # a monitoring error never blocks
    tick.close()


# ---------------------------------------------------------------- starved zero-run nodes drop under backlog_keep
def test_the_real_tick_drops_starved_zero_run_nodes_under_backlog_keep(ctx, store, tmp_path, caplog):
    """Live 10-22: 10 never-run nodes 17-19 h old held the median queue age over the throttle; the queue (30) was under
    backlog_keep (40), so alloc.drop never fired. Past drop_age_h a zero-run node drops with its own reason."""
    from loop.core import alloc as AL
    from loop.core.acquire import PROGRAM_TAG, restore_aged
    for nid in ("old", "young", "prog", "ported"):
        _node(store, nid)
    now = time.time()
    store.update_node("old", created_ts=now - 30 * 3600.0)
    store.update_node("young", created_ts=now - 10 * 3600.0)
    store.update_node("prog", created_ts=now - 30 * 3600.0, notes=PROGRAM_TAG)
    store.update_node("ported", created_ts=now - 30 * 3600.0, notes="port_of=elsewhere")
    tick = _real_tick(ctx, store, tmp_path)
    data = CD.load(tmp_path / "loop.db", ctx.task_id)
    statuses = {nid: tick.core.ST.Status.QUEUED for nid in data.nodes}
    assert AL.cfg_of(ctx).enabled and len(data.nodes) < AL.cfg_of(ctx).backlog_keep
    with caplog.at_level("INFO"):
        out = tick._age_out(data, statuses, [], now)
    assert len(out) == 1 and "old" in out[0] and "starved_zero_run" in out[0]
    assert any("starved_zero_run" in r.getMessage() and "old" in r.getMessage() for r in caplog.records)
    for nid in ("young", "prog", "ported"):
        assert store.get_node(nid).status is NodeStatus.QUEUED
    node = store.get_node("old")
    assert node.status is NodeStatus.REJECTED and node.reject_reason.startswith(AL.STARVED_REASON)
    assert node.reject_reason.startswith("dropped_backlog") and "aged_out_queued" not in node.reject_reason
    assert restore_aged(store, ["old"]) == ["old"] and store.get_node("old").status is NodeStatus.QUEUED
    # restored: a fresh window (updated_ts), so it is not dropped again at once
    data = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert tick._age_out(data, {nid: tick.core.ST.Status.QUEUED for nid in data.nodes}, [], now + 3600.0) == []
    tick.close()


def test_starved_never_drops_an_open_block_node_or_a_node_with_a_run(ctx, store, tmp_path):
    from loop.core import alloc as AL
    for nid in ("a", "b"):
        _node(store, nid)
        store.update_node(nid, created_ts=1000.0)
    tick = _real_tick(ctx, store, tmp_path)
    data = CD.load(tmp_path / "loop.db", ctx.task_id)
    now = 1000.0 + 40 * 3600.0
    bl = AL.backlog(data, {"a": tick.core.ST.Status.QUEUED, "b": tick.core.ST.Status.SCREENING}, now)
    ac = AL.cfg_of(ctx)
    assert [r[0] for r in AL.starved(ac, bl, data.nodes, (), None, now)] == ["a"]     # b is not QUEUED (it has runs)
    assert AL.starved(ac, bl, data.nodes, {"a"}, None, now) == []                        # open block / held: exempt
    tick.close()


# ---------------------------------------------------------------- 10-22 10:39Z: 50 screens to an empty pool, ~3 GPUs free
def _part_pools(ctx, idle=56, **slurm):
    c = _pools(ctx, primary=0, idle=idle, **slurm)
    pools = tuple(dataclasses.replace(p, partition="gpu") for p in c.cfg.slurm.pools)
    return dataclasses.replace(c, cfg=dataclasses.replace(c.cfg, slurm=dataclasses.replace(c.cfg.slurm, pools=pools)))


class _Free(_Held):
    def __init__(self, free):
        super().__init__()
        self.free, self.probes = free, 0

    def free_gpus(self):
        self.probes += 1
        if isinstance(self.free, Exception):
            raise self.free
        return {"gpu": self.free}


def _fill(book, node, base=100_000, limit=120):
    n = 0
    while n < limit and book.pool(node, Stage.SCREEN, base + n):
        book.launch(node, Stage.SCREEN, base + n)
        n += 1
    return n


def _tick_book(ctx, store, tmp_path, adapter, **slurm):
    tick = TK.Tick(_part_pools(ctx, **slurm), store, adapter, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)
    tick.book.poll()
    store.set_state(R.K_STARVED, {})
    return tick


def _seed_running(tick, store, k, probe_free):
    """k runs of ours on idle (SLURM reports them running), then the free-GPU reading the NEXT tick will see."""
    node = _node(store, "n0")
    tick.book.adapter.free = 10_000
    for i in range(k):
        tick.book.launch(node, Stage.SCREEN, 900_000 + i)
    tick.book.adapter.free = probe_free
    tick.book._free = None


def test_real_tick_empty_idle_pool_with_three_free_gpus_takes_three_not_fifty(ctx, store, tmp_path):
    adapter = _Free(3)
    tick = _tick_book(ctx, store, tmp_path, adapter)
    _seed_running(tick, store, 6, 3)
    tick.book.pool_states = {"primary": (0, 0), "idle": (6, 0)}      # 10:39Z: idle 6r/0p, 56 declared, 3 free GPUs
    before = adapter.probes
    n = _fill(tick.book, _node(store, "n1"))
    assert 1 <= n <= 4, f"{n} launches in one tick for 3 free GPUs"
    assert adapter.probes == before + 1                              # one sinfo per tick, cached
    tick.book.poll()                                                 # next tick: the probe is read again (lazily)
    tick.book.limits()
    assert adapter.probes == before + 2
    tick.close()


def test_real_tick_no_gpu_free_still_allows_the_floor_of_two(ctx, store, tmp_path):
    tick = _tick_book(ctx, store, tmp_path, _Free(0))
    n = _fill(tick.book, _node(store, "n1"))
    assert n == 2
    tick.close()


def test_real_tick_probe_failure_falls_back_to_max_new_per_tick(ctx, store, tmp_path):
    tick = _tick_book(ctx, store, tmp_path, _Free(RuntimeError("sinfo down")), max_new_per_tick=4)
    n = _fill(tick.book, _node(store, "n1"))
    assert n == 4
    assert len(store.events("free_gpu_probe_failed")) == 1
    tick.book.poll()
    _fill(tick.book, _node(store, "n2"), base=200_000)
    assert len(store.events("free_gpu_probe_failed")) == 1          # logged once per outage
    tick.close()


def test_real_tick_a_pool_with_real_free_gpus_still_fills_fast(ctx, store, tmp_path):
    tick = _tick_book(ctx, store, tmp_path, _Free(30))
    n = _fill(tick.book, _node(store, "n1"))
    assert n == 30
    tick.close()


def test_pending_jobs_count_toward_the_cap_whatever_the_pool_held_before(ctx, store, tmp_path):
    """43 already pending (the 10:39Z leftovers): nothing new, even though the partition shows free GPUs."""
    tick = _tick_book(ctx, store, tmp_path, _Free(30))
    _seed_running(tick, store, 54, 30)
    tick.book.pool_states = {"primary": (0, 0), "idle": (11, 43)}
    assert tick.book.limits()["idle"] - tick.book.used()["idle"] <= 0
    assert not tick.book.pool(_node(store, "n1"), Stage.SCREEN, 100_000)
    tick.close()


def test_the_new_submit_cap_keeps_the_smoke_first_rule(ctx, store, tmp_path):
    adapter = _Free(30)
    adapter.smokes_pending = lambda: {"idle": 1}
    tick = _tick_book(ctx, store, tmp_path, adapter)
    assert tick.book.smoke_pending == {"idle": 1}
    assert _fill(tick.book, _node(store, "n1")) == 0
    tick.close()


def test_no_free_gpu_probe_on_the_adapter_leaves_the_old_gate(ctx, store, tmp_path):
    tick = TK.Tick(_part_pools(ctx, idle=10), store, _Held(), TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)
    tick.book.poll()
    assert tick.book.free_gpus("idle") is None
    tick.close()


def test_the_new_cap_leaves_the_test_spill_to_an_idle_pool_working(ctx, store, tmp_path):
    adapter = _Free(5)
    c = _part_pools(ctx, idle=10)
    pools = (dataclasses.replace(c.cfg.slurm.pools[0], slots=6), c.cfg.slurm.pools[1])
    c = dataclasses.replace(c, cfg=dataclasses.replace(c.cfg, slurm=dataclasses.replace(c.cfg.slurm, pools=pools)))
    tick = TK.Tick(c, store, adapter, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)
    adapter.pool_capacity = lambda: {"primary": {"limit": 8, "others": 8, "loop": 0, "effective": 0}}   # primary cut by sweeps
    tick.book.poll()
    store.set_state(R.K_STARVED, {})
    run = tick.book.launch(_node(store, "n1"), Stage.CONFIRM, 100_900)
    assert run.pool == "idle" and run.id in tick.book.spilled()      # the TEST spill (21cc141) still works, under the probe
    tick.close()


# ---------------------------------------------------------------- the below-bar line, hourly
def test_the_real_tick_says_the_incumbent_is_below_the_bar_at_most_hourly(ctx, store, tmp_path):
    """10-22: ``labless: incumbent n1248@065b87f not submitted, does not beat bar...`` was logged on every new TEST seed
    (a few minutes apart, 7 times in 3 h). Through the real Tick's ``labless``: said once, silent while the hour
    holds even as the TEST-seed count moves, said again after it, and silent again for an unchanged count."""
    from .core.test_bar import ledger as bar_ledger
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, labless=dataclasses.replace(ctx.cfg.labless, enabled=True, bar_node="bar", leader_median=0.6810443)))
    t = [10_000.0]
    tick = TK.Tick(ctx, store, _Held(), TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=lambda: t[0])
    _node(store, "big")
    say = lambda n: tick.labless.step("big", bar_ledger({"big": (0.679, 5, n)}))      # noqa: E731
    assert len(say(5)) == 1                                        # first time: said
    t[0] += 120.0
    assert say(6) == []                                            # a new TEST seed 2 min later: silent
    t[0] += 600.0
    assert say(7) == []
    t[0] += 3600.0
    assert len(say(8)) == 1                                        # past the hour: said once more
    t[0] += 60.0
    assert say(8) == [] and say(9) == []
    assert len([e for e in store.events() if e["event"] == "labless_below_bar"]) == 2
    tick.close()
