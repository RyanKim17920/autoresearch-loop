"""loop.tick wired to the real core (``ModuleCore``) on the ledger fixture: the historical seed blocks are migrated
once; seed actions launch SCREEN runs on SELECT seeds, test actions open a TEST block (never-used seeds) and launch
CONFIRM (candidate) + REMINT (incumbent) on it; the open tests' verdicts close / re-mint / extend their blocks."""

from __future__ import annotations

import dataclasses

from loop import tick as TK
from loop.contracts import ROOT_NODE_ID, Stage
from loop.core import data as CD
from loop.core import seeds as S
from loop.store import SqliteStore

from .core.conftest import TASK
from .test_tick import _ledger


def _module_tick(ctx, tmp_path, core):
    db = _ledger(tmp_path / "live.db")
    ctx = dataclasses.replace(ctx, task_id=TASK, cfg=dataclasses.replace(ctx.cfg, slots=64))
    store = SqliteStore(db, TASK, weights=ctx.cfg.weights)
    t = [2e9]
    tick = TK.Tick(ctx, store, TK.DryRunAdapter(store, lambda: t[0]), core, db_path=db, clock=lambda: t[0], dry=True)
    return tick, store, t


class _WithTests(TK.ModuleCore):
    """The real core's ranking with its tests replaced by a test of the ``n_tests`` best test beliefs (the fixture's
    beliefs are all weak: without a λ every positive-value test would launch)."""

    def __init__(self, n_tests=2):
        super().__init__()
        self.n_tests = n_tests
        self.floors = {"floor": 0.0}                   # the fixture's seeds sit under the value floor: launch them

    def rank(self, data, blocks, statuses, post, free_slots, llm_free):
        acts = super().rank(data, blocks, statuses, post, free_slots, llm_free)
        entry, self.A.test_entry_ok = self.A.test_entry_ok, lambda *a: True       # plumbing: any candidate
        try:
            opts = self.A.ledger_options(data, blocks, statuses, TK.incumbent_of(statuses), self.costs)
        finally:
            self.A.test_entry_ok = entry
        best = sorted((o for o in opts if o.kind == "test"), key=lambda o: -o.m)[:self.n_tests]
        return [TK.Action("test", o.node, o.lane, o.gpu_h, 1.0, o.runs) for o in best] + \
            [a for a in acts if a.kind != "test" and a.node not in {o.node for o in best}]


def test_module_core_tick_migrates_blocks_and_launches_seed_and_test_runs_on_their_pools(ctx, tmp_path):
    tick, store, _ = _module_tick(ctx, tmp_path, _WithTests())
    rep = tick.step()
    assert any("historical blocks migrated" in x for x in rep.lines) and store.seed_blocks()
    inc = TK.incumbent_of(rep.statuses)
    assert inc != ROOT_NODE_ID and {a.kind for a in rep.actions} >= {"seed", "idea"}
    assert any(x.startswith("would brief ") for x in rep.lines)                          # dry: no session
    runs = {r.id: r for r in store.runs() if r.id in rep.launched}
    blocks = store.seed_blocks()
    tests = [b for b in blocks if b.id.startswith("test-")]
    assert len(tests) == 2 and all(b.pool == CD.TEST and b.open for b in tests)
    for b in tests:                                    # CONFIRM (candidate) + REMINT (incumbent) on never-used seeds
        assert b.node_ids == (b.purpose.split()[1], inc) and len(b.seeds) == 5 and min(b.seeds) >= 100_000
        assert all(f"{b.node_ids[0]}:confirm:{s}" in runs and f"{inc}:remint:{s}" in runs for s in b.seeds)
    screens = [r for r in runs.values() if r.stage is Stage.SCREEN]
    assert screens and all(S.pool_of(r.seed, blocks) == CD.SELECT for r in screens)
    for r in runs.values():
        S.check_submit(r.node_id, r.seed, blocks)
    core = TK.ModuleCore()
    led = CD.load(tick.db_path, TASK)
    st = core.derive(led, blocks, core.noise(led))
    assert all(st[b.node_ids[0]].value == "testing" for b in tests)
    rep2 = tick.step()                                 # nothing new for the open tests: their runs exist
    assert not [x for x in rep2.launched if x.split(":")[0] in {b.node_ids[0] for b in tests}]
    assert {a.node for a in rep2.actions if a.kind == "test"}.isdisjoint({b.node_ids[0] for b in tests})
    tick.close()
    store.close()


class _Verdicts(_WithTests):
    """The real core, ranking only tests (once), with stubbed verdicts for the blocks it opened."""

    def __init__(self, kinds):
        super().__init__(len(kinds))
        self.kinds, self.ranked = kinds, False

    def rank(self, data, blocks, statuses, post, free_slots, llm_free):
        acts, self.ranked = ([] if self.ranked else super().rank(data, blocks, statuses, post, free_slots,
                                                                  llm_free)), True
        return [a for a in acts if a.kind == "test"]

    def decide(self, data, blocks, noise, incumbent):
        from loop.core.test import Kind, Verdict
        opened = sorted((b for b in blocks if b.pool == CD.TEST and b.open and b.id.startswith("test-")),
                        key=lambda b: b.id)
        return [(b, b.node_ids[0], Verdict(Kind(k), 12.0, 5, 0.003, "stub", (), 3 if k == "extend" else 0))
                for b, k in zip(opened, self.kinds)]


def test_tick_closes_passed_and_failed_tests_and_extends_promising_ones(ctx, tmp_path):
    tick, store, t = _module_tick(ctx, tmp_path, _Verdicts(["pass", "fail", "extend"]))
    tick.step()
    opened = sorted((b for b in store.seed_blocks() if b.id.startswith("test-")), key=lambda b: b.id)
    assert len(opened) == 3
    t[0] += 60
    rep = tick.step()
    after = {b.id: b for b in store.seed_blocks()}
    passed, failed, extended = (after[b.id] for b in opened)
    assert passed.closed_ts == t[0] and failed.closed_ts == t[0] and extended.open
    [remint] = [b for b in after.values() if b.id.startswith("remint-")]
    assert remint.pool == CD.SELECT and remint.seeds == passed.seeds and remint.node_ids == passed.node_ids[:1]
    assert len(extended.seeds) == 8 and extended.seeds[:5] == opened[2].seeds
    assert all(f"{extended.node_ids[0]}:confirm:{s}" in rep.launched for s in extended.seeds[5:])
    # S2 (10-13): the extension's seeds are TEST seeds for core.data (a seeds_drawn event of TEST purpose)
    assert set(extended.seeds) <= CD.load(tick.db_path, TASK).test_seeds
    assert any(e.get("block") == extended.id and e.get("why") == "extend" and e["seeds"] == list(extended.seeds[5:])
               for e in store.events("seeds_drawn"))
    assert [e["verdict"] for e in store.events("test_verdict")] == ["pass", "fail"]
    assert [r["kind"] for r in passed.verdicts] == ["pass"] and [r["kind"] for r in failed.verdicts] == ["fail"]
    assert passed.verdicts[0]["vs"] == passed.node_ids[-1] and extended.verdicts is None
    assert all(b.verdicts is not None for b in after.values() if b.pool == CD.TEST and not b.open)   # history frozen
    t[0] += 60
    rep = tick.step()                                  # the stub says EXTEND again: an extended block never grows
    assert len({b.id: b for b in store.seed_blocks()}[extended.id].seeds) == 8
    tick.close()
    store.close()


def test_a_dead_pair_gets_a_replacement_seed(ctx, tmp_path):
    from loop.contracts import RunStatus
    tick, store, t = _module_tick(ctx, tmp_path, _Verdicts(["open"]))
    tick.step()
    [b] = [b for b in store.seed_blocks() if b.id.startswith("test-")]
    store.update_run(f"{b.node_ids[0]}:confirm:{b.seeds[1]}", status=RunStatus.CRASH)
    t[0] += 60
    rep = tick.step()
    [after] = [x for x in store.seed_blocks() if x.id == b.id]
    assert b.seeds[1] not in after.seeds and len(after.seeds) == 5 and after.open
    [new] = set(after.seeds) - set(b.seeds)
    assert {f"{b.node_ids[0]}:confirm:{new}", f"{b.node_ids[1]}:remint:{new}"} <= set(rep.launched)
    assert new in CD.load(tick.db_path, TASK).test_seeds                      # the replacement is a TEST seed too
    tick.close()
    store.close()


def test_historical_blocks_with_no_runs_are_closed_as_void_with_a_logged_reason(ctx, tmp_path):
    """The fixture's extract keeps some seeds_drawn events without their runs: those blocks were reconstructed OPEN
    with no members. They are closed as VOID at their draw and logged, never left open (their seeds stay owned)."""
    tick, store, _ = _module_tick(ctx, tmp_path, TK.StubCore())
    led = CD.load(tick.db_path, TASK)
    void = [b for b in S.reconstruct(led) if not any(r.seed in b.seeds for r in led.runs)]
    assert void and all(not b.open and b.closed_ts == b.drawn_ts and b.purpose.endswith(f"({S.VOID_NO_RUNS})")
                        for b in void)
    for b in void:                                     # a ledger migrated before the fix: reopen them
        store.put_seed_block(dataclasses.replace(b, closed_ts=None, purpose=b.purpose.split(" (")[0]))
    tick._migrated = 0
    rep = tick.step()
    after = {b.id: b for b in store.seed_blocks()}
    assert all(not after[b.id].open for b in void)
    assert sum("VOID" in x for x in rep.lines) == len(void)
    assert {e["block"] for e in store.events("seed_block_void")} == {b.id for b in void}
    with __import__("pytest").raises(S.SeedPoolViolation):              # a void TEST block's seeds stay unusable
        S.check_submit("n0655", next(b for b in void if b.pool == CD.TEST).seeds[0], list(after.values()))
    tick.close()
    store.close()


class _Means(_Verdicts):
    def __init__(self, means):
        super().__init__(["pass"] * len(means))
        self.means = means

    def decide(self, data, blocks, noise, incumbent):
        from loop.core.test import Kind, Verdict
        opened = sorted((b for b in blocks if b.pool == CD.TEST and b.open and b.id.startswith("test-")),
                        key=lambda b: b.id)
        return [(b, b.node_ids[0], Verdict(Kind.PASS, 30.0, 5, m, "stub", (), 0, 0.002))
                for b, m in zip(opened, self.means)]


def test_two_passes_in_one_tick_remint_only_the_best(ctx, tmp_path):
    tick, store, t = _module_tick(ctx, tmp_path, _Means([0.002, 0.004]))
    tick.step()
    opened = sorted((b for b in store.seed_blocks() if b.id.startswith("test-")), key=lambda b: b.id)
    t[0] += 60
    rep = tick.step()
    [remint] = [b for b in store.seed_blocks() if b.id.startswith("remint-")]
    assert remint.node_ids == (opened[1].node_ids[0],)
    assert any("PASSED without a switch" in x and opened[0].node_ids[0] in x for x in rep.lines)
    assert TK.incumbent_of(rep.statuses) == opened[1].node_ids[0]
    tick.close()
    store.close()


class _Switched(_Verdicts):
    """``_Verdicts`` whose derive reports ``new`` as the incumbent once set (an incumbent switch)."""

    new = None

    def derive(self, data, blocks, noise):
        from loop.core.status import Status
        st = dict(super().derive(data, blocks, noise))
        if self.new is not None:
            st = {n: (Status.PASSED if s is Status.INCUMBENT else s) for n, s in st.items()}
            st[self.new] = Status.INCUMBENT
        return st


def _switch_setup(ctx, tmp_path):
    core = _Switched(["open"])
    tick, store, t = _module_tick(ctx, tmp_path, core)
    tick.step()
    [b] = [b for b in store.seed_blocks() if b.id.startswith("test-")]
    return tick, store, t, core, b


def test_a_test_against_a_former_incumbent_closes_as_stale_control_and_frees_its_gpus(ctx, tmp_path):
    """A1 (10-13): n2498's test held 4 primary GPUs against n0655 after n1248 became the incumbent."""
    from loop.contracts import RunStatus, VoidType
    tick, store, t, core, b = _switch_setup(ctx, tmp_path)
    cand, old = b.node_ids
    led = CD.load(tick.db_path, TASK)
    core.new = next(n for n in sorted(led.nodes) if n not in b.node_ids and n != ROOT_NODE_ID)
    mine = [r for r in store.runs(status=[RunStatus.RUNNING]) if r.seed in b.seeds and r.node_id in b.node_ids]
    assert len(mine) == 2 * len(b.seeds)
    t[0] += 60
    rep = tick.step()
    [after] = [x for x in store.seed_blocks() if x.id == b.id]
    assert after.closed_ts == t[0] and after.verdicts == ()
    assert any("stale_control" in x for x in rep.lines)
    assert [e["block"] for e in store.events("test_stale_control")] == [b.id]
    for r in mine:
        got = store.get_run(r.id)
        assert got.status is RunStatus.VOID and got.void_type is VoidType.CANCELLED
    tick.close()
    store.close()


def test_stale_control_cancels_only_pending_runs_when_running_ones_may_finish(ctx, tmp_path):
    from loop.contracts import RunStatus
    tick, store, t, core, b = _switch_setup(ctx, tmp_path)
    tick.ctx = dataclasses.replace(tick.ctx, cfg=dataclasses.replace(
        tick.ctx.cfg, test=dataclasses.replace(tick.ctx.cfg.test, stale_cancel_running=False)))
    led = CD.load(tick.db_path, TASK)
    core.new = next(n for n in sorted(led.nodes) if n not in b.node_ids and n != ROOT_NODE_ID)
    pend = f"{b.node_ids[0]}:confirm:{b.seeds[0]}"
    orig = tick._stale_controls

    def pending_then(*a):                              # SLURM holds this one PENDING at the poll
        tick.book.pending.add(pend)
        return orig(*a)
    tick._stale_controls = pending_then
    t[0] += 60
    tick.step()
    assert store.get_run(pend).status is RunStatus.VOID
    assert store.get_run(f"{b.node_ids[0]}:confirm:{b.seeds[1]}").status is RunStatus.RUNNING
    tick.close()
    store.close()


def test_a_stale_control_is_re_paired_when_the_new_incumbent_ran_every_seed(ctx, tmp_path):
    from loop.contracts import RunStatus
    tick, store, t, core, b = _switch_setup(ctx, tmp_path)
    cand, old = b.node_ids
    led = CD.load(tick.db_path, TASK)
    other = next(n for n in sorted(led.nodes) if n not in b.node_ids and n != ROOT_NODE_ID)
    store.update_seed_block(b.id, node_ids=(other, cand, old))      # a two-candidate block; cand ran every seed
    tick._blocks = None
    core.new = cand
    t[0] += 60
    rep = tick.step()
    [after] = [x for x in store.seed_blocks() if x.id == b.id]
    assert after.open and after.node_ids == (other, cand)
    assert [e["new"] for e in store.events("test_repaired")] == [cand]
    assert all(store.get_run(f"{old}:remint:{s}").status is RunStatus.VOID for s in b.seeds)
    assert all(store.get_run(f"{cand}:confirm:{s}").status is RunStatus.RUNNING for s in b.seeds)
    assert not any(x.startswith(f"{cand}:remint:") for x in rep.launched)       # the control keeps its CONFIRM runs
    tick.close()
    store.close()


def test_heartbeat_shows_the_gap_and_the_acquisition_mode(ctx, tmp_path, monkeypatch):
    """A5 (10-13): hb ... gap=+.0042±.0011 mode=target|incumbent."""
    from loop.core import acquire as A
    tick, store, _ = _module_tick(ctx, tmp_path, TK.ModuleCore())
    tick.step()
    monkeypatch.setattr(A, "target_on", lambda: False)
    assert " gap=" not in tick.heartbeat(1, 0.1)                       # no target: nothing
    monkeypatch.setattr(A, "target_on", lambda: True)
    monkeypatch.setattr(A, "gap_se", lambda led, inc, t=None: (0.0042, 0.0011))
    monkeypatch.setattr(A, "TARGET_ACTIVE", True)
    assert " gap=+.0042±.0011 mode=target " in tick.heartbeat(1, 0.1)
    monkeypatch.setattr(A, "gap_se", lambda led, inc, t=None: (-0.001, 0.0011))
    monkeypatch.setattr(A, "TARGET_ACTIVE", False)
    assert " gap=-.0010±.0011 mode=incumbent " in tick.heartbeat(1, 0.1)
    tick.close()
    store.close()
