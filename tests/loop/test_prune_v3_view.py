"""v3 prune: store statuses are never updated (every node stays QUEUED / SCREENING), so only the DERIVED statuses
close a node -- and a derived SCREENING node used to be kept for life (10-19: 2357 nodes, 1.9 TB). A screening node
now SETTLES once no open work, run or grace window holds it."""

from __future__ import annotations

from types import SimpleNamespace

import dataclasses
import pytest

from loop import tick as TK
from loop.core import data as CD
from loop import prune as PR
from loop.contracts import ROOT_NODE_ID, NodeStatus, RunStatus, Stage
from tests.loop.test_evalonly import ck, node, prune_ctx

NOW = 10 * 86400.0
OLD = NOW - 3 * 86400.0


def view(**kw):
    base = dict(incumbent_id="inc", banked=(), closed=set(), keep=set(), screening=set(), queued=set(), open=set())
    return SimpleNamespace(**{**base, **kw})


def _setup(ctx, store, tmp_path):
    store.ensure_root("croot")
    node(store, "inc", NodeStatus.QUEUED, ts=OLD)                            # store status stale: QUEUED for life
    node(store, "stale", NodeStatus.QUEUED, ts=OLD)                          # derived SCREENING, settled
    node(store, "dead", NodeStatus.SCREENING, ts=OLD)                        # derived FAILED
    node(store, "live", NodeStatus.SCREENING, ts=OLD)                        # derived SCREENING, a run is RUNNING
    node(store, "fresh", NodeStatus.SCREENING, ts=OLD)                       # derived SCREENING, ran inside the grace
    node(store, "kid", NodeStatus.SCREENING, parent="par", ts=OLD)           # live child ...
    node(store, "par", NodeStatus.SCREENING, ts=OLD)                         # ... its parent: never pruned
    node(store, "blk", NodeStatus.SCREENING, ts=OLD)                         # derived SCREENING in an open block
    d = {n: ck(ctx, store, tmp_path, n, Stage.SCREEN, 11) for n in ("inc", "stale", "dead", "live", "fresh", "par", "blk")}
    d["kid"] = ck(ctx, store, tmp_path, "kid", Stage.SCREEN, 11, status=RunStatus.RUNNING)
    d["live2"] = ck(ctx, store, tmp_path, "live", Stage.SCREEN, 12, status=RunStatus.RUNNING)
    for r in store.runs():                                                   # runs finished long ago, but 'fresh'
        store.update_run(r.id, submitted_ts=NOW - 3600.0 if r.node_id == "fresh" else OLD)
    return d


def test_a_settled_screening_node_and_a_derived_closed_node_are_pruned_live_work_is_kept(ctx, store, tmp_path):
    d = _setup(ctx, store, tmp_path)
    snap = view(closed={"dead"}, screening={"stale", "live", "fresh", "kid", "par", "blk"}, keep={"inc", "blk"})
    drop, kept = PR.plan(prune_ctx(ctx, reserve_per_model=0), store, snap, NOW)
    gone = {p.parent.parent.name for p, _ in drop}
    assert gone == {"stale", "dead"}                                         # stale store status, derived-closed/settled
    keep = {n for n in ("inc", "live", "fresh", "kid", "par", "blk") if (d[n] / "latest.pt").exists()}
    assert keep == {"inc", "live", "fresh", "kid", "par", "blk"} and kept >= 5
    assert PR.classify(prune_ctx(ctx), store, snap, NOW)[0] >= {"live", "fresh", "kid", "par", "blk", "inc"}


def test_without_the_v3_screening_view_nothing_new_is_pruned(ctx, store, tmp_path):
    _setup(ctx, store, tmp_path)
    drop, _ = PR.plan(prune_ctx(ctx, reserve_per_model=0), store, view(closed={"dead"}), NOW)
    assert {p.parent.parent.name for p, _ in drop} == {"dead"}


def test_the_incumbent_and_root_and_a_running_slurm_run_survive_settling(ctx, store, tmp_path):
    d = _setup(ctx, store, tmp_path)
    root = ck(ctx, store, tmp_path, ROOT_NODE_ID, Stage.BASELINE, 11)
    snap = view(screening={"stale", "live", "inc"}, keep=set())              # even listed screening, never pruned
    drop, _ = PR.plan(prune_ctx(ctx, reserve_per_model=1), store, snap, NOW)
    gone = {p.parent.parent.name for p, _ in drop}
    assert "live" not in gone and "inc" not in gone and ROOT_NODE_ID not in gone and "stale" in gone
    assert (root / "latest.pt").exists() and (d["live2"] / "latest.pt").exists()


def test_a_plan_written_after_the_prune_plan_protects_its_source_before_the_unlinks(ctx, store, tmp_path):
    """The Pruner re-reads the eval-only plan files right before it unlinks: a plan launched since ``plan()`` ran (one
    resuming a settled node's checkpoint) keeps that checkpoint."""
    import json
    d = _setup(ctx, store, tmp_path)
    snap = view(screening={"stale"}, keep={"inc"})
    st = tmp_path / "state"
    (st / "evalonly" / "stale").mkdir(parents=True)
    pc = prune_ctx(ctx, reserve_per_model=0, dry_run=False)
    pc = __import__("dataclasses").replace(pc, state_dir=str(st))
    real = PR.plan

    def late_plan(*a, **k):
        out = real(*a, **k)                                                  # planned: stale's checkpoint goes ...
        (st / "evalonly" / "stale" / "p.json").write_text(json.dumps({"source_run": "stale:screen:11"}))
        return out                                                           # ... then an eval-only plan names it
    PR.plan = late_plan
    try:
        PR.Pruner(pc, store, lambda: NOW).tick(snap)
    finally:
        PR.plan = real
    assert (d["stale"] / "latest.pt").exists()


def test_every_live_deletion_is_audited_before_the_unlink_and_a_failed_audit_write_skips_the_file(ctx, store, tmp_path):
    """prune_deleted.jsonl gets one line per deleted file (ts, path, bytes, node_id, run_id, reason), written first."""
    import dataclasses
    import json
    d = _setup(ctx, store, tmp_path)
    snap = view(closed={"dead"}, screening={"stale"}, keep={"inc"})
    st = tmp_path / "state"
    st.mkdir()
    pc = dataclasses.replace(prune_ctx(ctx, reserve_per_model=0, dry_run=False), state_dir=str(st))
    seen = []
    real_unlink = type(d["stale"]).unlink

    def spy(self, *a, **k):                                                  # the line is already on disk at unlink time
        seen.append(self.name == "latest.pt" and any(json.loads(x)["path"] == str(self)
                                                     for x in (st / "prune_deleted.jsonl").read_text().splitlines()))
        return real_unlink(self, *a, **k)
    type(d["stale"]).unlink = spy
    try:
        PR.Pruner(pc, store, lambda: NOW).tick(snap)
    finally:
        type(d["stale"]).unlink = real_unlink
    rows = [json.loads(x) for x in (st / "prune_deleted.jsonl").read_text().splitlines()]
    assert {r["node_id"] for r in rows} == {"stale", "dead"} and all(seen) and len(seen) == len(rows)
    assert all(set(r) == {"ts", "path", "bytes", "node_id", "run_id", "reason"} and r["bytes"] == 100 for r in rows)
    assert {r["node_id"]: r["reason"] for r in rows} == {"stale": "settled screening", "dead": "derived-closed"}
    assert not (d["stale"] / "latest.pt").exists() and not (d["dead"] / "latest.pt").exists()


def test_a_failed_audit_write_skips_the_deletion(ctx, store, tmp_path):
    import dataclasses
    d = _setup(ctx, store, tmp_path)
    snap = view(closed={"dead"}, screening={"stale"}, keep={"inc"})
    st = tmp_path / "state"
    st.mkdir()
    (st / "prune_deleted.jsonl").mkdir()                                     # opening it for append fails
    pc = dataclasses.replace(prune_ctx(ctx, reserve_per_model=0, dry_run=False), state_dir=str(st))
    PR.Pruner(pc, store, lambda: NOW).tick(snap)
    assert (d["stale"] / "latest.pt").exists() and (d["dead"] / "latest.pt").exists()


def test_the_alt_base_protection_has_a_one_sd_hysteresis_band_and_a_configurable_top(ctx, tmp_path):
    """alt_top (default 60) best standing nodes keep their checkpoint, and so does every node within one posterior sd
    of the Nth's value; a node clearly below the band settles."""
    from loop.spec import PruneCfg
    from tests.loop.conftest import InlineExecutor
    from tests.loop.test_ops import _quiet
    from tests.loop.test_port import _commit, _node, _two_epochs
    assert PruneCfg().alt_top == 60
    sim, store = _two_epochs(ctx, tmp_path)
    for i, nid in enumerate(("a", "b", "c", "d")):
        _node(store, nid, _commit(sim, store.get_node("root2").commit, nid, {"train.lr": (6 + i) * 1e-4}), "root2", 12.0 + i)
    cfg = dataclasses.replace(ctx.cfg, prune=dataclasses.replace(ctx.cfg.prune, alt_top=2))
    c2 = dataclasses.replace(ctx, cfg=cfg)
    tick = TK.Tick(c2, store, sim, TK.StubCore(), db_path=tmp_path / "loop.db", executor=InlineExecutor, **_quiet())
    # a .020, b .015, c .0105 (within 1 sd = .01 of the 2nd, b: .015 - .01 = .005 <= c), d .000 (below the band)
    tick.post = SimpleNamespace(beliefs=lambda: {"a": (0.02, 1e-4, 5), "b": (0.015, 1e-4, 5),
                                                 "c": (0.0105, 1e-4, 5), "d": (0.0, 1e-4, 5)})
    led = CD.load(tmp_path / "loop.db", c2.task_id)
    st = {n: "SCREENING" for n in ("a", "b", "c", "d")} | {"inc@new": "INCUMBENT"}
    view_ = tick.prune_view(led, st)
    assert {"a", "b", "c"} <= view_.keep and "d" in view_.screening and "d" not in view_.keep
    tick.close()
    store.close()


def test_audit_write_failure_skips_the_file_but_a_full_disk_still_frees_space(tmp_path, monkeypatch):
    import errno
    from types import SimpleNamespace
    from loop import prune as P
    logged = []
    pr = P.Pruner.__new__(P.Pruner)
    pr.ctx = SimpleNamespace(state_dir=str(tmp_path / "st"))
    pr.store = SimpleNamespace(log=lambda *a, **k: logged.append((a, k)))
    pr._audit(tmp_path / "a.pt", 5, ("n1", "r1", "closed"), 1.0)
    assert '"node_id": "n1"' in (tmp_path / "st" / "prune_deleted.jsonl").read_text()
    (tmp_path / "bad").write_text("x")
    pr.ctx = SimpleNamespace(state_dir=str(tmp_path / "bad"))          # not a directory: the write fails -> raises (skip)
    with pytest.raises(OSError):
        pr._audit(tmp_path / "a.pt", 5, None, 1.0)
    pr.ctx = SimpleNamespace(state_dir="")
    with pytest.raises(OSError):
        pr._audit(tmp_path / "a.pt", 5, None, 1.0)
    pr.ctx = SimpleNamespace(state_dir=str(tmp_path / "st2"))
    monkeypatch.setattr(P.Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")))
    pr._audit(tmp_path / "a.pt", 5, None, 1.0)                         # ENOSPC: no raise (the delete proceeds), logged
    assert logged and logged[-1][0][0] == "prune_audit_unwritable"


def test_the_best_challengers_by_absolute_level_keep_their_checkpoints_even_when_their_own_gain_ranks_low(ctx, tmp_path):
    """10-22 ops review: n3029 (the best challenger, post ~.6823) settled and was pruned: the alt-base ranking orders by the
    node's OWN increment over its parent, so a strong child of a strong parent ranks low. Challengers are protected by
    ABSOLUTE composite level: the top ``challenger_top``, everything within one sd of the best, and every node in TEST."""
    from loop.core import model as M
    from loop.spec import PruneCfg
    from tests.loop.conftest import InlineExecutor
    from tests.loop.test_ops import _quiet
    from tests.loop.test_port import _commit, _node, _two_epochs
    assert PruneCfg().challenger_top == 8
    sim, store = _two_epochs(ctx, tmp_path)
    ids = ("a", "b", "c", "d", "t")
    for i, nid in enumerate(ids):
        _node(store, nid, _commit(sim, store.get_node("root2").commit, nid, {"train.lr": (6 + i) * 1e-4}), "root2", 12.0 + i)
    c2 = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, prune=dataclasses.replace(ctx.cfg.prune, challenger_top=1,
                                                                                         alt_top=0)))
    tick = TK.Tick(c2, store, sim, TK.StubCore(), db_path=tmp_path / "loop.db", executor=InlineExecutor, **_quiet())
    level = {"a": 0.690, "b": 0.689, "c": 0.600, "d": 0.601, "t": 0.500, "inc@new": 0.680}
    obs = lambda lv: [SimpleNamespace(node="x", stage="screen")] * 3 and [lv]                  # noqa: E731
    noise = SimpleNamespace(sigma2={M.COMPOSITE: 0.0009 * 3})                                   # sd of a 3-seed mean: .0300
    tick.post = SimpleNamespace(beliefs=lambda: {}, fit=lambda base: (noise, {}), by={n: obs(v) for n, v in level.items()})
    real = M.parent_mean
    M.parent_mean = lambda o, t, w=None: (o[0], 3)
    try:
        led = CD.load(tmp_path / "loop.db", c2.task_id)
        st = {n: "SCREENING" for n in ids} | {"t": "TESTING", "inc@new": "INCUMBENT"}
        got = tick._prune_challengers(led, st)
        view_ = tick.prune_view(led, st)
    finally:
        M.parent_mean = real
    # a is the best (.690), b is within one sd (.030) of it, c / d are .09 below; t is in TEST (always); the incumbent is no challenger
    assert got == {"a", "b", "t"}
    assert {"a", "b"} <= view_.keep and {"c", "d"} <= view_.screening and "t" in view_.keep
    tick.close()
    store.close()
