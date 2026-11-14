"""Task conformance (11-10 generalization): every ``tasks/*.yaml`` loads; every v3 task (a ``loop:`` block) resolves,
names a conforming adapter, installs ITS OWN axes (never nanopath's by accident), resolves every score-unit default
against its own measured noise, and runs a dry lifecycle PLUMBING check on its own resolved config: smoke (the offline
simulator on the task's axes, not the task's own adapter) -> screen -> fresh-seed TEST -> re-root (the incumbent switch)
through the real tick and core on a dry executor. What it does NOT prove: the TEST verdict is stubbed (a pass) -- the
test statistics and the local-reroot tier are covered by tests/loop/core/test_test*.py and test_local_reroot_tick.py
-- and the ledger is a real nanopath ledger extract re-keyed onto the task's axes by position; the external target,
locks and pool sizes are relaxed. A new task passes this file before it is calibrated (docs/new_task.md)."""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import re
import sqlite3
from pathlib import Path

import pytest
import yaml

from kernel.model import TaskSpec
from loop import contracts as C
from loop import project as P
from loop import tick as TK
from loop.adapters.sim import SimAdapter
from loop.core import data as CD
from loop.spec import SIGMA_SCALED, load_context
from loop.store import SqliteStore

from .test_tick_core import _WithTests

REPO = Path(__file__).resolve().parents[2]
TASKS = sorted((REPO / "tasks").glob("*.yaml"))
V3 = [p for p in TASKS if (yaml.safe_load(p.read_text()) or {}).get("loop")]
SAMPLE = REPO / "tests" / "loop" / "core" / "fixtures" / "ledger_sample.json"


def _ids(paths):
    return [p.stem for p in paths]


def _measured(ctx):
    """An UNMEASURED task (the template, before calibrate) gets synthetic sigmas so its defaults resolve."""
    if math.isfinite(ctx.cfg.panel_sd):
        return ctx
    from loop.spec import sigma_scaled
    cfg = dataclasses.replace(ctx.cfg, panel_sd=0.004, axis_sd={a: 0.005 for a in ctx.cfg.axes})
    raw = (yaml.safe_load(Path(ctx.spec_path).read_text()) or {})["loop"]
    return dataclasses.replace(ctx, cfg=sigma_scaled(cfg, raw))


@pytest.mark.parametrize("path", TASKS, ids=_ids(TASKS))
def test_every_task_yaml_loads(path):
    spec = TaskSpec.load(path)
    assert spec.name and spec.slots.weights


def test_the_v3_tasks_are_the_expected_ones():
    assert {"nanobrain", "nanopath_v2", "fake_loop", "_template"} <= set(_ids(V3))


@pytest.mark.parametrize("path", V3, ids=_ids(V3))
def test_v3_task_resolves_with_its_own_axes_and_a_conforming_adapter(path, tmp_path):
    ctx = _measured(load_context(path, state_dir=tmp_path, require_measured=False))
    mod, _, cls = ctx.cfg.execution_adapter.partition(":")
    adapter = getattr(importlib.import_module(mod), cls)
    assert C.adapter_conformance(adapter) == []
    explicit_default = (yaml.safe_load(path.read_text())["loop"] or {}).get("project", "absent") is None
    if not explicit_default:                                         # (fake_loop opts into the built-in registry)
        assert set(P.PROBE_KEYS.values()) == set(ctx.cfg.axes)      # the registry is this task's, not nanopath's
    assert P.target() in P.AXES and set(P.GUARD_AXES) <= set(P.AXES)
    for dotted in SIGMA_SCALED:                                      # every score-unit default resolved, finite
        node = ctx.cfg
        for part in dotted.split("."):
            node = getattr(node, part) if node is not None else None
        assert node is None or (math.isfinite(node) and node >= 0), dotted


def test_adapter_catalogue_lists_every_probe_in_the_loop():
    """Every ``getattr`` / ``hasattr`` probe on an adapter in loop/ names a REQUIRED or OPTIONAL member."""
    rx = re.compile(r"(?:getattr|hasattr)\((?:self\.)?_?(?:adapter|ad)\b, \"(\w+)\"")
    probed = {m for f in (REPO / "loop").rglob("*.py") for m in rx.findall(f.read_text())}
    unknown = probed - set(C.ADAPTER_REQUIRED) - set(C.ADAPTER_OPTIONAL)
    assert not unknown, f"add these to loop.contracts.ADAPTER_OPTIONAL: {sorted(unknown)}"


def _ledger(path: Path, task_id: str, axes) -> Path:
    """The core fixture ledger (synthetic, in the shape of a real one) re-keyed onto ``axes``: nanopath's axis keys map in
    order onto the task's, extra ones are dropped, the aggregate is the task's weighted panel."""
    sample = json.loads(SAMPLE.read_text())
    src = list(P._PROBE_KEYS.values())
    w = {a: 1.0 / len(axes) for a in axes}
    SqliteStore(path, task_id).close()
    con = sqlite3.connect(path)
    for table in ("nodes", "runs"):
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for row in sample[table]:
            row = {**row, "task_id": task_id}
            if table == "runs" and row.get("scores"):
                sc = json.loads(row["scores"])
                new = {a: sc[k] for k, a in zip(src, axes) if k in sc}
                row["scores"] = json.dumps(new)
                if len(new) == len(axes):
                    row["aggregate"] = sum(w[a] * v for a, v in new.items())
            cols = [c for c in row if c in have]
            con.execute(f"INSERT INTO {table} ({', '.join(map(repr, cols))}) VALUES ({', '.join('?' * len(cols))})",
                        [row[c] for c in cols])
    for r in sample["log"]:
        con.execute("INSERT INTO log (task_id, ts, event, fields) VALUES (?, ?, ?, ?)",
                    (task_id, r["ts"], r["event"], r["fields"]))
    con.commit()
    con.close()
    return path


class _Lifecycle(_WithTests):
    """The real core: tick 1 ranks (screens + the best test belief); afterwards every opened TEST block passes."""

    def __init__(self):
        super().__init__(1)

    def decide(self, data, blocks, noise, incumbent):
        from loop.core.test import Kind, Verdict
        return [(b, b.node_ids[0], Verdict(Kind("pass"), 12.0, 5, 0.003, "conformance stub", (), 0))
                for b in blocks if b.pool == CD.TEST and b.open and b.id.startswith("test-")]


@pytest.mark.parametrize("path", V3, ids=_ids(V3))
def test_v3_task_runs_smoke_screen_test_reroot_on_its_own_config(path, tmp_path):
    ctx = _measured(load_context(path, state_dir=tmp_path / "state", require_measured=False))
    # smoke: the offline executor honours the task's axes and the smoke contract
    sim = SimAdapter(dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, execution_adapter="sim")))
    wt = sim.prepare_worktree(parent_commit=sim.root_commit(), name="conformance")
    assert sim.smoke(wt).ok
    # screen -> test -> re-root: the real tick + core on the task's resolved config (dry executor)
    db = _ledger(tmp_path / "live.db", ctx.task_id, list(P.PROBE_KEYS.values()))
    # plumbing only: no external (Labless) target, no deployed-HEAD lock check (a dev tree differs from HEAD)
    cfg = ctx.cfg
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        cfg, slots=64, slurm=dataclasses.replace(cfg.slurm, pools=tuple(dataclasses.replace(p, slots=max(p.slots, 32))
                                                                         for p in cfg.slurm.pools)), labless=dataclasses.replace(cfg.labless, enabled=False, leader_median=None),
        acquire=dataclasses.replace(cfg.acquire, target_mode="incumbent"),
        core_meta=dataclasses.replace(cfg.core_meta, locks=False)))
    store = SqliteStore(db, ctx.task_id, weights=ctx.cfg.weights)
    t = [2e9]
    tick = TK.Tick(ctx, store, TK.DryRunAdapter(store, lambda: t[0]), _Lifecycle(), db_path=db,
                   clock=lambda: t[0], dry=True)
    try:
        rep = tick.step()
        launched = [store.get_run(r) for r in rep.launched]
        assert any(r is not None and r.stage is C.Stage.SCREEN for r in launched), rep.lines[-20:]
        [test] = [b for b in store.seed_blocks() if b.id.startswith("test-")]
        cand, inc = test.node_ids
        t[0] += 60
        tick.step()
        after = {b.id: b for b in store.seed_blocks()}
        assert [r["kind"] for r in after[test.id].verdicts] == ["pass"]
        [remint] = [b for b in after.values() if b.id.startswith("remint-")]
        assert remint.node_ids == (cand,)                            # the re-root: the candidate is the new incumbent
        assert TK.incumbent_of(tick.statuses) == cand and cand != inc
    finally:
        tick.close()
        store.close()


def test_score_unit_defaults_scale_with_the_measured_panel_sd_unless_pinned():
    from loop.spec import LoopConfig, build, sigma_scaled
    base = {"execution_adapter": "a:b", "axes": ["x"], "weights": {"x": 1.0}, "axis_sd": {"x": 0.01}}
    cfg = sigma_scaled(build(LoopConfig, {**base, "panel_sd": 0.01}, "loop"), {})
    assert cfg.status.outlier_sd_d == pytest.approx(0.01) and cfg.test.local_reroot_min_gain == pytest.approx(0.006)
    assert cfg.labless.sibling_gain == pytest.approx(0.025) and cfg.prior.tau0 == pytest.approx(0.015)
    pinned = sigma_scaled(build(LoopConfig, {**base, "panel_sd": 0.01, "status": {"outlier_sd_d": 0.0034}}, "loop"),
                          {"status": {"outlier_sd_d": 0.0034}})
    assert pinned.status.outlier_sd_d == 0.0034                      # a yaml value always wins
    nan = sigma_scaled(build(LoopConfig, {**base, "panel_sd": float("nan")}, "loop"), {})
    assert nan.status.outlier_sd_d == 0.0034                         # unmeasured (calibrate): the literals stand


def test_a_spec_without_a_project_block_gets_its_own_axes(tmp_path):
    raw = yaml.safe_load((REPO / "tasks" / "fake_loop.yaml").read_text())
    del raw["loop"]["project"]                                       # absent (not the explicit null)
    (tmp_path / "t.yaml").write_text(yaml.safe_dump(raw))
    ctx = load_context(tmp_path / "t.yaml", state_dir=tmp_path / "s")
    assert tuple(P.AXES) == ("alpha", "beta", "gamma") and P.guard() is None and not P.HARM_H
    assert ctx.cfg.project.noise_floor_seeds == P.NOISE_FLOOR_SEEDS
