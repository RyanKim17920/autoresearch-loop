"""loop.tick: 200 ticks end-to-end on the sim adapter with stub core functions (ideas -> research sessions -> nodes ->
selection seeds -> fresh-seed tests -> an incumbent; infra voids retried, an OOM laddered to activation
checkpointing, a seed-pool violation refused before submit), and ``loop shadow`` on a ledger fixture."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import yaml

from loop import runner as R
from loop import tick as TK
from loop.adapters.sim import SimAdapter
from loop.contracts import ROOT_NODE_ID, CandidateRequest, Node, NodeStatus, Operator, RunStatus, Stage, Strategy
from loop.store import SqliteStore

from .conftest import InlineExecutor
from .core.conftest import SAMPLE, TASK

FRESH = 100_000
N_SEEDS, MAX_NODES = 3, 6


class _Adapter(SimAdapter):
    def oom_ladder(self, commit):            # the nanopath adapter's ladder: the runner adds activation checkpointing
        return ("expandable_segments",)


class SimCore(TK.StubCore):
    """A toy decision core over the ledger view: a node is SCREENING until it has N_SEEDS done selection runs, then
    TESTING until its test block is done, then INCUMBENT if it beat root on the test seeds, else FAILED. Every
    seed-pool rule is the stub's; ``check_submit`` refuses a SCREEN run on a TEST seed."""

    def __init__(self, sim):
        self.sim, self.checked, self.violate_at = sim, 0, None
        self.n_ranks = 0

    def derive(self, data, blocks, noise):
        done = {}
        for r in data.runs:
            if r.status == "done":
                done.setdefault(r.node, []).append(r)
        out = {}
        for nid in data.nodes:
            if nid == ROOT_NODE_ID:
                continue
            sel = [r for r in done.get(nid, []) if r.seed < FRESH]
            test = [r for r in done.get(nid, []) if r.seed >= FRESH and r.stage == "confirm"]
            if len(sel) < N_SEEDS:
                out[nid] = "SCREENING"
            elif not any(r.node == nid and r.seed >= FRESH for r in data.runs):
                out[nid] = "SCREENED"
            elif len(test) < TK.StubCore.TEST_K:
                out[nid] = "TESTING"
            else:
                out[nid] = "PASSED" if self.sim.true_panel(self._commit(nid)) > self.sim.true_panel(
                    self.sim.root_commit()) else "FAILED"
        passed = sorted(n for n, s in out.items() if s == "PASSED")
        if passed:
            out[passed[0]] = "INCUMBENT"
        return out

    def _commit(self, nid):
        return self.store.get_node(nid).commit

    def rank(self, data, blocks, statuses, post, free_slots, llm_free):
        self.n_ranks += 1
        st = self.derive(data, blocks, None)
        acts = []
        if self.violate_at is not None and self.n_ranks == self.violate_at and st:
            acts.append(TK.Action("seed", "VIOLATE:" + sorted(st)[0], "explore", 1.0, 1.0))
        for nid, s in sorted(st.items()):
            if s == "SCREENING":
                live = sum(r.node == nid and r.seed < FRESH and r.status in ("running", "submitting")
                           for r in data.runs)
                acts += [TK.Action("seed", nid, "explore", 1.0, 0.01)] * max(0, 1 - live)
            elif s == "TESTING" or s == "SCREENED" and not any(v == "TESTING" for v in st.values()):
                acts.append(TK.Action("test", nid, "explore", 10.0, 0.005))   # one block open at a time
        if len(data.nodes) - 1 < MAX_NODES:
            acts.append(TK.Action("idea", None, "explore", 2.0, 0.001))
        return acts

    def runs_for(self, action, data, blocks, statuses, cfg):
        if action.node and action.node.startswith("VIOLATE:"):      # a SCREEN run on a TEST seed
            return [TK.RunSpec(action.node.split(":", 1)[1], Stage.SCREEN, FRESH + 999)]
        return super().runs_for(action, data, blocks, statuses, cfg)

    def check_submit(self, run, blocks):
        self.checked += 1
        if run.seed >= FRESH and run.stage is Stage.SCREEN:
            raise ValueError(f"SeedPoolViolation: TEST seed {run.seed} used by a selection run")


class Researcher:
    """Writes a config change in a sim worktree and commits it (what a research session returns)."""

    def __init__(self, sim):
        self.sim, self.n = sim, 0

    def research(self, req, store, *args):
        self.n += 1
        wt = self.sim.prepare_worktree(parent_commit=self.sim.root_commit(), name=req.id)
        p = Path(wt.path) / "config.yaml"
        cfg = yaml.safe_load(p.read_text())
        cfg["train"]["lr"] = 1e-3 * (0.5 if self.n % 2 else 2.0)
        if self.n == 3:
            cfg["train"]["epochs"] = 20                  # OOMs above 12 unless activation checkpointing
        p.write_text(yaml.safe_dump(cfg))
        return SimpleNamespace(status="ok", commit=self.sim.commit(wt, message=req.id))


def _ideate(tick, ideas, data, statuses):
    out = []
    for i, _ in enumerate(ideas[:tick.research.free()]):
        rid = f"req-{len(tick.store.nodes())}-{i}-{len(tick.research.inflight)}"
        tick.research.start(CandidateRequest(id=rid, task_id=tick.ctx.task_id, strategy=Strategy.EXPLORE,
                                             operator=Operator.DRAFT, island="main", parent_ids=(ROOT_NODE_ID,),
                                             brief="idea"))
        out.append(f"researcher started: {rid}")
    return out


def _intake(store):
    def intake(req, res):
        nid = f"n{len(store.nodes()):04d}"
        store.put_node(Node(id=nid, task_id=store.task_id, commit=res.commit, parent_id=ROOT_NODE_ID,
                            strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="f", mechanism_tag="m",
                            diff_hash=f"h-{req.id}", status=NodeStatus.QUEUED))
        return f"intake {req.id} -> {nid}"
    return intake


def test_200_ticks_end_to_end_on_the_sim(ctx, tmp_path):
    clock = iter(range(10**9))
    now = lambda: 1_000_000.0 + 60.0 * next(clock)          # noqa: E731
    db = tmp_path / "loop.db"
    store = SqliteStore(db, ctx.task_id, weights=ctx.cfg.weights)
    sim = _Adapter(ctx, duration_ticks=3)
    sim.oom_above = 12
    sim.inject_fault("n0002:screen:*", "infra", times=1)
    core = SimCore(sim)
    core.store, core.violate_at = store, 40
    tick = TK.Tick(ctx, store, sim, core, db_path=db, researcher=Researcher(sim), executor=InlineExecutor,
                   clock=now, ideate=_ideate, intake=_intake(store))
    lines, reports = [], []
    for _ in range(200):
        rep = tick.step()
        reports.append(rep)
        lines += rep.lines
    runs = store.runs()
    nodes = [n for n in store.nodes() if n.id != ROOT_NODE_ID]
    assert len(nodes) == MAX_NODES
    # every node got its selection seeds (CRN off root's none -> 1, 2, 3) and a test block vs the incumbent (root)
    for n in nodes:
        assert sorted(r.seed for r in runs if r.node_id == n.id and r.stage is Stage.SCREEN
                      and r.status is RunStatus.DONE) == [1, 2, 3], n.id
    tests = [r for r in runs if r.stage in (Stage.CONFIRM, Stage.REMINT)]
    assert tests and all(r.seed >= FRESH and r.status is RunStatus.DONE for r in tests)
    final = reports[-1].statuses
    assert set(final.values()) <= {"INCUMBENT", "PASSED", "FAILED"}
    assert TK.incumbent_of(final) != ROOT_NODE_ID                  # lowering lr helps in the sim
    controls = {r.node_id for r in tests if r.stage is Stage.REMINT}   # the incumbent of the day: root, then a pass
    assert ROOT_NODE_ID in controls and controls - {ROOT_NODE_ID} <= {n for n, st in final.items()
                                                                       if st in ("INCUMBENT", "PASSED")}
    # infra void retried on the same row; the OOM laddered to activation checkpointing and finished
    void_retried = store.get_run("n0002:screen:1")
    assert void_retried.status is RunStatus.DONE and void_retried.attempt == 1
    oom = [r for r in runs if r.mitigation == R.ACT_CKPT]
    assert oom and all(r.status is RunStatus.DONE for r in oom)
    # the seed-pool violation never reached the adapter; every other submit went through the guard
    refused = store.events("submit_refused")
    assert len(refused) == 1 and refused[0]["seed"] == FRESH + 999
    assert not (sim.root / "jobs" / f"{sim._job_id(store.get_run(refused[0]['run_id']))}.json").exists()
    jobs = len(list((sim.root / "jobs").glob("*.json")))
    assert tick.submits == jobs and core.checked == jobs + 1
    # no run row twice, nothing left running, no session left in flight
    assert len({r.id for r in runs}) == len(runs)
    assert not [r for r in runs if r.status in (RunStatus.RUNNING, RunStatus.SUBMITTING)]
    assert not tick.research.inflight and store.get_state("driver.inflight") == []
    tick.close()
    store.close()


def test_live_ticks_until_stop_and_closes(ctx, tmp_path):
    t = [1_000_000.0]
    db = tmp_path / "loop.db"
    sim = _Adapter(ctx, duration_ticks=3)
    core = SimCore(sim)
    core.store = SqliteStore(db, ctx.task_id, weights=ctx.cfg.weights)
    printed = []
    n = TK.live(ctx, db, core, sim, Researcher(sim), max_ticks=60, stop=lambda: t[0] >= 1_000_000.0 + 25 * 60.0,
                clock=lambda: t[0], sleep=lambda dt: t.__setitem__(0, t[0] + dt), out=printed.append,
                executor=InlineExecutor, ideate=_ideate, intake=_intake(core.store))
    assert 0 < n < 60 and printed                                   # stopped between ticks, not by max_ticks
    store = SqliteStore(db, ctx.task_id, weights=ctx.cfg.weights)
    assert [x for x in store.nodes() if x.id != ROOT_NODE_ID] and store.runs()
    store.close()
    core.store.close()


def test_cli_run_is_the_v3_tick_and_run_v2_the_old_driver(tmp_path, monkeypatch, fake_loop_spec):
    import loop.__main__ as M
    seen = {}
    monkeypatch.setattr(TK, "live", lambda ctx, db, core, adapter, researcher, **kw: seen.update(db=db, kw=kw) or 0)
    monkeypatch.setattr(M, "resolve_adapter", lambda ctx: SimAdapter(ctx))
    assert M.main(["run", "--task", str(fake_loop_spec), "--fake-llm", "--max-ticks", "1",
                   "--state-dir", str(tmp_path / "st")]) == 0
    assert seen["db"].suffix == ".db" and seen["kw"]["max_ticks"] == 1


# ---------------------------------------------------------------------------------------------- shadow
def _ledger(path: Path) -> Path:
    """A store-schema SQLite ledger holding the core fixture (an extract of the 10-08 nanopath-v2 ledger)."""
    sample = json.loads(SAMPLE.read_text())
    SqliteStore(path, TASK).close()                              # the store's schema
    con = sqlite3.connect(path)
    for table in ("nodes", "runs"):
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for row in sample[table]:
            cols = [c for c in row if c in have]
            con.execute(f"INSERT INTO {table} ({', '.join(map(repr, cols))}) VALUES ({', '.join('?' * len(cols))})",
                        [row[c] for c in cols])
    for r in sample["log"]:
        con.execute("INSERT INTO log (task_id, ts, event, fields) VALUES (?, ?, ?, ?)",
                    (r["task_id"], r["ts"], r["event"], r["fields"]))
    con.commit()
    con.close()
    return path


class LedgerCore(TK.StubCore):
    """Statuses from the ledger view: the last carrier is the incumbent; nodes with < 3 selection observations
    get one more seed; one test of the best-observed node."""

    def derive(self, data, blocks, noise):
        inc = data.carriers[-1][1]
        by = data.by_node()
        return {n: ("INCUMBENT" if n == inc else "SCREENING" if len(by.get(n, [])) < 3 else "SCREENED")
                for n in data.nodes}

    def rank(self, data, blocks, statuses, post, free_slots, llm_free):
        st = self.derive(data, blocks, None)
        seeds = [TK.Action("seed", n, data.nodes[n].lane, 1.0, 0.01) for n, s in sorted(st.items())
                 if s == "SCREENING" and data.nodes[n].parent is not None][:3]
        return seeds + [TK.Action("idea", None, "explore", 2.0, 0.001)]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_shadow_runs_on_a_copy_logs_statuses_and_actions_and_never_writes_the_live_ledger(ctx, tmp_path):
    live = _ledger(tmp_path / "live.db")
    before = _sha(live)
    copy = tmp_path / "shadow" / "copy.db"
    copy.parent.mkdir()
    ctx = dataclasses.replace(ctx, task_id=TASK)
    printed = []
    t = [1e9]                                                       # fake time, advanced by the naps between ticks
    n = TK.shadow(ctx, copy, LedgerCore(), refresh_from=live, refresh_s=120.0, max_ticks=3, every_s=60.0,
                  clock=lambda: t[0], sleep=lambda dt: t.__setitem__(0, t[0] + dt), out=printed.append)
    assert n == 3 and _sha(live) == before                         # the live ledger is never written
    recs = [json.loads(x) for x in Path(f"{copy}.shadow.jsonl").read_text().splitlines()]
    assert [r["tick"] for r in recs] == [0, 1, 2]
    assert recs[0]["incumbent"] != ROOT_NODE_ID and recs[0]["statuses"]["INCUMBENT"] == 1
    assert {a["kind"] for a in recs[0]["actions"]} == {"seed", "idea"} and recs[0]["would_submit"]
    assert sum("copy refreshed" in p for p in printed) == 2          # before tick 0, and at 120 s (tick 2)
    con = sqlite3.connect(copy)
    jobs = [j for (j,) in con.execute("SELECT job_id FROM runs WHERE job_id LIKE 'dry-%'")]
    con.close()
    assert jobs                                                       # dry-run rows live only in the copy


def test_shadow_cli_with_the_core_stubs(ctx, tmp_path, fake_loop_spec, capsys):
    from loop.__main__ import main
    live = _ledger(tmp_path / "live.db")
    copy = tmp_path / "c" / "copy.db"
    copy.parent.mkdir()
    rc = main(["shadow", "--task", str(fake_loop_spec), "--db", str(copy), "--refresh-from", str(live),
               "--task-id", TASK, "--max-ticks", "1", "--every-s", "0"])
    assert rc == 0 and "shadow tick 0" in capsys.readouterr().out
    assert Path(f"{copy}.shadow.jsonl").exists()
