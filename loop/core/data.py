"""loop.core.data — a read-only, typed view of the ledger (docs/loop-core-v3.md §2, §8).

``load(db, task_id)`` reads ``nodes`` / ``runs`` / ``log`` of ONE task and returns a frozen :class:`Ledger`:
every run (:class:`Run`), every DONE run as an observation (:class:`Obs`: six probe axes + the official panel),
and every node with its derived ``lane`` and ``carrier`` (:class:`Node`). Nothing is written.

Seed pools (verified on the 2026-10-08 ledger copy, 4,917 runs):
- TEST: seeds of a ``seeds_drawn`` block whose purpose is ``remint`` / ``audit`` / ``confirm`` (never-used draws
  >= ``fresh_base``), plus seeds no stage but ``audit`` / ``remint`` ever ran (the fixed held-out 201-204 and the
  one-off 990000/990001 audits). Their control runs (stage parent / baseline / remint) are TEST too.
- SELECT: everything else -- screen / promote / parent / baseline / calibrate, the shared confirm pool
  (14, 101-108), and the ``legs`` factorial blocks drawn >= ``fresh_base`` (>= fresh_base alone is NOT a test seed).
``HELD_OUT_STAGES`` (confirm / audit / remint) are the runs a node was never selected on: its fresh outcome.

Base epochs (docs §8 "Base epochs"): every node descends from one BASE ROOT (a node without a parent; its commit is
the base SHA). ``Node.base`` / ``Run.base`` / ``Obs.base`` = that SHA; ``Ledger.epoch`` = the base of the latest base
root. A base change (e.g. common random numbers) changes the data order, so runs only pair within an epoch.
A node tagged ``port_of=<old id>`` in its notes is the graft of an old-epoch node onto the new base."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

from loop import project as _P
from loop.project import AXES, OFFICIAL_WEIGHTS, PROBE_KEYS  # noqa: E402,F401 - live views (loop.project)

#: AXES / PROBE_KEYS / OFFICIAL_WEIGHTS are LIVE views of the per-process project registry (loop.project): with no
#: ``loop.project`` block they are nanopath's six probe axes, their summary keys and ~/nanopath/base/probe.py
#: SCORE_WEIGHTS (the official panel; ``runs.aggregate`` is exactly this weighted sum)
PANEL = "panel"
ROOT = "root"
SELECT, TEST = "select", "test"
HELD_OUT_STAGES = frozenset({"confirm", "audit", "remint"})
TEST_PURPOSES = ("remint", "audit", "confirm")
TEST_ONLY_STAGES = frozenset({"audit", "remint"})
LANES = ("explore", "crossover", "exploit", "factorial")
#: notes tags of the old HP lanes (tune / ladders / campaigns / variants / legs / design): all lane ``factorial``
FACTORIAL_TAGS = ("tune=", "tune_of=", "legs=", "campaign_of=", "variants_of=", "variant=", "ladder=", "block_of=")
PORT_TAG = re.compile(r"\bport_of=(\S+)")


@dataclass(frozen=True)
class Run:
    id: str
    node: str
    seed: int
    stage: str
    status: str
    gpu_h: float
    submitted_ts: float
    done_ts: float           # run_finished log event (else submitted_ts): when its result became available
    output_dir: str
    resumed: int
    base: str = ""           # the base SHA of the node's lineage (its base root's commit)


@dataclass(frozen=True)
class Obs:
    """One DONE run: ``axes`` (short names, :data:`AXES`) and the official ``panel``."""
    run_id: str
    node: str
    seed: int
    stage: str
    pool: str
    axes: Mapping[str, float]
    panel: float
    gpu_h: float
    ts: float                # availability (= Run.done_ts)
    resumed: int
    val_loss_q25: float | None = None
    base: str = ""
    #: the run's optimizer steps (``loop.project.throughput.key`` in its scores; None = not recorded / filter off):
    #: a wall-clock budget's throughput covariate (core.model.throughput_clean)
    steps: float | None = None

    def value(self, target: str, weights: Mapping[str, float] | None = None) -> float:
        """An axis, the panel, or (``weights`` given) the weighted composite of the axes."""
        if target == PANEL:
            return self.panel
        if target in self.axes:
            return self.axes[target]
        if weights is None:
            raise KeyError(f"unknown target {target!r} (a composite needs weights)")
        return sum(w * self.axes[a] for a, w in weights.items())


@dataclass(frozen=True)
class Node:
    id: str
    parent: str | None
    second_parent: str | None
    strategy: str
    family: str
    lane: str
    carrier: str             # nearest ancestor that had passed a held-out audit when the node was created
    created_ts: float
    status: str              # the OLD loop's stored status: informative only, never a decision input
    notes: str
    base: str = ""           # base SHA: the commit of its nearest parentless ancestor (itself for a base root)
    port_of: str = ""        # the old-epoch node this one is the graft of (notes ``port_of=<id>``)


@dataclass(frozen=True)
class Ledger:
    nodes: Mapping[str, Node]
    runs: tuple[Run, ...]
    obs: tuple[Obs, ...]
    test_seeds: frozenset[int]
    carriers: tuple[tuple[float, str], ...]     # (ts, node) a node became a carrier (root at 0)
    draws: tuple[tuple[float, str, tuple[int, ...]], ...] = ()     # seeds_drawn events: (ts, purpose, seeds)
    incumbents: tuple[tuple[float, str, str], ...] = ()            # recorded switches: (ts, node, previous)
    audits: tuple[tuple[float, str, str], ...] = ()                # recorded audits: (ts, node, verdict)
    bases: tuple[tuple[float, str, str], ...] = ()                 # base roots: (created_ts, node, base SHA)
    verdicts: tuple[tuple[float, str, str], ...] = ()              # test_verdict events: (ts, node, pass | fail)

    @property
    def epoch(self) -> str:
        """The current base: that of the latest base root ('' for a ledger without base information)."""
        return self.bases[-1][2] if self.bases else ""

    @property
    def base_root(self) -> str:
        return self.bases[-1][1] if self.bases else ROOT

    def in_epoch(self, base: str | None = None) -> "Ledger":
        """Only the observations of ``base`` (default: the current epoch): what pairs, fits noise, and is CUPED."""
        b = self.epoch if base is None else base
        return replace(self, obs=tuple(o for o in self.obs if o.base == b))

    def change_parents(self) -> dict[str, str]:
        """child -> parent over the ONE-change pairs (ρ is change-size specific: on the CRN base a small change
        shares ρ .92 of the seed effect with its parent, a node vs root .56). A port is many changes from its parent."""
        return {k: n.parent for k, n in self.nodes.items() if n.parent is not None and not n.port_of}

    def far_parents(self) -> dict[str, str]:
        """child -> a several-changes relative (a port's parent, else the grandparent): the ρ of a far pair."""
        out = {}
        for k, n in self.nodes.items():
            if n.parent is not None and n.port_of:
                out[k] = n.parent
            elif n.parent is not None and (p := self.nodes.get(n.parent)) is not None and p.parent is not None:
                out[k] = p.parent
        return out

    def port(self, node_id: str, base: str | None = None) -> str | None:
        """``node_id`` itself if it is in ``base`` (default: the current epoch), else its counterpart there: its latest
        port, or (``node_id`` a port) the node it was ported from / another port of it."""
        b = self.epoch if base is None else base
        n = self.nodes.get(node_id)
        if n is not None and n.base == b:
            return node_id
        if n is not None and n.port_of:
            return self.port(n.port_of, b)
        ports = [m for m in self.nodes.values() if m.port_of == node_id and m.base == b]
        return max(ports, key=lambda m: m.created_ts).id if ports else None

    def by_node(self) -> dict[str, list[Obs]]:
        out: dict[str, list[Obs]] = {}
        for o in self.obs:
            out.setdefault(o.node, []).append(o)
        return out

    def before(self, ts: float) -> "Ledger":
        """The ledger as it was known at ``ts``: observations available strictly before it."""
        return replace(self, obs=tuple(o for o in self.obs if o.ts < ts))

    def carrier_at(self, node_id: str, ts: float) -> str:
        known = {n for t, n in self.carriers if t <= ts}
        return nearest_carrier(node_id, {k: v.parent for k, v in self.nodes.items()}, known)


# -- pure derivations --------------------------------------------------------------------------------------------
def lane_of(strategy: str, notes: str) -> str:
    """explore / crossover / exploit / factorial (moonshot and revisit are explore; baseline is its own)."""
    if strategy == "baseline":
        return "baseline"
    if strategy == "design" or any(t in (notes or "") for t in FACTORIAL_TAGS):
        return "factorial"
    return strategy if strategy in ("crossover", "exploit") else "explore"


def nearest_carrier(node_id: str, parents: Mapping[str, str | None], carriers: set[str]) -> str:
    """The closest strict ancestor of ``node_id`` in ``carriers`` (root if none)."""
    cur, seen = parents.get(node_id), set()
    while cur is not None and cur not in seen:
        if cur in carriers:
            return cur
        seen.add(cur)
        cur = parents.get(cur)
    return ROOT


def bases_of(parents: Mapping[str, str | None], commits: Mapping[str, str]) -> dict[str, str]:
    """node -> the commit of its nearest parentless ancestor (its base root); '' for a broken lineage."""
    out: dict[str, str] = {}
    for nid in parents:
        path, cur = [], nid
        while cur in parents and cur not in out and cur not in path and parents[cur] is not None:
            path.append(cur)
            cur = parents[cur]
        if cur in out:
            base = out[cur]
        elif cur in parents and parents[cur] is None:
            base = out[cur] = commits.get(cur, "")
        else:                                    # a missing ancestor or a cycle
            base = ""
        for x in path:
            out[x] = base
    return out


def test_seeds(drawn: Iterable[tuple[str, Iterable[int]]], stage_seeds: Iterable[tuple[str, int]]) -> frozenset[int]:
    """TEST seeds: never-used draws for remint / audit / confirm, plus seeds only audit / remint ever ran."""
    out = {int(s) for purpose, seeds in drawn if purpose.split()[0] in TEST_PURPOSES for s in seeds}
    stages: dict[int, set[str]] = {}
    for stage, seed in stage_seeds:
        stages.setdefault(int(seed), set()).add(stage)
    return frozenset(out | {s for s, st in stages.items() if st <= TEST_ONLY_STAGES})


def val_loss_q25(output_dir: str | Path) -> float | None:
    """``val_total`` at the validation point nearest 25% of training (metrics.jsonl), None if unavailable."""
    path = Path(output_dir) / "metrics.jsonl"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    last, vals = 0, []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        last = max(last, int(row.get("step") or 0))
        if "val_total" in row:
            vals.append((int(row["step"]), float(row["val_total"])))
    if not vals or not last:
        return None
    return min(vals, key=lambda sv: abs(sv[0] - 0.25 * last))[1]


# -- the loader --------------------------------------------------------------------------------------------------
def _connect(db: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def load(db: str | Path, task_id: str, *, val_loss: bool = False) -> Ledger:
    """The ledger of ``task_id`` (read-only). ``val_loss``: read each done run's metrics.jsonl (slow, optional)."""
    con = _connect(db)
    try:
        rows = con.execute("SELECT * FROM runs WHERE task_id=? ORDER BY rowid", (task_id,)).fetchall()
        nrows = con.execute("SELECT id, parent_id, second_parent_id, strategy, family, created_ts, status, notes, "
                            "\"commit\" "
                            "FROM nodes WHERE task_id=? ORDER BY created_ts, rowid", (task_id,)).fetchall()
        events = con.execute("SELECT ts, event, fields FROM log WHERE task_id=? AND event IN "
                             "('run_finished', 'seeds_drawn', 'audit', 'incumbent', 'carrier', 'test_verdict') "
                             "ORDER BY id",
                             (task_id,)).fetchall()
    finally:
        con.close()
    finished: dict[str, float] = {}
    drawn, carriers, draws, incumbents, audits, verdicts = [], [(0.0, ROOT)], [], [], [], []
    for e in events:
        f = json.loads(e["fields"] or "{}")
        if e["event"] == "run_finished" and f.get("status") == "done":
            finished[f["run_id"]] = e["ts"]
        elif e["event"] == "seeds_drawn":
            drawn.append((str(f.get("purpose", "")), f.get("seeds", ())))
            draws.append((e["ts"], str(f.get("purpose", "")), tuple(int(x) for x in f.get("seeds", ()))))
        elif e["event"] == "carrier":            # a v3 TEST-pass incumbent switch (tick._sync_carriers), from its close
            carriers.append((float(f.get("at", e["ts"])), f["node_id"]))
        elif (e["event"] == "audit" and f.get("verdict") == "passed") or e["event"] == "incumbent":
            carriers.append((e["ts"], f["node_id"]))
            if e["event"] == "incumbent":
                incumbents.append((e["ts"], f["node_id"], f.get("previous") or ""))
        if e["event"] == "audit":
            audits.append((e["ts"], f["node_id"], str(f.get("verdict"))))
        elif e["event"] == "test_verdict" and f.get("node_id"):
            verdicts.append((float(e["ts"]), str(f["node_id"]), str(f.get("verdict") or "").lower()))
    parents = {n["id"]: n["parent_id"] for n in nrows}
    base = bases_of(parents, {n["id"]: n["commit"] or "" for n in nrows})
    held = {c for _, c in carriers}                   # the port of a carrier carries its role into the new epoch
    carriers += [(float(n["created_ts"]), n["id"]) for n in nrows if _port_of(n["notes"] or "") in held]
    roots = tuple(sorted((float(n["created_ts"]), n["id"], base[n["id"]]) for n in nrows if n["parent_id"] is None))
    runs = tuple(Run(r["id"], r["node_id"], int(r["seed"]), r["stage"], r["status"], float(r["gpu_h"] or 0.0),
                     float(r["submitted_ts"] or 0.0), finished.get(r["id"], float(r["submitted_ts"] or 0.0)),
                     r["output_dir"] or "", int(r["resumed"] or 0), base.get(r["node_id"], "")) for r in rows)
    tseeds = test_seeds(drawn, ((r.stage, r.seed) for r in runs))
    scores = {r["id"]: json.loads(r["scores"] or "{}") for r in rows}
    aggregate = {r["id"]: r["aggregate"] for r in rows}
    steps_key = _P.throughput()[0]
    obs = tuple(
        Obs(r.id, r.node, r.seed, r.stage, TEST if r.seed in tseeds else SELECT,
            {a: float(scores[r.id][k]) for a, k in PROBE_KEYS.items()}, float(aggregate[r.id]), r.gpu_h,
            r.done_ts, r.resumed, val_loss_q25(r.output_dir) if val_loss and r.output_dir else None, r.base,
            _steps(scores[r.id], steps_key))
        for r in runs if r.status == "done" and aggregate[r.id] is not None
        and all(k in scores[r.id] for k in PROBE_KEYS.values()))
    nodes = {n["id"]: Node(n["id"], n["parent_id"], n["second_parent_id"], n["strategy"] or "", n["family"] or "",
                           lane_of(n["strategy"] or "", n["notes"] or ""),
                           nearest_carrier(n["id"], parents, {c for t, c in carriers if t <= n["created_ts"]}),
                           float(n["created_ts"]), n["status"] or "", n["notes"] or "", base[n["id"]],
                           _port_of(n["notes"] or "")) for n in nrows}
    return Ledger(nodes, runs, obs, tseeds, tuple(carriers), tuple(draws), tuple(incumbents),
                  tuple(audits), roots, tuple(verdicts))


def _steps(scores: Mapping[str, object], key: str) -> float | None:
    """The run's step count under ``key`` (a positive finite number), else None."""
    v = scores.get(key) if key else None
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return x if x > 0 and x == x and x != float("inf") else None


def load_checked(db: str | Path, task_id: str, task: str | Path | None = None, *, val_loss: bool = False) -> Ledger:
    """:func:`load` for the CLIs (backtest / seeds): ``task`` (a task yaml) installs its ``loop.project`` first, so a
    non-nanopath ledger maps its own axes; a ledger with DONE runs but 0 observations raises (the axes did not match
    the installed project: pass ``--task``)."""
    if task:
        from loop.spec import TaskSpec, _parse, install_registry
        cfg, _ = _parse(TaskSpec.load(task))
        install_registry(cfg)                   # project + its noise floor, as load_context
    led = load(db, task_id, val_loss=val_loss)
    if not led.obs and any(r.status == "done" for r in led.runs):
        raise SystemExit(f"{task_id}: {sum(r.status == 'done' for r in led.runs)} done runs but 0 observations: their "
                         f"scores lack the installed project's axis keys {sorted(PROBE_KEYS.values())}"
                         + ("" if task else "; pass --task <the task yaml> to install its loop.project"))
    return led


def _port_of(notes: str) -> str:
    m = PORT_TAG.findall(notes)
    return m[-1] if m else ""


def summary(led: Ledger) -> dict[str, object]:
    """Counts to check a load against the ledger: runs by (stage, status), observations by (stage, pool)."""
    by_run: dict[str, int] = {}
    for r in led.runs:
        by_run[f"{r.stage}/{r.status}"] = by_run.get(f"{r.stage}/{r.status}", 0) + 1
    by_obs: dict[str, int] = {}
    for o in led.obs:
        by_obs[f"{o.stage}/{o.pool}"] = by_obs.get(f"{o.stage}/{o.pool}", 0) + 1
    lanes: dict[str, int] = {}
    for n in led.nodes.values():
        lanes[n.lane] = lanes.get(n.lane, 0) + 1
    return {"nodes": len(led.nodes), "runs": len(led.runs), "obs": len(led.obs), "runs_by_stage": by_run,
            "obs_by_pool": by_obs, "lanes": lanes, "carriers": [c for _, c in led.carriers],
            "bases": [f"{n}@{b[:7]}" for _, n, b in led.bases], "epoch": led.epoch[:7],
            "test_seeds": len(led.test_seeds)}
