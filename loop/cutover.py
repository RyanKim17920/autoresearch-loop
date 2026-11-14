"""loop.cutover — start a new base epoch in the ledger (docs/loop-core-v3.md §8 "Base epochs"; step 12).

:func:`ingest_rebase` writes the new base root (a node without a parent: its commit IS the base SHA), the ports of
old nodes onto it (``port_of=<old id>`` in their notes) and any other calibration nodes, and their finished runs read
from ``<dir>/<seed>/summary.json`` as BASELINE runs -- so the re-baseline acquisition schedules first after a base
change is already satisfied by the calibration. Additive and idempotent (existing nodes / runs are skipped).

``python -m loop.cutover --db <ledger COPY>`` ingests the crn-knobs calibration (crn-knobs @ 065b87f: root, the
a node port 6a6c1ef and its kde child 7065375, seeds 700001-700005). The live ledger is only written at the cutover.

:func:`port_node` (the tick's ``port`` hook) creates the port of an old-epoch node: its change over its nearest
ancestor that has a counterpart on the current base (its parent when that is ported -- the old incumbent -> its port --
an old base root -> the current base root) grafted onto that counterpart with the adapter's ``graft_port`` (config per
key via ``merge_config``, code 3-way). The port is a new node ``<old>@<base7>`` tagged ``port_of=<old>``; the old node is
never touched. A conflict (or a change the carrier already holds) is recorded as a REJECTED port node with the reason
-- so the node is not offered again -- and logged ``port_conflict``; the tick may ask ONE researcher session to port it."""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from loop.contracts import Node, NodeStatus, Operator, Run, RunStatus, Stage, Strategy
from loop.core.data import FACTORIAL_TAGS, OFFICIAL_WEIGHTS, PROBE_KEYS, Ledger


@dataclass(frozen=True)
class Arm:
    node: str
    commit: str
    parent: str | None          # None: the new base root
    family: str
    dir: str                    # <dir>/<seed>/summary.json
    port_of: str = ""
    note: str = ""


def load_arms(path: str | Path) -> tuple[Arm, ...]:
    """The new base's calibration arms from a JSON list of ``{node, commit, parent, family, dir[, port_of, note]}``."""
    rows = json.loads(Path(path).read_text())
    return tuple(Arm(r["node"], r["commit"], r.get("parent"), r.get("family", ""), r["dir"], r.get("port_of", ""),
                     r.get("note", "")) for r in rows)


def run_of(task_id: str, node: str, seed: int, summary: dict[str, Any], output_dir: str) -> Run:
    """A done BASELINE run from a nanopath ``summary.json`` (the six probe axes; aggregate = the official panel)."""
    scores = {k: float(summary[k]) for k in PROBE_KEYS.values()}
    panel = sum(OFFICIAL_WEIGHTS[a] * scores[k] for a, k in PROBE_KEYS.items())
    wall = float(summary.get("train_loop_wall_seconds", 0.0)) + float(summary.get("final_probe_wall_seconds", 0.0))
    return Run(id=Run.make_id(node, Stage.BASELINE, seed), task_id=task_id, node_id=node, seed=seed,
               stage=Stage.BASELINE, status=RunStatus.DONE, job_id=str(summary.get("slurm_job_id", "")),
               scores=scores, aggregate=panel, submitted_ts=None, gpu_h=wall / 3600.0, output_dir=output_dir)


def ingest_rebase(store: Any, arms: Sequence[Arm], seeds: Sequence[int], now: float) -> list[str]:
    """Write the arms' nodes and their finished runs (skipping what exists); returns what was done."""
    out = []
    for i, a in enumerate(arms):
        if store.get_node(a.node) is None:
            notes = " ".join(x for x in (f"port_of={a.port_of}" if a.port_of else "", a.note) if x)
            parent = store.get_node(a.parent) if a.parent else None
            store.put_node(Node(id=a.node, task_id=store.task_id, commit=a.commit, parent_id=a.parent,
                                strategy=Strategy.BASELINE if a.parent is None else Strategy.EXPLORE,
                                operator=None if a.parent is None else Operator.IMPROVE,
                                family=a.family or (parent.family if parent else "") or a.node,
                                mechanism_tag="", diff_hash=f"base-{a.commit[:12]}", status=NodeStatus.BASELINE,
                                created_ts=now + i, updated_ts=now + i, notes=notes))
            out.append(f"node {a.node} @ {a.commit[:7]} (parent {a.parent}{', port of ' + a.port_of if a.port_of else ''})")
        for s in seeds:
            path = Path(a.dir) / str(s) / "summary.json"
            run = run_of(store.task_id, a.node, s, json.loads(path.read_text()), str(path.parent))
            if store.get_run(run.id) is not None:
                continue
            store.put_run(dataclasses.replace(run, submitted_ts=now + len(arms)))
            store.log("run_finished", run_id=run.id, node_id=a.node, status="done", aggregate=run.aggregate)
            out.append(f"run {run.id}: panel {run.aggregate:.4f}")
    return out


# -- ports ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Ported:
    old: str
    node: str                   # the port node ('' when no anchor was found)
    parent: str                 # its parent on the current base (the anchor's counterpart)
    anchor: str                 # the old ancestor whose change-to-``old`` was grafted
    how: str                    # applied | contained | conflict: ... | no anchor
    commit: str = ""

    @property
    def ok(self) -> bool:
        return self.how == "applied"


def port_id(old: str, base: str) -> str:
    return f"{old}@{base[:7]}"


def anchor(led: Ledger, old: str) -> tuple[str, str] | None:
    """(old ancestor A, its counterpart on the current base): the nearest strict ancestor of ``old`` with a usable
    counterpart -- an old base root maps to the current base root, any other to its latest non-rejected port."""
    seen, cur = set(), led.nodes[old].parent if old in led.nodes else None
    while cur is not None and cur not in seen:
        seen.add(cur)
        n = led.nodes.get(cur)
        if n is None:
            return None
        if n.parent is None:
            return (cur, led.base_root) if n.base != led.epoch else None
        p = led.port(cur)
        if p is not None and p != cur and led.nodes[p].status != NodeStatus.REJECTED.value:
            return cur, p
        cur = n.parent
    return None


def _tags(notes: str) -> str:
    """The lane-bearing notes tags of the old node (a factorial-lane node stays factorial); nothing else is copied
    (a wave / port tag of the old node must not make the port look like one)."""
    return " ".join(t for t in (notes or "").split() if t.startswith(FACTORIAL_TAGS))


def port_node(store: Any, adapter: Any, led: Ledger, old: str, now: float, *, onto: str | None = None,
              tag: str = "", rel: str = "port_of", src: str | None = None, lane_tags: bool = True) -> Ported:
    """Graft ``old``'s change onto the current base (see the module docstring) and record the port node.
    ``onto``: graft ``old``'s OWN change (over its parent) onto that current-base node instead -- a factorial's diff
    knob on a carrier (``tag`` e.g. ``for_wave=<id>``; the id gets ``~<onto>`` when a plain port exists). ``rel``:
    the notes tag that names ``old`` (``port_of``; a prerequisite port: ``prereq_of``). ``src`` (with ``onto``): graft
    ``old``'s change over that ANCESTOR instead of over its parent (a re-root port: a ladder point's whole change over
    the old incumbent, its structural parent's included). ``lane_tags`` False: the old node's lane tags are NOT copied
    (a ported ladder point is a plain candidate on the new incumbent, never a point of the old node's ladder)."""
    base = led.epoch
    nid = port_id(old, base) if rel == "port_of" else f"{old}.pq@{base[:7]}"    # a prereq: never the port's id
    if onto is not None:
        nid = nid if store.get_node(nid) is None or store.get_node(nid).parent_id == onto else f"{nid}~{onto}"
    if (got := store.get_node(nid)) is not None:            # idempotent: an earlier tick made it
        return Ported(old, nid, got.parent_id or "", "", "exists", got.commit)
    a = anchor(led, old) if onto is None else (src, onto) if src else (led.nodes[old].parent, onto) \
        if led.nodes[old].parent else None
    if a is None:
        store.log("port_conflict", node_id=old, base=base, reason="no ancestor with a counterpart on the base")
        return Ported(old, "", "", "", "no anchor")
    src, onto = a
    o, s, t = store.get_node(old), store.get_node(src), store.get_node(onto)
    sha, how = adapter.graft_port(s.commit, o.commit, t.commit, f"port-{old}-{base[:7]}")
    ok = how == "applied" and bool(sha)
    reason = "" if ok else f"port {how}" if how.startswith("conflict") else f"port {how}: {onto} already holds it"
    # ``rel``: the tag naming the original -- ``prereq_of`` for a prerequisite port (v3 review K3: never a port_of, so
    # Ledger.port() never resolves the original to it)
    notes = " ".join(x for x in (f"{rel}={old} port_anchor={src}", tag, _tags(o.notes) if lane_tags else "",
                                 "" if ok else "port_failed") if x)
    store.put_node(Node(id=nid, task_id=store.task_id, commit=sha if ok else t.commit, parent_id=onto,
                        strategy=o.strategy, operator=o.operator, family=o.family, mechanism_tag=o.mechanism_tag,
                        diff_hash=f"port-{(sha or t.commit)[:12]}-{old}",
                        status=NodeStatus.QUEUED if ok else NodeStatus.REJECTED, hypothesis_id=o.hypothesis_id,
                        levers=o.levers, reject_reason=reason, created_ts=now, updated_ts=now, notes=notes))
    if ok:
        store.log("port", node_id=nid, port_of=old, anchor=src, onto=onto, commit=sha)
    else:
        store.log("port_conflict", node_id=old, port=nid, anchor=src, onto=onto, reason=reason)
    return Ported(old, nid, onto, src, how, sha if ok else "")


def port_brief(store: Any, adapter: Any, p: Ported) -> str:
    """The brief of the one researcher session a conflicting port may get: re-implement the change on the carrier."""
    o, s = store.get_node(p.old), store.get_node(p.anchor)
    try:
        diff = adapter.diff_commits(s.commit, o.commit)[:6000] if hasattr(adapter, "diff_commits") else ""
    except RuntimeError:
        diff = ""
    return (f"PORT {p.old} onto {p.parent} (the current base). Its change over {p.anchor} did not graft: {p.how}.\n"
            f"Re-implement exactly that change on {p.parent}'s code and config -- the same mechanism, nothing else; "
            f"where {p.parent} already moved a key, keep the change's intent (say how in the claim).\n"
            + (f"The change ({p.anchor}..{p.old}):\n```diff\n{diff}\n```" if diff else ""))


def main(argv: Sequence[str] | None = None) -> int:
    import time
    from loop.store import SqliteStore
    ap = argparse.ArgumentParser(prog="python -m loop.cutover", description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True, help="the ledger (a COPY until the cutover)")
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--arms", required=True, help="JSON list of the new base's calibration arms (see load_arms)")
    ap.add_argument("--seeds", required=True, help="comma-separated calibration seeds")
    args = ap.parse_args(argv)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    store = SqliteStore(args.db, args.task_id)
    try:
        for line in ingest_rebase(store, load_arms(args.arms), seeds, time.time()):
            print(line)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
