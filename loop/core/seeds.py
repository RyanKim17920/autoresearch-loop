"""loop.core.seeds — seed pools and their enforcement (docs/loop-core-v3.md §2 "Seed pools"; LOCKED).

A :class:`SeedBlock` is a set of seeds drawn for one purpose and the nodes allowed to run on them.
- ``SELECT``: screening / CRN seeds (below ``fresh_base``, or a factorial wave's own block).
- ``TEST``: :func:`draw_test` -- seeds >= ``fresh_base`` that no run and no block ever used. Only the block's
  nodes (candidate, incumbent) may run on them: :func:`check_submit` raises :class:`SeedPoolViolation`.
A seed belongs to the LATEST block (by ``drawn_ts``) that contains it: on a PASS the test seeds are re-issued as a
SELECT block (the new incumbent's CRN seeds), which releases them to its children.
Test data never enters screening until its block closes (:func:`screening_obs`).

:func:`reconstruct` rebuilds the historical blocks from the ledger (the amended §8 mapping); :func:`migrate`
writes them into the additive ``seed_blocks`` table. ``python -m loop.core.seeds --db <copy> [--write]``."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from loop.core.data import ROOT, SELECT, TEST, TEST_ONLY_STAGES, TEST_PURPOSES, Ledger, Obs, load, load_checked

FRESH_BASE = 100_000
TERMINAL = frozenset({"done", "crash", "void"})


class SeedPoolViolation(RuntimeError):
    """A run on a TEST seed by a node outside that seed's block."""


@dataclass(frozen=True)
class SeedBlock:
    id: str
    pool: str                       # SELECT | TEST
    seeds: tuple[int, ...]
    node_ids: tuple[str, ...]       # a test block: (candidate, incumbent); historical blocks: candidates, then controls
    purpose: str                    # "test a node", "extend", "wave W3", historical "confirm a node", "audit a node", ...
    drawn_ts: float
    closed_ts: float | None = None
    #: the FROZEN judgements of a closed TEST block (``status.record``: node, vs, kind, e, n, mean, sd, rho, df, near,
    #: ...), decided once when it closed with that moment's noise; None = not frozen yet (``status.freeze``)
    verdicts: tuple[Mapping[str, Any], ...] | None = field(default=None, compare=False, hash=False)

    @property
    def open(self) -> bool:
        return self.closed_ts is None


def owner(seed: int, blocks: Iterable[SeedBlock], ts: float | None = None) -> SeedBlock | None:
    """The block a seed belongs to (at ``ts``, default now): the latest drawn one that contains it."""
    have = [b for b in blocks if seed in b.seeds and (ts is None or b.drawn_ts <= ts)]
    return max(have, key=lambda b: (b.drawn_ts, b.id)) if have else None


def pool_of(seed: int, blocks: Iterable[SeedBlock], ts: float | None = None) -> str:
    b = owner(seed, blocks, ts)
    return b.pool if b is not None else SELECT


def check_submit(node_id: str, seed: int, blocks: Iterable[SeedBlock], ts: float | None = None) -> None:
    """Raise :class:`SeedPoolViolation` if ``node_id`` may not run on ``seed`` (``runner._submit`` calls this)."""
    b = owner(seed, blocks, ts)
    if b is not None and b.pool == TEST and node_id not in b.node_ids:
        raise SeedPoolViolation(f"seed {seed} belongs to TEST block {b.id} ({b.purpose}: {', '.join(b.node_ids)}); "
                                f"{node_id} may not run on it")


def draw(k: int, used: Iterable[int], blocks: Iterable[SeedBlock] = (), fresh_base: int = FRESH_BASE) -> tuple[int, ...]:
    """The ``k`` smallest seeds >= ``fresh_base`` that no run used and no block holds (never-used draws)."""
    taken = set(used) | {s for b in blocks for s in b.seeds}
    out, s = [], fresh_base
    while len(out) < k:
        if s not in taken:
            out.append(s)
        s += 1
    return tuple(out)


def draw_test(k: int, used: Iterable[int], blocks: Iterable[SeedBlock] = (), fresh_base: int = FRESH_BASE) -> tuple[int, ...]:
    return draw(k, used, blocks, fresh_base)


def screening_obs(obs: Iterable[Obs], blocks: Sequence[SeedBlock]) -> list[Obs]:
    """Observations the screening posterior may use: none on a seed of a still-open TEST block."""
    hidden = {s for b in blocks if b.pool == TEST and b.open for s in b.seeds}
    return [o for o in obs if o.seed not in hidden]


# -- migration from the ledger ----------------------------------------------------------------------------------
CONTROL_STAGES = ("parent", "baseline", "remint")
VOID_NO_RUNS = "void: drawn, never run"      # a historical block none of whose seeds any run used: closed, not left open


def _block(bid: str, pool: str, seeds: Sequence[int], purpose: str, led: Ledger, ts: float | None = None,
           nodes: Sequence[str] | None = None) -> SeedBlock:
    """A historical block: the nodes that ran on its seeds (``nodes``: only these), candidates first, then the
    controls (control stages, and root). Closed when every run is terminal; a block NO run ever used (a draw whose
    runs were never submitted) is VOID -- closed at its draw, purpose ``<purpose> (void: drawn, never run)`` -- never
    an open block without members (its TEST seeds stay owned by it, so still never reused)."""
    runs = [r for r in led.runs if r.seed in set(seeds) and (nodes is None or r.node in nodes)]
    ctrl = {r.node for r in runs if r.stage in CONTROL_STAGES or r.node == ROOT}
    cands = sorted({r.node for r in runs} - ctrl)
    drawn = ts if ts is not None else min(r.submitted_ts for r in runs if r.node in cands or not cands)
    if not runs:
        return SeedBlock(bid, pool, tuple(sorted(seeds)), (), f"{purpose} ({VOID_NO_RUNS})", drawn, drawn)
    closed = max(r.done_ts for r in runs) if all(r.status in TERMINAL for r in runs) else None
    return SeedBlock(bid, pool, tuple(sorted(seeds)), tuple(cands + sorted(ctrl)), purpose, drawn, closed)


def reconstruct(led: Ledger) -> list[SeedBlock]:
    """The historical blocks (§8, amended): every ``seeds_drawn`` block (TEST for remint / audit / confirm draws,
    SELECT for the ``legs`` factorial draws) plus, on the fixed held-out seeds that only audit / remint ever ran,
    one TEST block per audited node (its seeds, root as control: 201-204, the one-off 990000) and one for the
    root-only leftovers (990001)."""
    out = [_block(f"hist-draw-{i:02d}", TEST if p.split()[0] in TEST_PURPOSES else SELECT, seeds, p, led, ts)
           for i, (ts, p, seeds) in enumerate(led.draws)]
    drawn = {s for b in out for s in b.seeds}
    stages: dict[int, set[str]] = {}
    for r in led.runs:
        stages.setdefault(r.seed, set()).add(r.stage)
    fixed = {s for s, st in stages.items() if st <= TEST_ONLY_STAGES and s not in drawn}
    audited: dict[str, set[int]] = {}
    for r in led.runs:
        if r.seed in fixed:
            audited.setdefault(r.node, set()).add(r.seed)
    root_only = audited.pop(ROOT, set()) - {s for ss in audited.values() for s in ss}
    for node, seeds in audited.items():
        out.append(_block(f"hist-audit-{node}", TEST, sorted(seeds), f"audit {node}", led, nodes=(node, ROOT)))
    if root_only:
        out.append(_block("hist-audit-root", TEST, sorted(root_only), "audit root", led, nodes=(ROOT,)))
    return sorted(out, key=lambda b: (b.drawn_ts, b.id))


def violations(led: Ledger, blocks: Sequence[SeedBlock]) -> list[str]:
    """Historical runs that :func:`check_submit` would have refused (expected: none)."""
    out = []
    for r in led.runs:
        try:
            check_submit(r.node, r.seed, blocks, r.submitted_ts + 1e-6)
        except SeedPoolViolation as exc:
            out.append(f"{r.id}: {exc}")
    return out


def migrate(store, led: Ledger) -> int:
    """Write the reconstructed blocks into ``seed_blocks`` (only when the task has none); returns rows written. A VOID
    block (no run ever used its seeds) is logged as ``seed_block_void`` with the reason."""
    if store.seed_blocks():
        return 0
    blocks = reconstruct(led)
    for b in blocks:
        store.put_seed_block(b)
        if b.purpose.endswith(f"({VOID_NO_RUNS})"):
            store.log("seed_block_void", block=b.id, pool=b.pool, seeds=list(b.seeds), purpose=b.purpose,
                      reason="historical block reconstructed with no runs on its seeds: closed as VOID at its draw")
    return len(blocks)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loop.core.seeds", description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--task-id", default="nanopath-v2-rescore")
    ap.add_argument("--task", default=None, help="the task yaml: installs its loop.project (axes / probe keys) before the ledger is read")
    ap.add_argument("--write", action="store_true", help="write the blocks into seed_blocks (a COPY only)")
    args = ap.parse_args(argv)
    led = load_checked(args.db, args.task_id, args.task)
    blocks = reconstruct(led)
    test = {s for b in blocks if b.pool == TEST for s in b.seeds}
    print(json.dumps({"blocks": [f"{b.id} {b.pool} {b.purpose} seeds={list(b.seeds)} nodes={list(b.node_ids)}"
                                 for b in blocks],
                      "test_seeds_match_data": test == set(led.test_seeds), "violations": violations(led, blocks)},
                     indent=1))
    if args.write:
        from loop.store import SqliteStore
        store = SqliteStore(args.db, args.task_id)
        try:
            print("written:", migrate(store, led))
        finally:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
