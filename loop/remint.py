"""loop.remint — SEED ROTATION (operator 2026-10-04: "the original point of overfitx was to prevent winner's curse by
switching seeds at times").

:func:`draw` hands out seeds no node has run at any stage and no earlier draw took (>= ``seeds.fresh_base``, above the
screen blocks), recording every draw (``seeds.drawn``). Each new held-out audit draws its own block (the audit seeds
201-204 decided every audit and every audited incumbent comparison until 2026-10-04: a node's one lucky seed-203 run
made it the best audited estimate for a day); the incumbent runs on that block too (stage ``remint``) so the
challenger / incumbent comparison stays paired on seeds neither was selected on."""

from __future__ import annotations

import time
from typing import Any

from loop import seeds as S
from loop.spec import LoopContext

K_DRAWN = "seeds.drawn"           # [{seeds, purpose, ts}]: every rotated draw, in order


def used_seeds(ctx: LoopContext, store: Any) -> set[int]:
    """Every seed any run used, every configured seed set, and every seed drawn before (in flight or not)."""
    p = ctx.cfg.seeds
    out = {int(r.seed) for r in store.runs()} | set(S.STACK_SEEDS)
    out |= {int(s) for name in ("screen", "promote", "confirm_pool", "audit", "calibration") for s in getattr(p, name)}
    return out | {int(s) for d in store.get_state(K_DRAWN) or [] for s in d.get("seeds") or ()}


def draw(ctx: LoopContext, store: Any, k: int, purpose: str, now: float | None = None) -> list[int]:
    """``k`` never-used seeds (the lowest >= ``seeds.fresh_base``), reserved at once (a draw in the same tick never
    repeats them) and logged."""
    used, out, s = used_seeds(ctx, store), [], int(ctx.cfg.seeds.fresh_base)
    while len(out) < k:
        if s not in used:
            out.append(s)
        s += 1
    now = time.time() if now is None else now
    store.set_state(K_DRAWN, [*(store.get_state(K_DRAWN) or []), {"seeds": out, "purpose": purpose, "ts": now}])
    store.log("seeds_drawn", seeds=out, purpose=purpose)
    return out



# ------------------------------------------------------------------------------------------------ confirm blocks
K_CBLOCKS = "seeds.confirm_blocks"   # [{parent, seeds, nodes}]: fresh confirm blocks, in draw order


def confirm_block(ctx: LoopContext, store: Any, node: Any, own: Any, now: float | None = None
                  ) -> tuple[list[int], int] | None:
    """(confirm seeds, sibling rank) of a CONFIRMING node with ``seeds.confirm_rotate``: a block of never-used seeds
    (:func:`draw`) its parent's next confirming children share, at most ``seeds.block_children`` of them (they share
    the parent's control runs; rank r starts on the block's r-th seed, so concurrent siblings confirm on different
    seeds). Every block is fresh: no candidate confirms on a seed hundreds of others confirmed on (live 2026-10-05:
    90 confirmations on 14 / 101-104, per-seed mean paired delta +0.0034 at 14 to -0.0025 at 104, permutation
    p=0.001; 4 of the 6 CONFIRMED confirmed on 14). None: the fixed ``confirm_pool`` (rotation off, or the node
    started confirming on it before rotation -- it finishes there)."""
    from loop.contracts import NodeStatus, Stage
    if not ctx.cfg.seeds.confirm_rotate or node.status is not NodeStatus.CONFIRMING:
        return None
    blocks = list(store.get_state(K_CBLOCKS) or [])
    for b in blocks:
        if node.id in b["nodes"]:
            return [int(s) for s in b["seeds"]], b["nodes"].index(node.id)
    if any(r.stage is Stage.CONFIRM for r in own):
        return None
    pid, cap = S.control_of(node)[0], ctx.cfg.seeds.block_children
    cur = next((b for b in reversed(blocks) if b["parent"] == pid), None)
    if cur is None or len(cur["nodes"]) >= cap:
        k = max(ctx.cfg.promotion.confirm_max_seeds, cap)
        cur = {"parent": pid, "seeds": draw(ctx, store, k, f"confirm {pid}", now), "nodes": []}
        blocks.append(cur)
    cur["nodes"].append(node.id)
    store.set_state(K_CBLOCKS, blocks)
    store.log("confirm_block", node_id=node.id, parent=pid, seeds=cur["seeds"])
    return [int(s) for s in cur["seeds"]], len(cur["nodes"]) - 1


def confirm_seeds(store: Any, parent: str) -> set[int]:
    """Every fresh confirm seed drawn for ``parent``'s children."""
    return {int(s) for b in store.get_state(K_CBLOCKS) or [] if b["parent"] == parent for s in b["seeds"]}


# ------------------------------------------------------------------------------------------------ the re-mint
K_REMINT = "driver.remint"        # loop.world.K_REMINT: {"rounds": [{inc, nodes, seeds, ts, why, closed?, result?}]}
REMINT_K = 2                      # audited challengers re-minted with the incumbent


def _gap(ctx: LoopContext, store: Any, c: str, inc: str) -> float:
    """The challenger's paired gain over the incumbent (fresh seeds; else the audit gates' means)."""
    from loop.world import K_AUDIT, audit_paired, fresh_paired
    if (g := fresh_paired(ctx, store, c, inc) or audit_paired(store, c, inc)) is not None:
        return g[0]
    gate = (store.get_state(K_AUDIT) or {}).get("gate") or {}
    return float((gate.get(c) or {}).get("mean") or 0.0) - float((gate.get(inc) or {}).get("mean") or 0.0)


def challengers(ctx: LoopContext, store: Any, snapshot: Any) -> list[str]:
    """The top ``REMINT_K`` audited CONFIRMED non-vetoed challengers of the incumbent by paired gain over it."""
    from loop.contracts import NodeStatus
    from loop.world import K_AUDIT, held_back
    inc, passed = snapshot.incumbent_id, set((store.get_state(K_AUDIT) or {}).get("passed") or ())
    held = held_back(store)
    out = [n.id for n in store.nodes(status=[NodeStatus.CONFIRMED]) if n.id != inc and n.id in passed
           and n.id not in held and (p := snapshot.nodes.get(n.id)) is not None and not p.vetoed]
    return sorted(out, key=lambda c: (-_gap(ctx, store, c, inc), c))[:REMINT_K]


def due(ctx: LoopContext, store: Any, snapshot: Any, now: float) -> str:
    """Why a re-mint round starts now ('' = none): the audited incumbent has none yet, or every
    ``promotion.remint_hours`` (one round at a time)."""
    from loop.contracts import ROOT_NODE_ID
    from loop.world import K_AUDIT
    hours, inc = ctx.cfg.promotion.remint_hours, getattr(snapshot, "incumbent_id", ROOT_NODE_ID)
    if hours <= 0 or inc == ROOT_NODE_ID or inc not in set((store.get_state(K_AUDIT) or {}).get("passed") or ()):
        return ""
    rounds = (store.get_state(K_REMINT) or {}).get("rounds") or []
    if rounds and not rounds[-1].get("closed"):
        return ""
    if not any(r["inc"] == inc for r in rounds):
        return "new incumbent"
    return "every %gh" % hours if now - float(rounds[-1]["ts"]) >= 3600.0 * hours else ""


def tick(ctx: LoopContext, store: Any, snapshot: Any, infra_retry: int,
         now: float | None = None) -> tuple[list[str], list[S.RunNeed]]:
    """Start a round when :func:`due` (the incumbent + :func:`challengers` on one never-used block); the runs the open
    round still needs; close it once they all ended (the incumbent rule re-applies at the next fit on those seeds:
    loop.world.reminted)."""
    from loop.contracts import Stage
    from loop.world import fresh_paired
    if snapshot is None:
        return [], []
    now, acts, needs = time.time() if now is None else now, [], []
    book = dict(store.get_state(K_REMINT) or {"rounds": []})
    rounds = [dict(r) for r in book.get("rounds") or []]
    if (why := due(ctx, store, snapshot, now)) and (cs := challengers(ctx, store, snapshot)):
        inc = snapshot.incumbent_id
        seeds = draw(ctx, store, len(ctx.cfg.seeds.audit), f"remint {inc}", now)
        rounds.append({"inc": inc, "nodes": [inc, *cs], "seeds": seeds, "ts": now, "why": why})
        store.log("remint_started", node_id=inc, challengers=cs, seeds=seeds, why=why)
        acts.append(f"re-mint ({why}): incumbent {inc} and challenger(s) {cs} on never-used seeds {seeds}")
    if rounds and not (cur := rounds[-1]).get("closed"):
        open_ = False
        for nid in cur["nodes"]:
            got, o = S.stage_needs(nid, Stage.REMINT, cur["seeds"], store.runs(node_id=nid), infra_retry, cur["inc"])
            needs += got
            open_ = open_ or o
        if not open_:
            res = {}
            for c in cur["nodes"][1:]:
                g = fresh_paired(ctx, store, c, cur["inc"], strict=True)
                res[c] = None if g is None else {"m": g[0], "sd": g[1] ** 0.5}
            cur.update(closed=now, result=res)
            store.log("remint_done", node_id=cur["inc"], seeds=cur["seeds"], result=res)
            acts.append(f"re-mint of {cur['inc']} on {cur['seeds']} done: " + ", ".join(
                f"{c} {r['m']:+.5f} ± {r['sd']:.5f}" if r else f"{c} unpaired" for c, r in res.items())
                + " (non-selection seeds); the incumbent rule re-applies at the next fit")
    if rounds != (book.get("rounds") or []):
        store.set_state(K_REMINT, {**book, "rounds": rounds})
    return acts, needs


def lines(store: Any) -> str:
    return "\n".join(f"re-mint {r['inc']} vs {r['nodes'][1:]} on {r['seeds']} ({r['why']})"
                     + (" done" if r.get("closed") else " running")
                     for r in (store.get_state(K_REMINT) or {}).get("rounds") or [])
