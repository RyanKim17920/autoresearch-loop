"""loop.regrad -- ONE-SHOT retro of the shrunk-estimate graduation rule (``promotion.shrunk_grad_mde``): every leg
factorial design, campaign and variant pass CLOSED in the last ``hours`` without a graduate is re-judged under the
current rule and graduates now when it passes (fresh confirmation decides, as for a new one). The driver runs it once
(state key :data:`K_RETRO`); ``dry`` only reports."""

from __future__ import annotations

import time
from typing import Any

from loop.contracts import PANEL

K_RETRO = "graduation.retro_shrunk"


def _sigma(ctx: Any, snapshot: Any) -> float:
    return float(snapshot.sigma.get(PANEL, ctx.cfg.panel_sd)) if snapshot is not None else ctx.cfg.panel_sd


def retro(ctx: Any, store: Any, adapter: Any, snapshot: Any, *, new_id: Any, now: float | None = None,
          hours: float = 48.0, dry: bool = False) -> list[str]:
    from loop import campaign as CP
    from loop import legs as LG
    from loop import variants as VR
    now = time.time() if now is None else now
    since, out, tag = now - hours * 3600.0, [], "retro (dry)" if dry else "retro"
    # leg factorials: each analysed design of a closed batch that sent no optimum
    bk = LG.book(store)
    for b in bk["batches"]:
        if float(b.get("closed") or 0) < since:
            continue
        for name, res in (b.get("result") or {}).items():
            if res.get("node") or name not in b.get("designs", {}):
                continue
            d = b["designs"][name]
            ana = LG.analyse(ctx, store, b, d, _sigma(ctx, snapshot))
            gate = LG.graduation(ctx, store, snapshot, d, ana, b["batch"])
            levels = LG.grad_levels(ana, gate)
            if not (gate["ok"] and gate["gain"] > 0 and levels):
                out.append(f"{tag}: leg factorial L{b['batch']} {name} on {b['carrier']}: still no graduate "
                           f"({gate['why']})")
                continue
            nid = "(dry)" if dry else LG._graduate(ctx, store, adapter, b, name, levels, gate["gain"], new_id, now)
            if not dry:
                res.update(node=nid, gate=gate, regraduated=now)
                store.set_state(LG.K_LEGS, bk)
                store.log("legs_regraduated", batch=b["batch"], design=name, node=nid, gate=gate, levels=levels)
            out.append(f"{tag}: leg factorial L{b['batch']} {name} on {b['carrier']}: node {nid} {levels} "
                       f"({gate['reason']}) -> confirm / audit on fresh seeds")
    # campaigns: closed without a graduate
    cb = CP.book(store)
    for c in cb["done"]:
        if float(c.get("closed") or 0) < since or c.get("tuned") or not c.get("best"):
            continue
        rs = CP.rows(ctx, store, snapshot, c)
        eff = VR._shrunk(rs) if rs else {}
        why, tuned, best, head = CP.graduate(ctx, store, snapshot, c, rs, eff, new_id, now, dry=dry)
        if tuned and not dry:
            c.update(why=why, tuned=tuned, regraduated=now)
            store.set_state(CP.K_CAMP, cb)
            store.log("campaign_regraduated", node_id=c["base"], campaign=c["id"], tuned=tuned, best=best, why=why)
        out.append(f"{tag}: {head} " + (why if tuned else f"still no graduate ({why})"))
    # variant passes: done without a graduate (closed = their last variants_done event)
    done_at = {e.get("node_id"): e["ts"] for e in store.events("variants_done")}
    vb = VR.book(store)
    for bid, cur in vb.items():
        if cur.get("status") != "done" or cur.get("tuned") or float(done_at.get(bid) or 0) < since:
            continue
        rows = VR._rows(ctx, store, cur, _sigma(ctx, snapshot))
        eff = VR._shrunk(rows)
        if not eff or not cur.get("best"):
            continue
        probe = dict(cur)
        VR._finish(ctx, store, snapshot, probe, rows, eff, new_id, now, dry=True)
        if not probe["tuned"] or dry:
            out.append(f"{tag}: variants for {bid}: best {cur['best']} " + (
                f"-> graduate (dry){probe['why']}" if probe["tuned"] else "still no graduate"))
            continue
        line = VR._finish(ctx, store, snapshot, cur, rows, eff, new_id, now)
        store.set_state(VR.K_VAR, vb)
        store.log("variants_regraduated", node_id=bid, tuned=cur["tuned"], why=cur["why"])
        out.append(f"{tag}: {line}")
    return out


def once(ctx: Any, store: Any, adapter: Any, snapshot: Any, *, new_id: Any, now: float | None = None) -> list[str]:
    """Run :func:`retro` once per state (the driver calls this every tick; a no-op after the first)."""
    if snapshot is None or store.get_state(K_RETRO) or not getattr(ctx.cfg.promotion, "shrunk_grad_mde", 0.0):
        return []
    now = time.time() if now is None else now
    try:
        out = retro(ctx, store, adapter, snapshot, new_id=new_id, now=now)
    except Exception as exc:  # noqa: BLE001 -- a one-shot never takes the loop down
        out = [f"graduation retro error: {exc!r}"[:300]]
    store.set_state(K_RETRO, {"ran": now, "lines": out})
    store.log("graduation_retro", lines=out)
    return [x for x in out if "still no graduate" not in x]
