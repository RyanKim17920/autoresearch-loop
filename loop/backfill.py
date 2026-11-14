"""loop.backfill -- AUTOMATIC BACKFILL on a VERDICT EPOCH (operator 2026-10-06: "systemic fixes so these issues
don't recur, not one-off manual backfills").

A closed node's verdict was made by the rules of its day. Whenever a measurement / decision rule changes, a
:data:`EPOCHS` entry is appended IN THE SAME COMMIT (what changed + the verdict kinds it decided: ``screen`` =
SCREENED_OUT, ``confirm`` = NOT_CONFIRMED, ``void`` = VOID). The loop stamps each epoch the first time it sees it
(``backfill.seen``); every node of an affected kind that CLOSED BEFORE that stamp is re-evaluated under the current
rule (the live world model: the calibrated EB posterior on all its seeds) and, when that rule would NOT close it
now, queued for the CHEAPEST corrective action:

* ``promote`` -- a screen-out the current screen gate passes (loop.seeds.promote_ok): re-opened to SCREENING, it is
  re-judged at once and goes to confirmation (no new screen seed);
* ``early_confirm`` -- a screen-out the current gate routes to EARLY CONFIRMATION (loop.seeds.early_confirm_bf):
  re-opened to SCREENING, re-judged, it goes to fresh confirm seeds;
* ``confirm+1`` -- a NOT_CONFIRMED node on the incumbent's lineage whose E could still reach the bar with one more
  fresh confirm seed (loop.seeds.reachable): re-opened to CONFIRMING with one confirm seed past its plan;
* ``port`` -- a NOT_CONFIRMED node built off the incumbent's lineage that passes the current screen gate on all its
  seeds, or a root child that does with ``E[max(own gain, 0)] >= backfill.min_gain`` once root is not the
  incumbent: its change grafted onto the incumbent (the adapter's ``graft_port``: config merged per key -- a key the
  incumbent already holds equal / further the same way is kept --, code 3-way from the node's parent) as a new node
  screened against the incumbent (``backfill_of=``); a change the incumbent already holds is booked
  ``already_in_incumbent``, a genuine conflict ``port_conflict`` (with the conflicting files / keys) and handed to the
  researcher as ONE low-priority port side request (:func:`port_requests`); neither is ever retried;
* ``rerun`` -- a VOID node (an infra / partner void, an outage) with ``E[gain+] >= backfill.min_gain``: a fresh copy
  screens again (``retry_of=``).

A KNOB SHIFT (loop.shift: the incumbent changed and with it some config keys) is a dynamic epoch numbered past
these (:func:`shift_items`): the closed nulls it may have decided (loop.shift.affected) are ported onto the new
incumbent, best E[gain+] first round-robin over mechanism types, at most ``backfill.shift_max`` per shift, never one
the static epochs already port.

Value per run (E[gain+] / the action's runs) ranks the queue; at most ``backfill.per_tick`` start per tick and
``backfill.max_active`` are in flight. Each node is backfilled once per epoch (``backfill.book``); every start is
logged (``backfill``) and the queue is shown in ``loop status``, the heartbeat and ``--dry-run``. Voided dose ladders
need no entry: loop.tune.reopen_void_ladders re-opens them every tick."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from loop.contracts import PANEL, ROOT_NODE_ID, Node, NodeStatus, RunStatus, Stage, design_point

K_SEEN = "backfill.seen"        # {epoch: ts the loop first saw it}
K_BOOK = "backfill.book"        # {node id: {"epoch", "action", "ts", "target", "value"}}

KINDS = {"screen": NodeStatus.SCREENED_OUT, "confirm": NodeStatus.NOT_CONFIRMED, "void": NodeStatus.VOID}
COST = {"promote": 3, "early_confirm": 3, "confirm+1": 1, "port": 5, "rerun": 2,     # runs (confirm seeds included)
        "campaign": 16}


@dataclass(frozen=True)
class Epoch:
    n: int
    what: str
    kinds: tuple[str, ...]


#: append-only: a rule change that could have decided a closed node differently adds one here, in its own commit
EPOCHS: tuple[Epoch, ...] = (
    Epoch(1, "inherited-carrier bias: the screen gate and P(U>U_inc) credited a child with its carrier's lead; own "
             "increments now over the nearest contained ancestor", ("screen", "confirm")),
    Epoch(2, "small delta_1: the confirm alternative is sigma_ucb*sqrt(2 ln(1/alpha)/n), frozen before confirm data",
             ("confirm",)),
    Epoch(3, "outages: a backend_unavailable request is no attempt; outage-voided runs", ("void",)),
    Epoch(4, "fixed confirm-pool seed luck: confirming siblings shared confirm seeds and their control runs (one seed "
             "read low for every child of one carrier); confirm rotation", ("confirm",)),
    Epoch(5, "EB pessimism: one Gaussian slab scored +0.004 carrier screens P(gain>0) 0.01-0.08; grid "
             "NPMLE on honest data + the value-of-information promote gate", ("screen", "confirm")),
    Epoch(6, "early confirmation replaced the screen extension: an extension-eligible screen goes to 3 fresh confirm "
             "seeds instead of 2 more screen seeds and 1 confirm seed", ("screen",)),
    Epoch(7, "difficult-tune campaigns: init / architecture / objective changes were judged at their parent's "
             "hyperparameters on <= 2 runs; now a computed hyperparameter campaign (loop.campaign) -- the "
             "campaign.retro_max best closed in the campaign.retro_hours before it", ("campaign",)),
)


@dataclass(frozen=True)
class Item:
    node_id: str
    action: str
    value: float          # E[max(own gain, 0)] under the current posterior
    epoch: int
    target: str = ""      # port: the incumbent it is grafted onto
    why: str = ""

    @property
    def score(self) -> float:
        return self.value / COST[self.action]

    def line(self) -> str:
        if self.why.startswith("retest after knob shift"):
            return (f"{self.node_id} {self.why} -- {self.action} onto {self.target} E[gain+]={self.value:.4f} "
                    f"(epoch {self.epoch})")
        return (f"{self.node_id} {self.action}{f' onto {self.target}' if self.target else ''} "
                f"E[gain+]={self.value:.4f} per-run={self.score:.5f} (epoch {self.epoch}; {self.why})")


def lineage(store: Any, nid: str) -> list[str]:
    out = []
    while nid and nid != ROOT_NODE_ID and (n := store.get_node(nid)) is not None:
        out.append(nid)
        nid = n.parent_id or ROOT_NODE_ID
    return out


def seen(store: Any, now: float) -> dict[int, float]:
    """epoch -> when the loop first saw it (stamped now for a new one)."""
    book = {int(k): float(v) for k, v in (store.get_state(K_SEEN) or {}).items()}
    if any(e.n not in book for e in EPOCHS):
        book.update({e.n: now for e in EPOCHS if e.n not in book})
        store.set_state(K_SEEN, {str(k): v for k, v in book.items()})
    return book


def _screen_ok(store: Any, node: Node) -> bool:
    """Every screen run of the node DONE (a futility stop cancelled one: re-opening it would void the node)."""
    runs = [r for r in store.runs(node_id=node.id) if r.stage is Stage.SCREEN]
    return bool(runs) and all(r.status is RunStatus.DONE for r in runs)


def live_e(ctx: Any, store: Any, snapshot: Any) -> Callable[[Node], float]:
    """A closed node's confirm E under the CURRENT ledger: the same loop.world confirm_evalue the confirm step
    (loop.seeds.step) judges it by -- its stored ``confirm_e`` is from its close, under that day's rules (a node:
    stored 1.14 booked a confirm+1 the step killed at once on the live 0.541). The corrected ledger is read once."""
    from loop.world import ComputedWorldModel, noise
    world, led = ComputedWorldModel(ctx), []

    def e(node: Node) -> float:
        if not led:
            led.append(noise.correct(ctx, noise.read_ledger(ctx, store), snapshot.sigma))
        return world.confirm_evalue(snapshot, node.id, store, led[0])[0]
    return e


def plan(ctx: Any, store: Any, snapshot: Any, now: float) -> list[Item]:
    """The ranked queue: every closed node an epoch it predates invalidated, promising under the current rule, not
    yet backfilled for that epoch, with the action that would test it."""
    from loop import seeds as S
    from loop.select import built_from
    from loop.world.evalues import CONFIRM_KEY, sigma_ucb
    from loop.world.noise import STATE_KEY as NOISE_KEY
    cfg, promo = ctx.cfg.backfill, ctx.cfg.promotion
    if snapshot is None:
        return []
    stamps, book = seen(store, now), store.get_state(K_BOOK) or {}
    inc = snapshot.incumbent_id or ROOT_NODE_ID
    line = set(lineage(store, inc))
    inside = built_from(store, inc) | {inc} if inc != ROOT_NODE_ID else set()
    redone = {m.group(1) for n in store.nodes() for m in [re.search(r"\b(?:backfill_of|retry_of)=(\S+)",
                                                                    n.notes or "")] if m}
    frozen, evalue = store.get_state(CONFIRM_KEY) or {}, live_e(ctx, store, snapshot)
    row = (store.get_state(NOISE_KEY) or {}).get(PANEL)     # the confirm sigma loop.driver hands loop.seeds.step
    sig = sigma_ucb(row["sigma"], row["df_eff"]) if row else ctx.cfg.panel_sd
    out = []
    for node in store.nodes(status=list(KINDS.values())):
        kind = next(k for k, s in KINDS.items() if s is node.status)
        eps = [e.n for e in EPOCHS if kind in e.kinds and node.updated_ts < stamps[e.n]]
        post = snapshot.nodes.get(node.id)
        b = book.get(node.id) or {}
        if b.get("action") == "port" and not b.get("node") and "outcome" not in b:
            b = {}          # booked by the line-level graft (before graft_port): its failure was no verdict, port again
        if not eps or post is None or post.gain_plus is None or node.id in inside or node.id in redone \
                or int(b.get("epoch", 0)) >= max(eps) or design_point(node) \
                or node.strategy.value == "design" or re.search(r"\b(?:legs_of|seed1|ladder)=", node.notes or ""):
            continue
        if post.vetoed:
            continue
        pid, action, target, why = node.parent_id or ROOT_NODE_ID, "", "", ""
        raw = (snapshot.eb_raw.get(node.id) or (None, None))[1]
        if pid == ROOT_NODE_ID and inc != ROOT_NODE_ID and kind != "void":
            if S.promote_ok(post, promo) and post.gain_plus >= cfg.min_gain:
                action, target, why = "port", inc, "a root child passing the current gate: tested on the incumbent"
        elif kind == "screen":
            if not _screen_ok(store, node):
                continue
            if S.promote_ok(post, promo):
                action, why = "promote", f"passes the current gate (P(gain>0)={post.p_gain:.2f})"
            elif (bf := S.early_confirm_bf(post, ctx.cfg, raw, store.runs(node_id=node.id), sig)) is not None:
                action, why = "early_confirm", f"early-confirmation eligible (BF={bf:.2f})"
        elif kind == "confirm":
            if pid not in line | {ROOT_NODE_ID}:
                if S.promote_ok(post, promo):
                    action, target, why = "port", inc, (f"built on {pid}, off the incumbent's lineage; P(gain>0)="
                                                        f"{post.p_gain:.2f} on all its seeds")
            elif node.id in frozen and S.reachable(e := evalue(node), frozen[node.id], 1, promo.alpha_confirm):
                action, why = "confirm+1", f"E={e:.3g} can reach the bar with one more seed"
        elif post.gain_plus >= cfg.min_gain:
            action, why = "rerun", "voided"
        if action == "port" and (inc == ROOT_NODE_ID or pid == inc):
            continue
        if action:
            out.append(Item(node.id, action, post.gain_plus, max(eps), target, why))
    have = {i.node_id for i in out}
    out += [i for i in shift_items(ctx, store, snapshot, book, redone) if i.node_id not in have]
    return sorted(out + campaign_items(ctx, store, snapshot, stamps, book), key=lambda i: (-i.score, i.node_id))


def shift_items(ctx: Any, store: Any, snapshot: Any, book: Mapping[str, Any], redone: set[str]) -> list[Item]:
    """The current knob shift's queue (loop.shift): closed nulls from before it that it may have decided, each a
    ``port`` onto the new incumbent; best E[gain+] first, round-robin over mechanism types (diversity), at most
    ``backfill.shift_max`` over the shift. Batch points, nodes in / on the new incumbent and nodes already re-done
    (``redone``: a port / retry of them exists) are left out."""
    from loop import shift as SH
    from loop.novelty import node_type
    from loop.select import built_from
    ev, cap = SH.current(store, snapshot), ctx.cfg.backfill.shift_max
    if ev is None or cap <= 0:
        return []
    room = cap - sum(int(v.get("epoch", 0)) == ev["n"] for v in book.values())
    inc = ev["to"]
    skip = built_from(store, inc) | {inc} | redone
    by: dict[str, list[tuple[float, str, str]]] = {}
    for node in store.nodes(status=[NodeStatus.SCREENED_OUT, NodeStatus.NOT_CONFIRMED]) if room > 0 else ():
        post = snapshot.nodes.get(node.id)
        if node.updated_ts >= ev["ts"] or node.id in skip or node.parent_id == inc or design_point(node) \
                or node.strategy.value == "design" or SH._BATCH.search(node.notes or "") \
                or re.search(r"\bladder=", node.notes or "") or int((book.get(node.id) or {}).get("epoch", 0)) >= ev["n"] \
                or post is None or post.gain_plus is None or post.vetoed:
            continue
        if why := SH.affected(ctx, store, snapshot, ev, node):
            by.setdefault(node_type(node) or node.family or "", []).append((post.gain_plus, node.id, why))
    lanes = [sorted(v, key=lambda x: (-x[0], x[1])) for v in by.values()]
    ranked = sorted(((i, *x) for lane in lanes for i, x in enumerate(lane)), key=lambda x: (x[0], -x[1], x[2]))
    return [Item(nid, "port", v, ev["n"], inc, f"retest after knob shift ({SH.label(ev)}): {why}")
            for _, v, nid, why in ranked[:max(0, room)]]


def campaign_items(ctx: Any, store: Any, snapshot: Any, stamps: Mapping[int, float],
                   book: Mapping[str, Any]) -> list[Item]:
    """The ``campaign`` epoch's queue: difficult-to-tune nodes (loop.campaign.why) closed in the ``campaign.retro_hours``
    before its stamp on <= ``campaign.retro_max_runs`` scored runs, declared / init swaps then best E[gain+] first, at
    most ``campaign.retro_max`` over the epoch; each is queued for a campaign (it is never re-opened itself)."""
    from loop import campaign as CP
    cc, eps = getattr(ctx.cfg, "campaign", None), [e.n for e in EPOCHS if "campaign" in e.kinds]
    if cc is None or not cc.enabled or not eps:
        return []
    ep = max(eps)
    room = cc.retro_max - sum(v.get("action") == "campaign" and int(v.get("epoch", 0)) == ep for v in book.values())
    done, out = CP.campaigned(store), []
    for node in store.nodes(status=list(CP.CLOSES)) if room > 0 else ():
        if not stamps[ep] - 3600.0 * cc.retro_hours <= node.updated_ts < stamps[ep] or node.id in done \
                or int((book.get(node.id) or {}).get("epoch", 0)) >= ep:
            continue
        runs = sum(r.status is RunStatus.DONE and bool(r.scores) for r in store.runs(node_id=node.id))
        if runs > cc.retro_max_runs or not (w := CP.why(ctx, store, node)) or not CP.startable(ctx, store, node):
            continue
        post = snapshot.nodes.get(node.id)
        out.append(Item(node.id, "campaign", (post.gain_plus or 0.0) if post is not None else 0.0, ep,
                        why=f"{w}; closed on {runs} run(s) at its parent's hyperparameters"))
    return sorted(out, key=lambda i: (CP.rank(i.why), -i.value, i.node_id))[:max(0, room)]   # init swaps first


def active(store: Any) -> list[str]:
    """Backfills still running: the re-opened node / the new node still in an active status."""
    from loop.contracts import ACTIVE_STATUSES
    live = {n.id for n in store.nodes(status=ACTIVE_STATUSES)}
    return [k for k, v in (store.get_state(K_BOOK) or {}).items() if (v.get("node") or k) in live]


def tick(ctx: Any, store: Any, adapter: Any, snapshot: Any, *, new_id: Callable[[], str], now: float,
         reopen: Callable[[Node, NodeStatus, str], Any]) -> list[str]:
    """Start the best queued backfills within the rate limits; ``reopen(node, status, reason)`` moves a closed node
    (the driver's transition, logged). Returns action lines."""
    cfg = ctx.cfg.backfill
    if not cfg.enabled or snapshot is None:
        return []
    room = min(cfg.per_tick, cfg.max_active - len(active(store)))
    acts, book = [], dict(store.get_state(K_BOOK) or {})
    from loop.campaign import rank as CP_rank
    q = plan(ctx, store, snapshot, now)          # a campaign item only queues (loop.campaign has its own caps)
    for item in [i for i in q if i.action != "campaign"][:max(0, room)] \
            + sorted((i for i in q if i.action == "campaign"),
                     key=lambda i: (CP_rank(i.why), -i.value, i.node_id))[:cfg.per_tick]:       # init swaps first
        node, made, extra, extra_book = store.get_node(item.node_id), item.node_id, 0, {}
        if item.action in ("promote", "early_confirm"):
            reopen(node, NodeStatus.SCREENING, f"backfill epoch {item.epoch}: {item.action} ({item.why})")
        elif item.action == "confirm+1":
            extra = int((book.get(node.id) or {}).get("extra", 0)) + 1
            reopen(node, NodeStatus.CONFIRMING, f"backfill epoch {item.epoch}: one more confirm seed ({item.why})")
        elif item.action == "port":
            made, how = port(ctx, store, adapter, node, item.target, new_id(), now)
            if not made:
                extra_book = {"outcome": "already_in_incumbent" if how == "contained" else "port_conflict",
                              **({"conflict": how.partition(": ")[2][:300]} if how != "contained" else {})}
        elif item.action == "campaign":            # queued for loop.campaign: the node itself stays closed
            from loop import campaign as CP
            CP.enqueue(store, node.id, f"backfill epoch {item.epoch}: {item.why}")
        else:
            made = new_id()
            store.put_node(dataclasses.replace(node, id=made, status=NodeStatus.SCREENING, reject_reason="",
                                               created_ts=now, updated_ts=now, notes=f"{node.notes or ''} retry_of={node.id}"))
            store.set_state(f"config_diff:{made}", dict(store.get_state(f"config_diff:{node.id}") or {}))
            store.log("node_transition", node_id=made, frm="new", to=NodeStatus.SCREENING.value,
                      reason=f"backfill epoch {item.epoch}: re-run of void {node.id}")
        book[node.id] = {"epoch": item.epoch, "action": item.action, "ts": now, "node": made or "",
                         "target": item.target, "value": round(item.value, 6), **({"extra": extra} if extra else {}),
                         **extra_book}
        store.set_state(K_BOOK, book)
        store.log("backfill", node_id=node.id, action=item.action, epoch=item.epoch, value=item.value,
                  target=item.target, made=made or "", why=item.why, **extra_book)
        acts.append(f"backfill: {item.line()}" + ("" if made else f" -- {extra_book.get('outcome')}"
                                                  + (f" ({extra_book['conflict']})" if extra_book.get("conflict") else "")))
    return acts


def port(ctx: Any, store: Any, adapter: Any, node: Node, onto_id: str, gid: str, now: float) -> tuple[str, str]:
    """``node``'s change grafted onto ``onto_id`` (the adapter's ``graft_port``: config merged per key, code 3-way
    from the node's parent) as a new SCREENING node of the same strategy / family / mechanism (``backfill_of=``):
    (its id, 'applied'); ('', 'contained') when the incumbent already holds the change; ('', 'conflict: ...')."""
    from loop.agents.researcher import diff_hash
    parent, onto = store.get_node(node.parent_id or ROOT_NODE_ID), store.get_node(onto_id)
    if parent is None or onto is None:
        commit, how = "", "conflict: no parent / incumbent node"
    elif hasattr(adapter, "graft_port"):
        commit, how = adapter.graft_port(parent.commit, node.commit, onto.commit, f"bf-{node.id}")
    elif hasattr(adapter, "graft"):
        commit = adapter.graft(parent.commit, node.commit, onto.commit, f"bf-{node.id}")
        how = "applied" if commit else "conflict: graft did not apply"
    else:
        commit, how = "", "conflict: the adapter cannot graft"
    if not commit:
        store.log("backfill_already_in_incumbent" if how == "contained" else "backfill_graft_failed",
                  node_id=node.id, onto=onto_id, outcome=how)
        return "", how
    store.put_node(dataclasses.replace(
        node, id=gid, commit=commit, parent_id=onto_id, second_parent_id=None, status=NodeStatus.SCREENING,
        diff_hash=diff_hash(f"backfill:{commit}"), confirm_alt=None, confirm_e=None, reject_reason="",
        created_ts=now, updated_ts=now, notes=f"{node.notes or ''} backfill_of={node.id}"))
    store.set_state(f"config_diff:{gid}", dict(store.get_state(f"config_diff:{node.id}") or {}))
    store.log("node_transition", node_id=gid, frm="new", to=NodeStatus.SCREENING.value,
              reason=f"backfill: {node.id} ({node.parent_id}) ported onto incumbent {onto_id}")
    return gid, how


def port_requests(ctx: Any, store: Any, snapshot: Any, selector: Any, limit: int, live: set[str]) -> list[Any]:
    """Researcher side requests (the driver's, lowest priority, inside ``researcher.max_side``) for ``port_conflict``
    books onto the CURRENT incumbent, best E[gain+] first, at most ``backfill.port_requests`` in flight (``live``:
    the request ids still researching); each node gets ONE (its book entry records the request), never a loop."""
    cfg = ctx.cfg.backfill
    if not cfg.enabled or snapshot is None or limit <= 0 or cfg.port_requests <= 0:
        return []
    book = dict(store.get_state(K_BOOK) or {})
    room = min(limit, cfg.port_requests - sum(v.get("request") in live for v in book.values()))
    todo = sorted((k for k, v in book.items() if v.get("outcome") == "port_conflict" and not v.get("request")
                   and v.get("target") == snapshot.incumbent_id and store.get_node(k) is not None),
                  key=lambda k: (-float(book[k].get("value", 0.0)), k))
    out = []
    for nid in todo[:max(0, room)]:
        req = selector.port_request(store, snapshot, nid, book[nid]["target"], book[nid].get("conflict", ""))
        book[nid] = {**book[nid], "request": req.id}
        out.append(req)
    if out:
        store.set_state(K_BOOK, book)
    return out


def extra_confirm(store: Any, node_id: str) -> int:
    return int((store.get_state(K_BOOK) or {}).get(node_id, {}).get("extra", 0))


def summary(ctx: Any, store: Any, snapshot: Any, now: float, top: int = 5) -> str:
    q, book = plan(ctx, store, snapshot, now), store.get_state(K_BOOK) or {}
    ports = [v.get("outcome", "applied") for v in book.values() if v.get("action") == "port"]
    return (f"{'on' if ctx.cfg.backfill.enabled else 'OFF (plan only)'}: queued {len(q)}, active {len(active(store))}"
            f"/{ctx.cfg.backfill.max_active}, done {len(book)}"
            + (f" (ports: {ports.count('applied')} applied, {ports.count('already_in_incumbent')} already in the "
               f"incumbent, {ports.count('port_conflict')} conflicts, "
               f"{sum(bool(v.get('request')) for v in book.values())} port requests)" if ports else "")
            + ("".join(f"\n  {i.line()}" for i in q[:top])))

