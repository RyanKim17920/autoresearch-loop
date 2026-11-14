"""loop.leader — OUR LABLESS LEADER: re-base the external bar when one of OUR submissions leads, and never resubmit.

The bar (``loop.core.bar``) is ``labless.leader_median + promote_margin``. ``leader_median`` is an operator number, so
when one of our own submissions becomes the validated Labless leader the bar went stale: the loop kept asking for
``old leader + margin`` while the board's leader was already ours at a higher level (live 11-10: our a48ee1990d /
node a node led the board while the bar stayed .5471 + .005 -- a local re-root onto a node would then have resubmitted
a node itself, which trivially clears the stale bar).

:func:`sync` (every tick, from the board rows ``loop.board`` stored; no HTTP of its own):

- the LEADER row = the most recently updated stored board row with ``is_leader`` true;
- it is OURS iff its submission id is in one of our submission books (``runner.labless`` / ``labless.submissions`` /
  ``labless.recorded``) or its code commit equals one of our nodes' commits (exact; >= 7 hex chars);
- ours: its estimate = a Labless rerun median when the row carries one (``labless.leader_median_keys``, top level or
  under ``metrics``; never ``metric_value``, the single submitted run), else the node's LOCAL mean on the current epoch
  (its fresh TEST seeds when >= ``min_test_seeds``, else all its seeds, needing >= ``min_test_seeds`` of them);
  the effective leader = max(configured ``leader_median``, estimate) -- never easier than the operator's number;
- NOT ours (another contributor leads): the stored re-base is dropped (the bar returns to the configured number);
- no rows / no leader row / a leader we cannot measure: nothing changes (a failed board fetch keeps the old rows, so
  the last known bar stands).

The re-base is state ``labless.leader`` and is applied with :func:`effective` (a ``dataclasses.replace`` of
``labless.leader_median``) by the tick and by ``loop status`` / ``loop dashboard``. Only a configured leader median
(or legacy ``bar_score``) is ever re-based: a task without a bar keeps none. ``labless.leader_rebase: false`` = off.

:func:`submitted` is the resubmission guard ``runner.IncumbentLabless`` consults first (``labless.dedupe_submitted``):
a node that already has a submission (any book, ``recorded``, or a board row with its commit), or whose code commit
another node already submitted, is never sent again."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Sequence

K_LEADER = "labless.leader"     # {run_id, node, estimate, source, n, configured, effective, ts}
_MIN_SHA = 7


def _cfg(ctx_or_cfg: Any) -> Any:
    cfg = getattr(ctx_or_cfg, "cfg", ctx_or_cfg)
    return getattr(cfg, "labless", None)


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(float(v)) else None


def _sha(v: Any) -> str:
    s = str(v or "").strip().lower()
    return s if len(s) >= _MIN_SHA and all(c in "0123456789abcdef" for c in s) else ""


def same_commit(a: Any, b: Any) -> bool:
    """Exact commit match (a short sha >= 7 hex chars matches as a prefix of the full one); '' never matches."""
    x, y = _sha(a), _sha(b)
    if not x or not y:
        return False
    return x == y or (min(len(x), len(y)) < 40 and (x.startswith(y) or y.startswith(x)))


def row_commit(row: Mapping[str, Any]) -> str:
    repo = row.get("repo") if isinstance(row.get("repo"), Mapping) else {}
    return str(row.get("git_commit") or repo.get("commit") or "")


def row_sid(row: Mapping[str, Any]) -> str:
    """The submission id of a board row (``run_sub_<id>`` -> ``<id>``; or ``metrics.submission_id``)."""
    m = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
    if m.get("submission_id"):
        return str(m["submission_id"])
    rid = str(row.get("run_id") or row.get("id") or "")
    return rid[len("run_sub_"):] if rid.startswith("run_sub_") else rid


def _sid_norm(s: Any) -> str:
    s = str(s or "").strip()
    return s[len("run_sub_"):] if s.startswith("run_sub_") else s


def our_submissions(ctx: Any, store: Any) -> dict[str, str]:
    """submission id -> node, from every submission book the loop keeps (v3 incumbent, v2 tiers, operator-recorded)."""
    from loop.labless import K_LABLESS as K_V2
    from loop.runner import K_LABLESS as K_V3
    out: dict[str, str] = {}
    for key in (K_V2, K_V3):
        try:
            book = store.get_state(key) or {}
        except Exception:  # noqa: BLE001 - an unreadable book is an empty one
            book = {}
        for nid, e in (book.items() if isinstance(book, Mapping) else ()):
            if isinstance(e, Mapping) and e.get("submission_id"):
                out[_sid_norm(e["submission_id"])] = str(nid)
    for nid, e in dict(getattr(_cfg(ctx), "recorded", None) or {}).items():
        if isinstance(e, Mapping) and e.get("submission_id"):
            out[_sid_norm(e["submission_id"])] = str(nid)
    return out


def _dead_copy(n: Any) -> bool:
    """A node that only CARRIES another node's commit: a failed graft (``port_failed``: ``cutover.port_node`` records
    the TARGET's commit on it) or a rejected / void node. 11-11 live: the failed moonshot-rescue port a node held
    a node's commit, sorted before it, and the leader re-root made it the incumbent."""
    st = getattr(n, "status", "")
    return " port_failed" in f" {getattr(n, 'notes', '') or ''} " or str(getattr(st, "value", st)).lower() in (
        "rejected", "void")


def _commits(store: Any) -> dict[str, str]:
    """node -> its code commit (nodes with a usable sha only; never a failed graft / rejected copy, :func:`_dead_copy`)."""
    try:
        return {n.id: str(n.commit) for n in store.nodes() if _sha(getattr(n, "commit", "")) and not _dead_copy(n)}
    except Exception:  # noqa: BLE001
        return {}


def our_node(ctx: Any, store: Any, row: Mapping[str, Any], *, sids: Mapping[str, str] | None = None,
             commits: Mapping[str, str] | None = None) -> str | None:
    """The node of ours a board row is (None: not ours). Submission id first, then the code commit; several nodes on
    one commit (11-11: a rescue port copy carried a node's commit and sorted first): a BUILT non-port node (its commit
    differs from its parent's, no ``port_of=``) first, then the oldest created, then the id."""
    sids = our_submissions(ctx, store) if sids is None else sids
    if (sid := _sid_norm(row_sid(row))) and sid in sids:
        return sids[sid]
    c = row_commit(row)
    commits = _commits(store) if commits is None else commits
    hits = [nid for nid, nc in commits.items() if same_commit(nc, c)]
    if not hits:
        return None
    try:
        nodes = {n.id: n for n in store.nodes()}
    except Exception:  # noqa: BLE001 - no node rows: the id order
        nodes = {}

    def rank(nid: str) -> tuple:
        n = nodes.get(nid)
        par = nodes.get(getattr(n, "parent_id", None) or "") if n is not None else None
        port = "@" in nid or "port_of=" in str(getattr(n, "notes", "") or "")
        copy = par is not None and same_commit(getattr(par, "commit", ""), commits.get(nid, ""))
        return (port or copy, float(getattr(n, "created_ts", 0.0) or 0.0) if n is not None else math.inf, nid)
    return min(hits, key=rank)


def leader_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The board's current leader: the most recently updated row with ``is_leader`` true (a stale row of an earlier
    leader that was never re-fetched loses to the newer one)."""
    leads = [r for r in rows if isinstance(r, Mapping) and r.get("is_leader") is True]
    if not leads:
        return None
    return max(leads, key=lambda r: (str(r.get("updated_at") or ""), str(r.get("run_id") or "")))


def rerun_median(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    """A Labless rerun / replication median the row carries (``keys``, top level then ``metrics``); None if none."""
    m = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
    for k in keys:
        for src in (row, m):
            if (v := _num(src.get(k))) is not None:
                return v
    return None


def local_estimate(led: Any, node: str, lc: Any) -> tuple[float, int, str] | None:
    """(mean, n seeds, which) of ``node`` on the current epoch: its fresh TEST seeds when >= min_test_seeds of them,
    else all its seeds (>= min_test_seeds needed); None when not measurable (no ledger, another epoch, too few seeds)."""
    if led is None or not node:
        return None
    from loop.core import bar as BR
    nid = node
    try:
        if (p := led.port(node)) is not None:
            nid = p
    except Exception:  # noqa: BLE001 - no port: the node itself
        pass
    if nid == getattr(led, "base_root", None):
        obs = led.in_epoch().by_node().get(nid, ())
    else:
        n = led.nodes.get(nid)
        if n is None or n.base != led.epoch:
            return None
        obs = led.in_epoch().by_node().get(nid, ())
    k = max(1, int(getattr(lc, "min_test_seeds", BR.MIN_TEST_SEEDS) or 1))
    t = BR.level(o for o in obs if BR.fresh_test(o))
    if t is not None and t.n >= k and math.isfinite(t.mean):
        return t.mean, t.n, "fresh TEST"
    a = BR.level(obs)
    if a is not None and a.n >= k and math.isfinite(a.mean):
        return a.mean, a.n, "screen+test"
    return None


def configured_leader(lc: Any) -> float | None:
    """The operator's ``labless.leader_median`` (None: not configured -- the legacy ``bar_score`` / ``bar_node`` bars
    are never re-based: their margin is ``bar_margin``, not ``promote_margin``)."""
    return _num(getattr(lc, "leader_median", None)) if lc is not None else None


def effective(cfg: Any, store: Any, spec_lc: Any = None) -> Any:
    """``cfg`` with the stored re-base applied: ``labless.leader_median`` = max(configured, the stored estimate).
    ``spec_lc``: the configured ``labless`` section (default ``cfg.labless``; the tick passes the spec's, since its
    ``cfg`` already carries the previous re-base). Off / nothing stored / no leader median configured: the configured
    value."""
    cur = getattr(cfg, "labless", None)
    lc = spec_lc if spec_lc is not None else cur
    if cur is None or lc is None:
        return cfg
    target = getattr(lc, "leader_median", None)
    if getattr(lc, "leader_rebase", True) and (conf := configured_leader(lc)) is not None:
        try:
            st = store.get_state(K_LEADER) or {}
        except Exception:  # noqa: BLE001 - no state: the configured bar
            st = {}
        est = _num(st.get("estimate")) if isinstance(st, Mapping) else None
        if est is not None and est > conf:
            target = est
    if getattr(cur, "leader_median", None) == target:
        return cfg
    return dataclasses.replace(cfg, labless=dataclasses.replace(cur, leader_median=target))


def sync(ctx: Any, store: Any, led: Any, now: float, *, rows: Sequence[Mapping[str, Any]] | None = None,
         spec_lc: Any = None) -> list[str]:
    """Re-judge who leads from the stored board rows and store the re-base (state K_LEADER). Returns log lines: one
    when the effective leader median moves (old -> new, why), one when the leader is ours but changes nothing.
    ``spec_lc``: the configured ``labless`` section (the tick passes the spec's, its ctx.cfg carries the re-base)."""
    lc = spec_lc if spec_lc is not None else _cfg(ctx)
    if lc is None or not getattr(lc, "leader_rebase", True):
        return []
    conf = configured_leader(lc)
    if conf is None:
        return []
    if rows is None:
        from loop.board import stored_runs
        rows = stored_runs(store)
    if not rows:
        return []                                       # never fetched: the last known bar stands
    st = dict(store.get_state(K_LEADER) or {})
    old_eff = max(conf, _num(st.get("estimate")) or -math.inf)
    lead = leader_row(rows)
    if lead is None:
        return []                                       # no leader marked: ambiguous, keep what we know
    rid = str(lead.get("run_id") or lead.get("id") or "")
    node = our_node(ctx, store, lead)
    if node is None:
        if not st.get("estimate") and st.get("run_id") == rid:
            return []
        store.set_state(K_LEADER, {"run_id": rid, "node": None, "estimate": None, "source": "not ours",
                                   "configured": conf, "effective": conf, "ts": now})
        if old_eff != conf:
            store.log("labless_leader_rebase", run_id=rid, node_id=None, old=old_eff, new=conf, why="not ours")
            return [f"labless: bar re-based leader {old_eff:.4f} -> {conf:.4f} (the configured leader_median): the "
                    f"Labless leader {rid} is not ours"]
        return []
    rm = rerun_median(lead, tuple(getattr(lc, "leader_median_keys", ()) or ()))
    if rm is not None:
        est, n, source = rm, 0, "Labless rerun median"
    elif (le := local_estimate(led, node, lc)) is not None:
        est, n, source = le[0], le[1], f"local mean on {le[1]} {le[2]} seeds"
    elif st.get("run_id") == rid and _num(st.get("estimate")) is not None:
        return []                                       # not measurable now: keep the stored estimate for this leader
    else:
        est, n, source = None, 0, "not measurable"
    new_eff = conf if est is None else max(conf, est)
    same = st.get("run_id") == rid and st.get("node") == node and st.get("source") == source \
        and (_num(st.get("estimate")) is None) == (est is None) \
        and (est is None or abs((_num(st.get("estimate")) or 0.0) - est) < 1e-7)
    if same:
        return []
    store.set_state(K_LEADER, {"run_id": rid, "node": node, "estimate": est, "source": source, "n": n,
                               "configured": conf, "effective": new_eff, "ts": now})
    margin = float(getattr(lc, "promote_margin", 0.0) or 0.0)
    if abs(new_eff - old_eff) > 1e-9:
        why = (f"our submission {rid} (node {node}) is the Labless leader; its {source} is "
               + (f"{est:.4f}" if est is not None else "unmeasured") + f" (configured leader_median {conf:.4f}: "
               "the higher stands)")
        store.log("labless_leader_rebase", run_id=rid, node_id=node, old=old_eff, new=new_eff, estimate=est,
                  source=source, configured=conf)
        return [f"labless: bar re-based leader {old_eff:.4f} -> {new_eff:.4f} (bar {old_eff + margin:.4f} -> "
                f"{new_eff + margin:.4f}): {why}"]
    store.log("labless_leader_ours", run_id=rid, node_id=node, estimate=est, source=source, configured=conf)
    if est is None:
        return [f"labless: our submission {rid} (node {node}) is the Labless leader but has no rerun median and too "
                f"few seeds on the current epoch to measure: bar stays {new_eff + margin:.4f}"]
    return [f"labless: our submission {rid} (node {node}) is the Labless leader; its {source} {est:.4f} does not "
            f"exceed the configured leader_median {conf:.4f}: bar stays {new_eff + margin:.4f}"]


def submitted(ctx: Any, store: Any, node: str) -> str:
    """'' when ``node`` was never submitted to Labless, else where its earlier submission is recorded: its own book
    entry, ``labless.recorded``, another node's submission of the SAME code commit, or a board row with its commit."""
    sids = our_submissions(ctx, store)
    for sid, nid in sorted(sids.items()):
        if nid == node:
            return f"submission {sid}"
    try:
        n = store.get_node(node)
    except Exception:  # noqa: BLE001
        n = None
    c = str(getattr(n, "commit", "") or "")
    if not _sha(c):
        return ""
    commits = _commits(store)
    for sid, nid in sorted(sids.items()):
        if nid != node and same_commit(commits.get(nid, ""), c):
            return f"submission {sid} of {nid} (same commit {c[:10]})"
    try:
        from loop.board import stored_runs
        rows = stored_runs(store)
    except Exception:  # noqa: BLE001 - no board: the books decide
        rows = []
    for r in rows:
        if isinstance(r, Mapping) and same_commit(row_commit(r), c):
            return f"board run {r.get('run_id') or r.get('id')} (same commit {c[:10]})"
    return ""


# ------------------------------------------------------------------------------ re-root onto our Labless leader
K_REROOT = "labless.leader_reroot"      # {run_id, node, previous, ts}: the leader submission already re-rooted onto
VALIDATED = frozenset({"validated", "leader"})


def our_validated_leader(ctx: Any, store: Any, rows: Sequence[Mapping[str, Any]] | None = None
                         ) -> tuple[str, str] | None:
    """(board run id, our node) when the board's current leader (:func:`leader_row`) is OURS and VALIDATED
    (``validation`` / ``raw_validation`` in validated / leader); else None."""
    if rows is None:
        from loop.board import stored_runs
        rows = stored_runs(store)
    lead = leader_row(rows or ())
    if lead is None:
        return None
    if not ({str(lead.get("validation") or "").lower(), str(lead.get("raw_validation") or "").lower()} & VALIDATED):
        return None
    node = our_node(ctx, store, lead)
    return (str(lead.get("run_id") or lead.get("id") or ""), node) if node else None


def lineage(led: Any, nid: str) -> set[str]:
    """``nid`` and every ancestor (parents, second parents, and the old-epoch node a port grafts, transitively)."""
    out: set[str] = set()
    todo = [nid]
    while todo:
        x = todo.pop()
        if not x or x in out:
            continue
        out.add(x)
        n = (getattr(led, "nodes", None) or {}).get(x)
        if n is not None:
            todo += [p for p in (n.parent, getattr(n, "second_parent", None), getattr(n, "port_of", "")) if p]
    return out


def reroot_plan(ctx: Any, store: Any, led: Any, inc: str, rows: Sequence[Mapping[str, Any]] | None = None
                ) -> tuple[str, str, str] | None:
    """(board run id, the leader node on the current epoch, why) when the incumbent must switch to our validated
    Labless leader: ``labless.leader_reroot`` on, our node leads, that leader was not re-rooted onto before, and the
    node (its port on the current epoch) is neither the incumbent nor one of its ancestors. None otherwise."""
    lc = _cfg(ctx)
    if lc is None or not getattr(lc, "leader_reroot", True) or led is None:
        return None
    got = our_validated_leader(ctx, store, rows)
    if got is None:
        return None
    rid, node = got
    done = store.get_state(K_REROOT) or {}
    if isinstance(done, Mapping) and done.get("run_id") == rid:
        return None                                     # once per leader: a later legitimate switch is never undone
    try:
        nid = led.port(node) or node
    except Exception:  # noqa: BLE001
        nid = node
    n = led.nodes.get(nid)
    if n is None or n.base != led.epoch:
        return None                                     # not on the current epoch: nothing to switch to
    if nid in lineage(led, inc):
        return None                                     # already the incumbent or one of its ancestors
    return rid, nid, f"our submission {rid} (node {node}) is the validated Labless leader"
