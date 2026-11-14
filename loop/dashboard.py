"""loop.dashboard — ONE self-contained HTML page (inline CSS/JS, no network) of what the loop ideates,
runs and learns (DESIGN §6.2). ``render(ctx, store, snapshot)`` reads the store and an already-fitted
snapshot (never fits, never writes); ``write`` puts it at ``<state_dir>/dashboard/index.html``
atomically. Every piece of stored text is LLM- or log-derived and therefore untrusted: server-rendered HTML is
escaped, and the graph view (experiment tree, score-over-time chart, detail panel; dashboard_assets/*.js|css,
inlined) is drawn client-side from ONE JSON blob that cannot close its <script> element, via textContent only."""

from __future__ import annotations

import html
import json
import re
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from loop import selftune as T
from loop.contracts import (BANDIT_ARMS, CLOSED_STATUSES, PANEL, ROOT_NODE_ID, NodeStatus, RunStatus, VoidType,
                            design_point)
from loop.design import K_DESIGN, fmt_effect
from loop import selftune
from loop.labless import K_LABLESS, gate_state
from loop.quality import backend_of, prediction_stats, predictions, quality_groups
from loop.select.bandit import brief_of

DIFF_MAX = 60_000                     # chars of one diff shown inline
_ACTIVE = (RunStatus.SUBMITTING, RunStatus.RUNNING)
#: {node_id: "passed" | "incumbent"}: the tick's DERIVED status for the nodes whose stored status cannot say it (v3 stores
#: neither: only the derive knows a test PASS / the incumbent); written by ``Tick._dashboard`` before each render
K_STATUS = "dashboard.derived_status"
K_RENDER = "dashboard.render"        # {ticks, ts}: the tick's render schedule
_PROBES = "select.probes_used"        # loop.select: "<review id>:<i>" of probe briefs served
DAY = 86400.0


def E(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _f(x: Any, fmt: str = "+.4f") -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)


def _ts(ts: float | None) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts)) if ts else "-"


def _age(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _table(head: Iterable[str], rows: Iterable[Iterable[Any]], cls: str = "") -> str:
    """Cells are pre-escaped HTML."""
    rows = list(rows)
    if not rows:
        return '<p class="muted">none yet</p>'
    th = "".join(f"<th>{E(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _chip(text: Any, cls: str = "") -> str:
    return f'<span class="chip {cls}">{E(text)}</span>'


def _pcls(p: float | None) -> str:
    return "" if p is None else "pos" if p >= 0.8 else "neg" if p <= 0.2 else ""


def _json(text: str) -> list:
    try:
        v = json.loads(text) if text else []
        return v if isinstance(v, list) else []
    except (TypeError, ValueError):
        return []


# ------------------------------------------------------------------ header
def arm_shares(events: Iterable[dict], now: float) -> str:
    """Share of BANDIT-DRAWN research by arm (research_started; design/tune rows, deterministic debugs, winner
    crosses and restart re-issues are not bandit draws), last 24 h and all time."""
    arms = {a.value for a in BANDIT_ARMS}
    drawn = [e for e in events if e["event"] == "research_started" and e.get("strategy") in arms
             and not (e.get("deterministic_debug") or e.get("winner_cross") or e.get("restarted"))]

    def fmt(rows: list[dict]) -> str:
        c = Counter("bold" if e.get("brief_kind") == "bold" else e["strategy"] for e in rows)   # v3 review S2: bold apart
        n = sum(c.values())
        return f"{', '.join(f'{a} {v / n:.0%}' for a, v in c.most_common())} (n={n})" if n else "none"
    return f"24h {fmt([e for e in drawn if e['ts'] > now - DAY])}; all-time {fmt(drawn)}"


def _header(ctx: Any, store: Any, snap: Any, nodes: list, runs: list, events: list, now: float) -> str:
    from loop.driver import pool_use
    inc = snap.nodes.get(snap.incumbent_id) if snap else None
    est = (inc.confirm_estimate or inc.utility) if inc else None
    best = max(((i, p.utility.mean) for i, p in (snap.nodes if snap else {}).items()
                if i != ROOT_NODE_ID and p.utility.n > 0), key=lambda t: t[1], default=None)
    conf = {n.id for n in nodes if n.status is NodeStatus.CONFIRMED}
    fresh = max(((i, p.confirm_estimate.mean) for i, p in (snap.nodes if snap else {}).items()
                 if i in conf and p.confirm_estimate is not None), key=lambda t: t[1], default=None)
    strat = Counter("bold" if "brief=bold" in (n.notes or "") else n.strategy.value for n in nodes)
    strat["compound"] = sum("compound=" in (n.notes or "") for n in nodes)
    alerts = [f"{c}x {e}" for e, c in Counter(e["event"] for e in events if e["event"].startswith("ALERT")).items()]
    slow = sum(r.void_type is VoidType.TOO_SLOW for r in runs)
    alerts += [f"{slow} run(s) too_slow"] if slow else []
    alerts += [f"{len(p)} operator proposal(s) open"] if (p := T.open_proposals(store)) else []
    fails = sum(e["event"] in ("research_failed", "research_error") and e["ts"] > now - DAY for e in events)
    alerts += [f"{fails} research failure(s) in 24h"] if fails else []
    alerts += ["sigma UNMEASURED"] if not math.isfinite(ctx.cfg.panel_sd) else []
    try:
        from loop.agents.llm import SdkLLMClient
        cl = SdkLLMClient(ctx.spec)
        llm = ", ".join(f"{r}={cl.tier(r)}" for r in ("researcher", "review", "scout"))
    except Exception:  # noqa: BLE001 - a status line never fails the page
        llm = "?"
    last_be = next((e.get("backend") for e in reversed(events) if e["event"] == "research_started"), None)
    spec = getattr(ctx, "spec_cfg", None) or ctx.cfg
    eff = T.effective(spec, store)
    floors = ", ".join(f"{a.value} {T.arm_floor(spec, a):.2f}->{T.arm_floor(eff, a):.2f}" for a in BANDIT_ARMS)

    def tile(k: str, v: str, sub: str = "", cls: str = "") -> str:
        return f'<div class="tile {cls}"><div class="k">{E(k)}</div><div class="v">{v}</div><div class="s">{sub}</div></div>'
    tiles = [
        tile("incumbent", E(snap.incumbent_id if snap else ROOT_NODE_ID),
             E(f"U {_f(est.mean)} ± {_f(est.sd, '.4f')}" if est else "baseline")),
        tile("confirmed", str(sum(n.status is NodeStatus.CONFIRMED for n in nodes)),
             E(f"of {len(nodes)} ideas")),
        tile("best posterior (EB)", E(_f(best[1]) if best else "-"), E(best[0] if best else "")),
        tile("best fresh-confirmed", E(_f(fresh[1]) if fresh else "-"), E(fresh[0] if fresh else "")),
        tile("slots", E(f"{sum(r.status in _ACTIVE for r in runs)}/{ctx.cfg.capacity}"), E(pool_use(ctx, runs, store.get_state(T.K_PENDING) or (),
                                                                               store.get_state("runbook.starved_since") or {}))),
        tile("research by arm", E(sum(e["event"] == "research_started" for e in events)),
             E(f"bandit draws: {arm_shares(events, now)}") + "<br>" + E(f"floors spec->effective: {floors}")),
        tile("ideas by arm", E(len(nodes)), E(", ".join(f"{a} {v}" for a, v in strat.most_common() if v))),
        tile("labless", E(gate_state(ctx, store)), E(f"{sum(bool(e.get('submission_id')) for e in store.get_state(K_LABLESS, {}).values())} submitted")),
        tile("alerts", E(len(alerts)), "<br>".join(map(E, alerts)) or "none", "alert" if alerts else ""),
        tile("LLM", E(f"last researcher: {last_be or '-'}"), E(llm)),
        tile("updated", E(time.strftime("%H:%M:%S", time.localtime(now))),
             E(f"{time.strftime('%Y-%m-%d %Z', time.localtime(now))}; sigma panel {_f(snap.sigma.get(PANEL) if snap else ctx.cfg.panel_sd, '.5f')}")),
    ]
    return f'<section id="top"><div class="tiles">{"".join(tiles)}</div></section>'


# ------------------------------------------------------------------ graph data (rendered client-side)
_DPT = re.compile(r"^d(\d+)pt\d+$")


GROUP_ORDER = ("augmentation", "objective", "optimization", "regularization", "training", "architecture",
               "data_curation", "eval_readout", "design", "moonshot", "crossover/compound")
GROUP_OPEN_MAX = 6                    # a family branch this small starts expanded
_ACTIVE_ST = {s.value for s in (NodeStatus.QUEUED, NodeStatus.SCREENING, NodeStatus.PROMOTING, NodeStatus.CONFIRMING)}


def _note(n: Any, key: str) -> str:
    return next((w.split("=", 1)[1] for w in (n.notes or "").split() if w.startswith(key + "=")), "")


def _group_of(d: dict) -> str:
    """The family branch a direct child of root hangs under."""
    if d["arm"] == "moonshot":
        return "moonshot"
    if d["arm"] == "crossover" or d["comp"]:
        return "crossover/compound"
    if d["batch"] is not None or d["arm"] == "design":
        return "design"
    return d["fam"] or "other"


def _stack_from(ctx: Any, store: Any, snap: Any, n: Any) -> tuple[list[str], bool]:
    """The node(s) a compound stacked its knobs from: ``stack_from=`` in its notes (recorded at intake),
    else inferred now from its stacked family's config keys vs today's stackable knobs (flagged)."""
    if rec := _note(n, "stack_from"):
        return [i for i in rec.split(",") if i], False
    comp = _note(n, "compound")
    if not comp or "+" not in comp or snap is None:
        return [], False
    from loop.select.intake import family_of, stackable
    try:
        stack = stackable(ctx, store, snap)
    except Exception:  # noqa: BLE001 - a missing edge never fails the page
        return [], False
    fam, out = comp.split("+", 1)[0], []
    for k, v in (store.get_state(f"config_diff:{n.id}") or {}).items():
        if k in stack and stack[k][2] != n.id and family_of(ctx, k) == fam and \
                json.dumps(stack[k][0], sort_keys=True, default=str) == json.dumps(v, sort_keys=True, default=str) \
                and stack[k][2] not in out:
            out.append(stack[k][2])
    return out, bool(out)


def _branches(data: list[dict], incumbent: str | None) -> tuple[list[dict], dict]:
    """Group root's direct children into family branches (sets each child's ``grp``); design points
    stay folded under their batch, which hangs under the parent most of its points share. A branch is
    ``hot`` when it holds an active node (queued .. confirming), a confirmed one or the incumbent; closed
    screens, however high their P, do not open it. Returns (groups in display order, batch id -> {p, grp})."""
    ids = {d["id"] for d in data}
    par: dict[str, str] = {}
    batches: dict[str, dict] = {}
    votes: dict[str, Counter] = defaultdict(Counter)
    for d in data:
        if d["batch"] is not None:
            bid = f"batch:d{d['batch']}"
            votes[bid][d["p"] if d["p"] in ids else ROOT_NODE_ID] += 1
            par[d["id"]] = bid
    for bid, c in votes.items():
        top = max(c.values())
        batches[bid] = {"p": next(k for k in c if c[k] == top), "grp": None}   # first seen wins a tie
        par[bid] = batches[bid]["p"]
    for d in data:
        if d["id"] not in par:
            par[d["id"]] = d["p"] if d["p"] in ids else ROOT_NODE_ID
    kids: dict[str, list[str]] = defaultdict(list)
    for c, p_ in par.items():
        kids[p_].append(c)
    by = {d["id"]: d for d in data}
    members: dict[str, list[str]] = defaultdict(list)
    for c in kids[ROOT_NODE_ID]:
        g = "design" if c in batches else _group_of(by[c])
        members[g].append(c)
        if c in batches:
            batches[c]["grp"] = g
        else:
            by[c]["grp"] = g

    def sub(i: str) -> list[dict]:
        return ([by[i]] if i in by else []) + [x for k in kids.get(i, []) for x in sub(k)]
    groups = []
    for g in sorted(members, key=lambda g: (GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER), g)):
        nodes = [x for m in members[g] for x in sub(m)]
        scored = [x for x in nodes if x["U"] is not None]
        best = max(scored, key=lambda x: x["U"], default=None)
        hot = any(x["st"] in _ACTIVE_ST or x["st"] == NodeStatus.CONFIRMED.value or x["id"] == incumbent
                  for x in nodes)
        groups.append({"id": f"grp:{g}", "name": g, "n": len(nodes), "kids": len(members[g]),
                       "U": best["U"] if best else None, "P": best["P"] if best else None,
                       "best": best["id"] if best else None, "hot": hot,
                       "open": len(nodes) <= GROUP_OPEN_MAX or hot})
    return groups, batches


def _diff_text(path: str) -> str:
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return ""
    return text[:DIFF_MAX] + ("\n... truncated" if len(text) > DIFF_MAX else "")


def _bank(ctx: Any, n: Any) -> str:
    """``<bank id> (<source>)`` for a node served from an idea bank; '' otherwise."""
    if not (bid := _note(n, "bank")):
        return ""
    from loop.select.bank import load
    src = next((e["source"] for e in load(getattr(ctx.cfg, "idea_banks", ())) if e["id"] == bid), "")
    return f"{bid} ({src})" if src else bid


def _node_data(ctx: Any, snap: Any, n: Any, hyp: Any, ho: Any, pms: list, trans: list[dict], nruns: list,
               st: str | None = None) -> dict:
    post = snap.nodes.get(n.id) if snap else None
    est = (post.confirm_estimate or post.utility) if post else None
    meas = est is not None and est.n > 0
    axes = []
    for t in (PANEL, *ctx.cfg.axes):
        g = post.utility if (post and t == PANEL) else (post.axes.get(t) if post else None)
        pred = hyp.predictions.get(t) if hyp else None
        axes.append([t, g.mean if g is not None and g.n else None, (snap.sigma.get(t) if snap else None) or 0.0,
                     ("pred " + ("+" if pred.sign > 0 else "-") + format(pred.size, ".4f")) if pred else ""])
    dpt = _DPT.match(n.mechanism_tag or "") if design_point(n) else None
    return {
        "id": n.id, "p": n.parent_id, "p2": n.second_parent_id, "st": st or n.status.value, "fam": n.family,
        "tag": n.mechanism_tag, "brief": brief_of(n) or "?", "arm": n.strategy.value,
        "op": n.operator.value if n.operator else "-", "be": backend_of(n), "ts": n.created_ts,
        "comp": _note(n, "compound"), "mp": [], "grp": None, "bank": _bank(ctx, n),
        "lev": ", ".join(f"{'+' if d > 0 else '-'}{k}" for k, d in sorted(n.levers.directions.items()) if d),
        "batch": int(dpt.group(1)) if dpt else None,
        "U": est.mean if meas else None, "sd": est.sd if meas else None, "P": post.p_positive if post and meas else None,
        "Pg": post.p_gain if post and meas else None, "Pi": post.p_beats_incumbent if post and meas else None,
        "veto": list(post.veto_axes) if post else [], "axes": axes,
        "claim": hyp.claim if hyp else "", "fals": hyp.falsification if hyp else "", "rej": n.reject_reason or "",
        "ho": [[k, v] for k, v in (("actions", ho.actions), ("learnings", ho.learnings), ("concerns", ho.concerns)) if v] if ho else [],
        "pm": [[pm.verdict, pm.explanation, pm.mechanism_reasoning or "", pm.suggested_next or ""] for pm in pms],
        "tr": [[trans[0]["frm"] if trans else n.status.value, n.created_ts, ""]]
              + [[str(t["to"]), t["ts"], str(t.get("reason") or "")[:160]] for t in trans],
        "runs": [[f"{r.stage.value}:{r.seed} {r.status.value}" + (f"/{r.void_type.value}" if r.void_type else ""),
                  r.status.value] for r in nruns],
        "diff": _diff_text(n.diff_path) if n.diff_path else "",
    }


def tune_card(store: Any, node_id: str) -> str:
    from loop.tune import card
    try:
        return card(store, node_id)
    except Exception:  # noqa: BLE001 - a card line is never worth a page
        return ""


def graph_data(ctx: Any, store: Any, snap: Any, nodes: list, runs: list, events: list, now: float,
               derived: Mapping[str, str] | None = None) -> dict:
    """Everything the client-side tree / chart / detail panel needs, as one JSON-able dict."""
    hyps = {h.id: h for h in store.hypotheses()}
    hos = {h.node_id: h for h in store.handoffs()}
    pms: dict[str, list] = defaultdict(list)
    for pm in store.postmortems():
        pms[pm.node_id].append(pm)
    trans: dict[str, list] = defaultdict(list)
    for e in events:
        if e["event"] == "node_transition":
            trans[e.get("node_id")].append(e)
    by_run: dict[str, list] = defaultdict(list)
    for r in runs:
        by_run[r.node_id].append(r)
    order = sorted(nodes, key=lambda n: (n.created_ts, n.id))
    sig = snap.sigma.get(PANEL) if snap else ctx.cfg.panel_sd
    data = [_node_data(ctx, snap, n, hyps.get(n.hypothesis_id or ""), hos.get(n.id), pms.get(n.id, []),
                       trans.get(n.id, []), by_run.get(n.id, []), (derived or {}).get(n.id)) for n in order]
    ids = {d["id"] for d in data} | {ROOT_NODE_ID}
    for n, d in zip(order, data):          # merge edges: both crossover parents; a compound's knob source
        if xo := _note(n, "xover"):
            d["mp"] = [i for i in xo.split(",") if i in ids][:2]
        elif n.strategy.value == "crossover":
            d["mp"] = [i for i in (n.parent_id, n.second_parent_id) if i in ids]
        elif d["comp"]:
            src, inferred = _stack_from(ctx, store, snap, n)
            d["mp"], d["sfi"] = [i for i in src if i in ids and i != n.id], inferred
        if tl := tune_card(store, n.id):   # the tuning pass, on the card's handoff list
            d["ho"] = d["ho"] + [["tuning", tl]]
    groups, batches = _branches(data, snap.incumbent_id if snap else None)
    return {"title": f"{ctx.task_id} loop", "now": now, "root": ROOT_NODE_ID,
            "incumbent": snap.incumbent_id if snap else ROOT_NODE_ID,
            "sigma": sig if sig is not None and math.isfinite(sig) else None,
            "statuses": [*(s.value for s in NodeStatus), *sorted(set((derived or {}).values()) - {s.value for s in NodeStatus})],
            "groups": groups, "batches": batches,
            "nodes": data}


def _blob(data: dict) -> str:
    """JSON that cannot close its <script> element (or open a comment / CDATA) whatever the text holds."""
    return (json.dumps(data, ensure_ascii=True, default=str)
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))


def _graph(nodes: list, derived: Mapping[str, str] | None = None) -> str:
    def sel(name: str, vals: Iterable[str]) -> str:
        opts = "".join(f'<option value="{E(v)}">{E(v)}</option>' for v in sorted(set(vals)))
        return f'<label>{name} <select id="f-{name}"><option value="">all</option>{opts}</select></label>'
    bar = ('<div class="filters">' + sel("status", [(derived or {}).get(n.id, n.status.value) for n in nodes])
           + sel("family", [n.family for n in nodes]) + sel("brief", [brief_of(n) or "?" for n in nodes])
           + sel("arm", [n.strategy.value for n in nodes])
           + '<label>search <input id="f-q" type="search" placeholder="text, id, tag"></label>'
           + '<span id="f-count" class="muted"></span><span id="legend" class="legend"></span></div>')
    empty = "" if nodes else '<p class="muted">no ideas yet</p>'
    return (f'<section id="ideas" class="tab">{bar}{empty}<div class="gwrap"><div class="gcol">'
            '<div class="pane"><div class="ph">Experiment tree <span class="muted small">wheel = zoom · drag = pan · '
            'double-click = fit · click a branch or batch to expand / collapse</span></div><svg id="tree" class="tree"></svg></div>'
            '<div class="pane"><div class="ph">Score over time <span class="muted small">corrected shrunk U ± sd '
            'by creation order · band = ±1 σ panel · stair = best confirmed</span></div>'
            '<svg id="chart" class="chart"></svg></div></div>'
            '<aside id="panel" class="panel"><p class="muted">click a node in the tree or chart</p></aside></div>'
            '<div id="tip" class="tip" hidden></div></section>')


# ------------------------------------------------------------------ reviews
def _reviews(store: Any, snap: Any, events: list) -> str:
    used = set(store.get_state(_PROBES, []) or [])
    acked = set(store.get_state(T.K_ACKED) or [])
    applied: dict[str, list] = defaultdict(list)
    for e in events:
        if e["event"] in ("selftune", "selftune_refused"):
            applied[e.get("source", "")].append(e)
    out = []
    for k, rv in enumerate(reversed(store.reviews())):
        metas = []
        for mid in rv.meta_hypothesis_ids:
            h, ev = store.get_hypothesis(mid), (snap.hypotheses.get(mid) if snap else None)
            trials = list(ev.trial_ids) if ev and ev.trial_ids else [nid for nid, _ in store.links(mid)]
            metas.append([E(mid), E(h.claim if h else "?"), E(ev.status.value if ev else "untested"),
                          E(_f(ev.e_value, ".2f") if ev else "-") + " / " + E(_f(ev.e_against, ".2f") if ev else "-"),
                          " ".join(f'<a href="#node-{E(t)}">{E(t)}</a>' for t in trials) or "-"])
        probes = [[_chip("used" if f"{rv.id}:{i}" in used else "pending", "pos" if f"{rv.id}:{i}" in used else ""),
                   E(mid or "-"), E(b)] for i, (mid, b) in enumerate(rv.probe_briefs)]
        tune = [[E(t.get("knob")), E(t.get("value")), E(t.get("reason"))] for t in _json(rv.selftune) if isinstance(t, dict)]
        tune += [[E(e.get("knob")), E(f"{e.get('old')} → {e.get('new')}" if e["event"] == "selftune" else "refused"),
                  E(e.get("reason") or e.get("why") or "")] for e in applied.get(rv.id, [])]
        props = [[_chip("acked" if f"{rv.id}#{i}" in acked else "open", "" if f"{rv.id}#{i}" in acked else "hot"),
                  E(f"{rv.id}#{i}"), E(p)] for i, p in enumerate(rv.proposals)]
        design = _json(rv.design)
        out.append(
            f'<details class="card" id="rv-{E(rv.id)}"{" open" if k == 0 else ""}><summary><b>{E(rv.id)}</b> '
            f'{_chip(rv.trigger, "hot" if rv.trigger == "first_principles" else "")} <span class="muted">{E(_ts(rv.created_ts))} · '
            f'at {rv.n_evals_at} evals</span></summary><p>{E(rv.summary)}</p>'
            + ("<h4>findings</h4><ul>" + "".join(f"<li>{E(f)}</li>" for f in rv.findings) + "</ul>" if rv.findings else "")
            + (f'<h4>explore bias</h4><p class="small">{E(", ".join(f"{a} {b:+.2f}" for a, b in rv.explore_bias.items()))}</p>' if rv.explore_bias else "")
            + ("<h4>meta-hypotheses</h4>" + _table(["id", "claim", "status", "E for / against", "trials"], metas) if metas else "")
            + ("<h4>probe briefs</h4>" + _table(["", "meta", "brief"], probes) if probes else "")
            + ("<h4>design request</h4>" + _table(["key", "low", "high"], [[E(d.get("key")), E(d.get("low")), E(d.get("high"))]
                                                                      for d in design if isinstance(d, dict)]) if design else "")
            + ("<h4>self-tune</h4>" + _table(["knob", "value", "reason"], tune) if tune else "")
            + ("<h4>operator proposals</h4>" + _table(["", "id", "proposal"], props) if props else "")
            + "</details>")
    return f'<section id="reviews" class="tab"><h2>Reviews</h2>{"".join(out) or "<p class=muted>no reviews yet</p>"}</section>'


# ------------------------------------------------------------------ world model
def _scatter(pts: list[tuple[str, str, float, float]]) -> str:
    if not pts:
        return '<p class="muted">no scored predictions yet</p>'
    W, H, P = 360, 260, 36
    lim = max(1e-6, *(abs(v) for _, _, a, b in pts for v in (a, b))) * 1.1
    sx = lambda v: P + (v + lim) / (2 * lim) * (W - 2 * P)         # noqa: E731
    sy = lambda v: H - P - (v + lim) / (2 * lim) * (H - 2 * P)     # noqa: E731
    dots = "".join(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="4" class="pt st-{E(st)}"><title>{E(i)} ({E(st)}): '
                   f'predicted {a:+.4f}, measured {b:+.4f}</title></circle>' for i, st, a, b in pts)
    return (f'<svg class="scatter" viewBox="0 0 {W} {H}" role="img" aria-label="predicted vs measured">'
            f'<line x1="{sx(-lim)}" y1="{sy(-lim)}" x2="{sx(lim)}" y2="{sy(lim)}" class="diag"/>'
            f'<line x1="{sx(0)}" y1="{P}" x2="{sx(0)}" y2="{H - P}" class="axl"/>'
            f'<line x1="{P}" y1="{sy(0)}" x2="{W - P}" y2="{sy(0)}" class="axl"/>{dots}'
            f'<text x="{W / 2}" y="{H - 8}" class="lbl" text-anchor="middle">predicted panel delta (±{lim:.4f})</text>'
            f'<text x="12" y="{H / 2}" class="lbl" text-anchor="middle" transform="rotate(-90 12 {H / 2})">measured increment</text></svg>')


def _world(ctx: Any, store: Any, snap: Any, nodes: list) -> str:
    if snap is None:
        return '<section id="world" class="tab"><h2>World model</h2><p class="muted">no world fit yet</p></section>'
    targets = [PANEL, *ctx.cfg.axes]
    lev: dict[str, dict] = defaultdict(dict)
    for e in snap.levers:
        lev[e.lever][e.target] = e
    lrows = [[E(l)] + [f'<span class="{_pcls(c.p_positive)}">{E(_f(c.mean))}±{E(_f(c.se, ".4f"))}</span>'
                       f'<br><span class="muted">P {c.p_positive:.2f} n {c.n_nonzero}</span>' if (c := by.get(t)) else "-"
                       for t in targets] for l, by in sorted(lev.items())]
    fam = [[E(k), E(f"{p.mean:+.4f} ± {p.sd:.4f}"), E(p.n)] for k, p in sorted(snap.family_means.items(), key=lambda kp: -kp[1].mean)]
    book = store.get_state(K_DESIGN, {}) or {}
    drows = []
    for r in book.get("batches", []):
        if "effects" not in r:
            drows.append([E(f"d{r.get('batch')}"), E("no estimate: " + str(r.get("error", "?"))), "", "", ""])
            continue
        for kb, e in zip(r["knobs"], r["effects"]):
            drows.append([E(f"d{r['batch']} (n={r.get('n')}, stacked {r.get('stacked') or 'none'})"), E(kb.get("key")),
                          E(f"{kb.get('low')!r} → {kb.get('high')!r}"),
                          f'<span class="{"pos" if e["significant"] and e["effect"] > 0 else "neg" if e["significant"] else ""}">'
                          f'{E(fmt_effect(e))}</span>',
                          E("; ".join(map(str, r.get("interactions", []))) or "-")])
    if cur := book.get("current"):
        pts = [store.get_node(i) for i in cur.get("points", {}).values() if i != ROOT_NODE_ID]
        drows.append([E(f"d{cur.get('batch')} running"), E(", ".join(str(kb.get("key")) for kb in cur.get("knobs", []))),
                      E(f"{sum(bool(p) and p.status in CLOSED_STATUSES for p in pts)}/{len(pts)} points closed"), "", ""])
    qrows = []
    for title, g, r in quality_groups(store, snap):
        p = r["pred"]
        qrows.append([E(title), E(g), E(r["n"]), E(_f(r["exec"], ".0%")), E(f"{r['hits']}/{r['screened']}"), E(r["conf"]),
                      E(f"{p['sign_acc']:.0%}/{p['n']} bias {p['bias']:+.4f}" + (
                          f" r {_f(p.get('pearson'), '+.2f')} ρ {_f(p.get('spearman'), '+.2f')}" if "pearson" in p else "")
                        if p["n"] else "-"),
                      E(f"{r['fams']}/{r['tags']}"), E(_f(r["dup"], ".0%")), E(f"{r['gpu']:.1f}"),
                      E(f"{r['gpu'] / (r['gain'] / 0.01):.0f}" if r["gain"] > 0 else "-"),
                      E(" ".join(f"{k}:{v}" for k, v in r["rej"].most_common(3)) or "-")])
    try:
        from loop.meta import bias_knob, scorecard
        mrows = [[E(c), E(text), E(per), E(f"{bias_knob(c)} = {b:+.2f}")] for c, text, per, b in scorecard(ctx, store, snap)]
    except Exception as exc:  # noqa: BLE001 - analytics never costs the page
        mrows = [[E("error"), E(f"{type(exc).__name__}: {exc}"[:200]), "", ""]]
    try:
        from loop.components import evaluate
        crows = [[E(r["component"]), E(r["verdict"]), E("-" if r["effect"] != r["effect"] else
                                                         f"{r['effect']:+.2f} ± {r['se']:.2f}"), E(r["n"]),
                  E(r["mode"] + (f" → {r['next']}" if r["next"] != r["mode"] else "")), E(r["source"]),
                  E(f"{r['with']['hits']}/{r['with']['scored']}"), E(f"{r['without']['hits']}/{r['without']['scored']}")]
                 for r in evaluate(ctx, store, snap)]
    except Exception as exc:  # noqa: BLE001
        crows = [[E("error"), E(f"{type(exc).__name__}: {exc}"[:200]), "", "", "", "", "", ""]]
    try:
        from loop.meta import playbook
        from loop.scout import book as scout_book, lines as scout_lines
        pb = playbook(store)
        prows = [[E(r["component"]), E(r["rule"]), E(r.get("evidence", ""))] for r in pb.get("rules", [])]
        pbh = (f'<h3>Playbook v{E(pb["version"])} <span class="muted">(consolidated rules the prompts carry; '
               f'{len(pb.get("retired", []))} retired, {len(pb.get("dropped", []))} dropped at the last '
               f'consolidation for numbers not in the records)</span></h3>' + _table(["component", "rule", "evidence"], prows)
               ) if pb.get("version") else '<h3>Playbook</h3><p class="muted small">none yet (consolidated with first-principles reviews)</p>'
        srows = [[E(_ts(r.get("ts"))), E(r.get("trigger")), E("ok" if r.get("ok") else "FAILED"), E(r.get("ideas")),
                  E(r.get("written")), E(r.get("dup")), E(r.get("refused")), E(r.get("searches")), E(r.get("fetches")),
                  E(", ".join(r.get("ids") or []) or str(r.get("error") or "")[:200])]
                 for r in reversed(scout_book(store)["runs"][-12:])]
        scout_html = (f'<h3>Research scout</h3><p class="small">{E(scout_lines(ctx, store, time.time())[0])}</p>'
                      + _table(["when", "trigger", "", "ideas", "written", "dup", "refused", "searches", "fetches",
                                "ids / error"], srows))
    except Exception as exc:  # noqa: BLE001
        pbh, scout_html = f'<p class="muted">playbook / scout unavailable: {E(type(exc).__name__)}</p>', ""
    pts, pairs = [], []
    for n in nodes:
        for a, b in predictions(store, snap, [n]):
            pts.append((n.id, n.status.value, a, b))
            pairs.append((a, b))
    ps = prediction_stats(pairs)
    cal = (f"sign right {ps['sign_acc']:.0%} of {ps['n']}; mean predicted {ps['mean_pred']:+.4f} vs measured "
           f"{ps['mean_meas']:+.4f}" + (f"; r {_f(ps.get('pearson'), '+.2f')}" if "pearson" in ps else "")) if ps["n"] else ""
    from loop.legs import lines as legs_lines
    stack_html = ('<h3>Leg factorial rounds</h3><pre class="small">' + E(legs_lines(store) or "none yet") + "</pre>")
    return ('<section id="world" class="tab"><h2>World model</h2>' + stack_html +
            '<h3>Lever trends <span class="muted">(effect per +1 step ± se; P(effect&gt;0))</span></h3>'
            + _table(["lever", *targets], lrows, "num")
            + '<h3>Family posteriors</h3>' + _table(["family", "panel mean ± sd", "n"], fam)
            + '<h3>Designed batches</h3>' + _table(["batch", "knob", "low → high", "effect ± se", "aliased 2FI"], drows)
            + '<h3>Ideation quality</h3>' + _table(["by", "group", "n", "exec", "hit/scr", "conf", "prediction", "fam/tag",
                                                   "dup", "GPU-h", "h/+.01", "rejections"], qrows, "num")
            + '<h3>Meta-learning <span class="muted">(each LLM component\'s decisions vs what was measured; '
              'reviews move the bias knobs)</span></h3>' + _table(["component", "record", "GPU-h/success", "bias"], mrows)
            + '<h3>Component audits <span class="muted">(causal: each optional component bypassed on a seeded '
              'share of its decisions; screened hit rate with − without)</span></h3>'
            + _table(["component", "verdict", "effect ± se", "n", "mode", "evidence", "hit with", "hit without"], crows)
            + pbh + scout_html
            + f'<h3>Prediction calibration</h3><p class="small">{E(cal)}</p>{_scatter(pts)}</section>')


# ------------------------------------------------------------------ operations
def _ops(ctx: Any, store: Any, nodes: list, runs: list, events: list, now: float) -> str:
    fam = {n.id: n for n in nodes}
    live = [[f'<a href="#node-{E(r.node_id)}">{E(r.node_id)}</a>', E(r.stage.value), E(r.seed), E(r.pool or "-"),
             E(r.job_id or "-"), E(r.status.value), E(_age(now - r.submitted_ts) if r.submitted_ts else "-"),
             E(f"{fam[r.node_id].family}/{fam[r.node_id].mechanism_tag}" if r.node_id in fam else "")]
            for r in runs if r.status in _ACTIVE]
    fin = {e.get("run_id"): e for e in events if e["event"] == "run_finished"}
    bad = [r for r in runs if r.status in (RunStatus.VOID, RunStatus.CRASH) and r.void_type is not VoidType.CANCELLED]
    bad.sort(key=lambda r: fin.get(r.id, {}).get("ts", 0.0), reverse=True)
    bad_rows = [[E(r.id), E(r.status.value + (f"/{r.void_type.value}" if r.void_type else "")),
                 E(_ts(fin.get(r.id, {}).get("ts"))), E(fin.get(r.id, {}).get("detail", "")),
                 f'<code>{E((r.log_tail or "").strip().splitlines()[-1][:200] if (r.log_tail or "").strip() else "")}</code>']
                for r in bad[:25]]
    day = [e for e in events if e["ts"] > now - DAY]
    ids = {e.get("run_id") for e in day if e["event"] == "run_finished"}
    by_st = Counter(e.get("status") for e in day if e["event"] == "run_finished")
    closed = Counter(e.get("to") for e in day if e["event"] == "node_transition"
                     and e.get("to") in {s.value for s in CLOSED_STATUSES})
    ev = Counter(e["event"] for e in day)
    thr = [["runs finished", E(", ".join(f"{k} {v}" for k, v in by_st.most_common()) or "0")],
           ["GPU-h (finished runs)", E(f"{sum(r.gpu_h for r in runs if r.id in ids):.1f}")],
           ["runs submitted", E(ev["run_submitted"])],
           ["nodes closed", E(", ".join(f"{k} {v}" for k, v in closed.most_common()) or "0")],
           ["research started / admitted", E(f"{ev['research_started']} / {sum(bool(e.get('admitted')) for e in day if e['event'] == 'intake')}")],
           ["research failed", E(ev["research_failed"] + ev["research_error"])]]
    if pl := selftune.preempt_block(ctx, store, now).split("\n", 1)[-1]:
        thr.append(["preemptible pool", E(pl)])
    recent = [[E(_ts(e["ts"])), E(e["event"]), E(", ".join(f"{k}={v}" for k, v in e.items() if k not in ("ts", "event"))[:240])]
              for e in reversed(events[-40:])]
    return ('<section id="ops" class="tab"><h2>Operations</h2><h3>Runs in flight</h3>'
            + _table(["node", "stage", "seed", "pool", "job", "status", "elapsed", "idea"], live)
            + '<h3>Last 24 h</h3>' + _table(["", "count"], thr)
            + '<h3>Recent void / crash / too_slow</h3>' + _table(["run", "status", "when", "detail", "log tail"], bad_rows)
            + '<details><summary>last 40 log events</summary>' + _table(["when", "event", "fields"], recent) + "</details></section>")


# ------------------------------------------------------------------ page
_ASSETS = Path(__file__).with_name("dashboard_assets")


def _clean(x: Any) -> Any:
    """Non-finite floats -> null (strict JSON), recursively."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def render(ctx: Any, store: Any, snapshot: Any, *, now: float | None = None) -> str:
    """The whole page. Reads only; ``snapshot`` may be None (no fit yet). Tree, chart and detail cards are
    drawn client-side from ONE embedded JSON blob; reviews / world model / operations are server-rendered."""
    now = time.time() if now is None else now
    nodes = [n for n in store.nodes() if n.id != ROOT_NODE_ID]
    runs, events = store.runs(), store.events()
    title = f"{ctx.task_id} loop"
    css = (_ASSETS / "dashboard.css").read_text()
    js = (_ASSETS / "dashboard.js").read_text()
    tabs = "".join(f'<a href="#" data-tab="{t}">{E(label)}</a>' for t, label in
                   (("ideas", "Graph"), ("reviews", "Reviews"), ("world", "World model"), ("ops", "Operations")))
    derived = {k: str(v) for k, v in (store.get_state(K_STATUS) or {}).items()}     # the tick's derived PASSED / INCUMBENT
    data = _blob(_clean(graph_data(ctx, store, snapshot, nodes, runs, events, now, derived)))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{E(title)}</title>'
            f'<noscript><meta http-equiv="refresh" content="60"></noscript><style>{css}</style></head><body>'
            f'<nav><b>{E(title)}</b>{tabs}</nav><main>'
            + _header(ctx, store, snapshot, nodes, runs, events, now) + _graph(nodes, derived)
            + _reviews(store, snapshot, events)
            + _world(ctx, store, snapshot, nodes)
            + _ops(ctx, store, nodes, runs, events, now)
            + f'</main><script type="application/json" id="loop-data">{data}</script><script>{js}</script></body></html>')


def write(ctx: Any, store: Any, snapshot: Any, path: str | Path | None = None) -> Path:
    """Render and replace ``path`` (default ``<state_dir>/dashboard/index.html``) atomically."""
    out = Path(path) if path else Path(ctx.state_dir) / "dashboard" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    page = render(ctx, store, snapshot)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=".index.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(page)
        os.chmod(tmp, 0o644)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return out


def serve(directory: str | Path, port: int = 8765, host: str = "127.0.0.1") -> None:
    """Serve ``directory`` read-only (GET/HEAD) on localhost until interrupted."""
    import functools
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class Handler(SimpleHTTPRequestHandler):
        def list_directory(self, path: Any) -> None:       # only the page, no listings
            self.send_error(404)

        def log_message(self, *_: Any) -> None:
            pass
    Path(directory).mkdir(parents=True, exist_ok=True)
    with ThreadingHTTPServer((host, port), functools.partial(Handler, directory=str(directory))) as srv:
        print(f"serving {directory} on http://{host}:{port}/", flush=True)
        srv.serve_forever()
