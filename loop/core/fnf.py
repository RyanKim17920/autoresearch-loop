"""loop.core.fnf — the fork-and-flush idea-diversity arm (docs/loop-core-v3.md §6 "Fork-and-flush").

Paper (Fork-and-Flush, arXiv 2610.07447): fork K = 4 branches from the incumbent workspace, each with a FLUSHED
context (empty but for a protected-incumbent notice) and ~6% of the budget, keep the best branch; rounds at 0 / 33 /
66% of the budget. Ablations: fork-only .57, flush-only .55, both .71 vs one agent .45; prompts saying "pick a
DIFFERENT approach" did nothing; their plateau-triggered rounds never fired.

Here a research session already starts with an empty chat. What carries a basin is the shared context every brief
gets: parent handoffs / diff + family postmortems (researcher prompt), the top-5 research notes by similarity, the
engagement lines, the headroom target, the plateau hypotheses and the computed facts (lineage coverage, axis trades).
Late in the 10-08 run 41% of new ideas were near-duplicates and the median nearest-neighbour similarity of new ideas
rose .37 -> .67. So a branch session is FLUSHED: ``request.ablated`` = :data:`FLUSH`; it sees the surface, the
families, the incumbent's diff, the locked rules and the incumbent's score (a protected anchor), and only its own
branch's notes (``research_notes/branch-<id>/``) and handoffs. The intake's dead-mechanism / legality filters stay on
(free), postmortem TEXT is hidden.

- :func:`trigger`     a round starts: the first one at once; then every :data:`EVERY_GPU_H` GPU-h (TUNABLE), on a new
                      plateau episode, or when the meta-report's diversity metric (median NN similarity of the last
                      48 h of new ideas, :func:`nn_similarity`) exceeds :data:`NN_SIM_MAX` (TUNABLE); the last two only
                      :data:`MIN_GAP_FRAC` of the cadence after the previous round
- :func:`assign`      an explore-lane session goes to a branch while the branches' explore GPU-h share is below
                      :data:`SHARE` (TUNABLE, the A/B arm), round-robin over the branches under their cap
- :func:`budgets`     the per-branch GPU-h left: ``BRANCH_FRAC`` x the GPU-h until the next round (the cadence),
                      enforced by ``acquire.rank`` / ``idle_fill`` on every non-test action of a branch node
- :func:`close`       selection on fresh-seed test credit (``ideation.credit`` keyed on ``branch=``) plus, for nodes
                      not yet tested, their shrunk screen posterior (:data:`SCREEN_CREDIT`); "no branch won" is valid;
                      :func:`merge` folds the winner's notes and handoffs into the shared notes, every branch dir
                      is archived
- :func:`ab`          the arm vs the shared-context control: credit per GPU-h with a bootstrap SE, distinct families per
                      day; ``adopt`` only after >= :data:`MIN_ROUNDS` closed rounds and a lead >= 2 SE (core.meta)

Nothing here names an idea."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

K = 4                         # branches per round
BRANCH_FRAC = 0.06            # a branch's GPU-h cap, as a share of the GPU-h until the next round
EVERY_GPU_H = 350.0           # TUNABLE ideation.fnf_every_gpu_h: the cadence (and the cap's horizon)
NN_SIM_MAX = 0.55             # TUNABLE ideation.fnf_nn_sim: diversity trigger on the median NN similarity
SHARE = 0.3                   # TUNABLE ideation.fnf_share: the arm's share of the explore-lane GPU-h (0 = off)
ADOPT_SHARE = 0.7             # the share once adopted (the shared-context control keeps 30%: still measured)
MIN_GAP_FRAC = 0.5            # plateau / diversity rounds at least this share of the cadence after the last round
WINDOW_H = 48.0               # the diversity metric's window
MIN_NEW = 6                   # ideas in the window before the diversity metric counts
NN_POOL = 1500                # earlier ideas a new one is compared with (the most recent)
MIN_ROUNDS = 2                # closed rounds before the A/B may adopt
N_BOOT = 400
#: close: a branch node without a fresh-seed test earns FRESH_SHRINK x max(0, its own-increment posterior) once it has
#: SCREEN_MIN_SEEDS screen seeds (``ideation.fnf_screen_credit`` / ``fnf_screen_min_seeds``; 10-13: rounds closed
#: before any branch node reached a test, so test-only credit could never pick a winner)
SCREEN_CREDIT = False         # v3-c7i: test gains only (ideation.fnf_screen_credit)
SCREEN_MIN_SEEDS = 2
#: what a branch session does not see (``CandidateRequest.ablated``): researcher -- family postmortem text, the parent's
#: handoffs + hypothesis; brief -- the shared notes, engagement lines, the plateau hypotheses, the computed facts
#: (lineage coverage and axis trades: what the shared search already tried). v3-c4i: the headroom table and the
#: computed levers / deficits / external sections are measurements, not the shared story: a branch keeps them
FLUSH = ("postmortem", "notes", "engagement", "handoffs", "plateau", "facts")
K_FNF = "ideation.fnf"        # {round, open: {...} | None, rounds: [closed summaries], last_plateau}
K_DIV = "ideation.fnf.diversity"   # the last meta-report diversity metric {ts, median, n, by_arm}
TAG = "branch="
TEST_STAGES = frozenset({"confirm", "remint", "audit"})
LIVE = frozenset({"submitting", "running"})


def branch_of(node: Any) -> str:
    """The node's ``branch=<round>.<k>`` tag ('' for a shared-context node)."""
    for t in (getattr(node, "notes", "") or "").split():
        if t.startswith(TAG):
            return t[len(TAG):]
    return ""


def idea_node(node: Any) -> bool:
    """A researcher-made idea node (intake tags ``request=``; not a wave point / x* / port)."""
    notes = getattr(node, "notes", "") or ""
    return "request=" in notes and "port_of=" not in notes and "for_wave=" not in notes


def run_hours(led: Any) -> float:
    from loop.core.acquire import run_hours as rh
    return rh(led)


def total_gpu(led: Any) -> float:
    return sum(r.gpu_h for r in led.runs)


def spent(led: Any, nodes: Sequence[str] | set[str], run_h: float | None = None) -> float:
    """GPU-h of the non-test runs of ``nodes``: finished at their GPU-h, live ones at a full run."""
    nodes, run_h = set(nodes), run_hours(led) if run_h is None else run_h
    return sum((run_h if r.status in LIVE else r.gpu_h) for r in led.runs
               if r.node in nodes and r.stage not in TEST_STAGES)


def state(store: Any) -> dict[str, Any]:
    return dict(store.get_state(K_FNF) or {"round": 0, "open": None, "rounds": []})


# ---------------------------------------------------------------------------------------------- the trigger
def trigger(led: Any, judged: Sequence[Any], st: Mapping[str, Any], div: Mapping[str, Any] | None) -> str | None:
    """Why a round starts (or the open one closes) now, else None."""
    from loop.core.ideation import plateau
    rnd = st.get("open")
    if rnd is None and not st.get("rounds"):
        return "start" if SHARE > 0 else None
    last = rnd or (st.get("rounds") or [{}])[-1]
    since = total_gpu(led) - float(last.get("gpu0", 0.0))
    if since >= EVERY_GPU_H:
        return "cadence"
    if SHARE <= 0 or since < MIN_GAP_FRAC * EVERY_GPU_H:
        return None
    p = plateau(led, judged)
    if p.active and st.get("last_plateau") != p.since:
        return "plateau"
    if div and float(div.get("ts", 0.0)) > float(last.get("ts", 0.0)) and int(div.get("n", 0)) >= MIN_NEW \
            and float(div.get("median", 0.0)) > NN_SIM_MAX:
        return f"diversity (median NN similarity {float(div['median']):.2f} > {NN_SIM_MAX:g})"
    return None


def open_round(st: dict[str, Any], led: Any, judged: Sequence[Any], incumbent: str, why: str, now: float
               ) -> dict[str, Any]:
    from loop.core.ideation import plateau
    n = int(st.get("round", 0)) + 1
    p = plateau(led, judged)
    st.update(round=n, last_plateau=p.since if p.active else st.get("last_plateau"))
    st["open"] = {"id": n, "ts": now, "gpu0": total_gpu(led), "cap": BRANCH_FRAC * EVERY_GPU_H,
                  "branches": [f"{n}.{k}" for k in range(1, K + 1)], "incumbent": incumbent, "trigger": why,
                  "sessions": {}}
    return st["open"]


def track_incumbent(st: dict[str, Any], incumbent: str, now: float) -> bool:
    """The open round's ``incumbent`` follows a switch made during the round (10-13: round 1 opened on
    a node and closed still recording it, though a node had passed mid-round); each switch is kept in
    ``switches`` ({from, to, ts}). True when the round changed (the caller persists ``st``)."""
    rnd = st.get("open")
    if not rnd or not incumbent or rnd.get("incumbent") == incumbent:
        return False
    rnd.setdefault("switches", []).append({"from": rnd.get("incumbent"), "to": incumbent, "ts": now})
    rnd["incumbent"] = incumbent
    return True


# ---------------------------------------------------------------------------------------------- sessions + caps
def round_nodes(led: Any, rnd: Mapping[str, Any], until: float | None = None) -> tuple[dict[str, list[str]], list[str]]:
    """({branch: its nodes}, control nodes): the round's explore-lane idea nodes, created in [ts, until)."""
    br = {b: [] for b in rnd.get("branches", ())}
    ctl = []
    t0, t1 = float(rnd.get("ts", 0.0)), math.inf if until is None else until
    for n in led.nodes.values():
        b = branch_of(n)
        if b in br:
            br[b].append(n.id)
        elif not b and n.lane == "explore" and idea_node(n) and t0 <= n.created_ts < t1:
            ctl.append(n.id)
    return br, ctl


def budgets(st: Mapping[str, Any] | None, led: Any) -> tuple[dict[str, str], dict[str, float]]:
    """(node -> branch, branch -> GPU-h left) for the open round's branches; ({}, {}) without one."""
    rnd = (st or {}).get("open")
    if not rnd:
        return {}, {}
    br, _ = round_nodes(led, rnd)
    run_h = run_hours(led)
    tags = {n: b for b, ns in br.items() for n in ns}
    return tags, {b: max(0.0, float(rnd["cap"]) - spent(led, ns, run_h)) for b, ns in br.items()}


def assign(st: dict[str, Any], led: Any, lane: str, idea_runs: int = 3) -> str:
    """The branch an explore-lane session goes to ('' = the shared-context control) and the session counted. A branch
    while the branches' weight (their non-test GPU-h + one screening's GPU-h per session started) is below ``SHARE``
    of the round's explore-lane weight; the lightest branch under its cap."""
    rnd = st.get("open")
    if not rnd or lane != "explore" or SHARE <= 0:
        return ""
    run_h = run_hours(led)
    e = idea_runs * run_h
    br, ctl = round_nodes(led, rnd)
    ses = rnd.setdefault("sessions", {})
    w = {b: spent(led, ns, run_h) + e * int(ses.get(b, 0)) for b, ns in br.items()}
    wc = spent(led, ctl, run_h) + e * int(ses.get("control", 0))
    open_ = [b for b in br if spent(led, br[b], run_h) < float(rnd["cap"])]
    pick = ""
    if open_ and sum(w.values()) < SHARE * (sum(w.values()) + wc):
        pick = min(open_, key=lambda b: (w[b], b))
    ses[pick or "control"] = int(ses.get(pick or "control", 0)) + 1
    return pick


# ---------------------------------------------------------------------------------------------- selection
def close(st: dict[str, Any], v: Any, now: float) -> dict[str, Any]:
    """Close the open round: per branch the fresh-seed test credit (``ideation.credit`` keyed on ``branch=``, gain =
    max(0, the node's last test-block δ) per GPU-h; with :data:`SCREEN_CREDIT` an untested node's shrunk screen
    posterior counts too); the winner is the branch with the most credit per GPU-h among those with a positive gain,
    else none ("no branch won": nothing is merged)."""
    from loop.core.acquire import FRESH_SHRINK
    from loop.core.ideation import credit
    track_incumbent(st, getattr(v, "incumbent", ""), now)      # the incumbent at close, not at open
    rnd = st["open"]
    br, ctl = round_nodes(v.led, rnd, now)
    cr = credit(v, key=lambda n: branch_of(n) if branch_of(n) in br else "",
                screen=(FRESH_SHRINK, SCREEN_MIN_SEEDS) if SCREEN_CREDIT else None)
    per = {b: cr.get(b, {"nodes": 0, "tested": 0, "gain": 0.0, "gpu_h": 0.0, "per_gpu_h": 0.0}) for b in br}
    won = [b for b in br if per[b]["gain"] > 0]
    winner = max(won, key=lambda b: (per[b]["per_gpu_h"], per[b]["gain"])) if won else ""
    out = {**{k: rnd[k] for k in ("id", "ts", "gpu0", "cap", "branches", "incumbent", "trigger")},
           "closed": now, "winner": winner, "credit": {b: {k: round(x, 6) for k, x in c.items()} for b, c in per.items()},
           "nodes": br, "control": ctl, "sessions": dict(rnd.get("sessions") or {}),
           **({"switches": list(rnd["switches"])} if rnd.get("switches") else {})}
    st.setdefault("rounds", []).append(out)
    st["open"] = None
    return out


def branch_dir(ctx: Any, branch: str) -> Path:
    from loop.agents import notes as NT
    return NT.notes_path(ctx, branch).parent


def merge(ctx: Any, store: Any, closed: Mapping[str, Any]) -> list[str]:
    """The winner's notes (its ``research_notes/branch-<id>/``) and its nodes' handoffs (as notes) go into the shared
    notes; every branch dir of the round is then archived (``research_notes/archive/``). No winner: nothing merged."""
    from loop.agents import notes as NT
    shared = NT.notes_path(ctx)
    acts = []
    if w := closed.get("winner"):
        have = {n["id"] for n in NT.load(shared)}
        rows = [n for n in NT.load(branch_dir(ctx, w) / "notes.jsonl") if n["id"] not in have]
        for nid in closed.get("nodes", {}).get(w, ()):
            for ho in store.handoffs(node_id=nid) if hasattr(store, "handoffs") else ():
                rid = "rn-" + hashlib.sha1(f"handoff:{nid}".encode()).hexdigest()[:8]
                if rid not in have:
                    node = store.get_node(nid)
                    rows.append({"id": rid, "url": "", "key": f"handoff:{nid}", "topic": "branch_handoff",
                                 "title": f"branch {w} handoff {nid} {getattr(node, 'mechanism_tag', '')}"[:120],
                                 "claim": ho.learnings, "mechanism": ho.actions, "setting": f"fork-and-flush branch {w}",
                                 "effect": json.dumps(closed["credit"].get(w, {})), "transfer": ho.concerns,
                                 "request": f"fnf-{w}", "ts": time.time()})
                    have.add(rid)
        if rows:
            shared.parent.mkdir(parents=True, exist_ok=True)
            with shared.open("a") as f:
                for r in rows:
                    f.write(json.dumps({**r, "branch": w}) + "\n")
                    NT.append_md(shared, r)
        acts.append(f"fnf round {closed['id']}: branch {w} won; {len(rows)} notes / handoffs merged into the shared notes")
    else:
        acts.append(f"fnf round {closed['id']}: no branch won (no positive fresh-seed test credit); nothing merged")
    for b in closed.get("branches", ()):
        src = branch_dir(ctx, b)
        if src.exists():
            dst = src.parent / "archive" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
    return acts


# ---------------------------------------------------------------------------------------------- the meta-report
def nn_similarity(store: Any, led: Any, now: float, embedder: Any = None, window_h: float = WINDOW_H
                  ) -> dict[str, Any]:
    """The median, over the idea nodes created in the last ``window_h`` hours, of each one's cosine to its nearest
    earlier idea node (intake embedding: claim + changed lines), overall and per arm (flush / shared)."""
    from loop.core.ideation import node_vector
    import bisect
    import math

    from loop.select.embed import HashingEmbedder
    emb = embedder or HashingEmbedder()
    ideas = sorted((n for n in led.nodes.values() if idea_node(n)), key=lambda n: n.created_ts)
    new = [n for n in ideas if n.created_ts >= now - window_h * 3600]
    sims: dict[str, list[float]] = {"flush": [], "shared": []}
    if new:
        # 10-22: ~170k pure-python cosines (three sums each) took 200 s inside the tick. Unit vectors once, then one
        # C-speed ``math.sumprod`` per pair; a node without a vector is skipped, another dimension scores 0 (``cosine``).
        vecs = [node_vector(store, m.id, emb) for m in ideas]
        unit = []
        for v in vecs:
            nrm = math.sqrt(math.sumprod(v, v)) if v is not None else 0.0
            unit.append(tuple(x / nrm for x in v) if v is not None and nrm else v)
        ts = [m.created_ts for m in ideas]
        for n in new:
            r = ideas.index(n)
            hi = bisect.bisect_left(ts, n.created_ts)
            vec = unit[r]
            if vec is None or hi == 0:
                continue
            best = max((0.0 if len(pv) != len(vec) else math.sumprod(vec, pv)
                        for pv in unit[max(0, hi - NN_POOL):hi] if pv is not None), default=None)
            if best is not None:
                sims["flush" if branch_of(n) else "shared"].append(best)
    allv = sims["flush"] + sims["shared"]
    return {"ts": now, "n": len(allv), "median": round(statistics.median(allv), 4) if allv else 0.0,
            "by_arm": {a: {"n": len(x), "median": round(statistics.median(x), 4) if x else None}
                       for a, x in sims.items()}}


def _arm_credit(rows: Sequence[tuple[float, float]]) -> float:
    g = sum(h for _, h in rows)
    return sum(x for x, _ in rows) / g if g else 0.0


def ab(v: Any, st: Mapping[str, Any], *, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """The arm (branch nodes of the closed rounds) vs the shared-context control (explore-lane idea nodes made during
    those rounds): fresh-seed credit per GPU-h (gain = max(0, last test-block δ)), its bootstrap SE (nodes resampled
    within each arm), distinct families per day. ``verdict``: adopt (>= MIN_ROUNDS rounds, arm - control >= 2 SE),
    keep testing, or insufficient."""
    rounds = list(st.get("rounds") or [])
    gain = {j.node: max(0.0, j.verdict.mean) for j in v.judged}
    gpu: dict[str, float] = {}
    for r in v.led.runs:
        gpu[r.node] = gpu.get(r.node, 0.0) + r.gpu_h
    arm = sorted({n for r in rounds for ns in (r.get("nodes") or {}).values() for n in ns if n in v.led.nodes})
    ctl = sorted({n for r in rounds for n in r.get("control") or () if n in v.led.nodes})
    rows = {"flush": [(gain.get(n, 0.0), gpu.get(n, 0.0)) for n in arm],
            "shared": [(gain.get(n, 0.0), gpu.get(n, 0.0)) for n in ctl]}
    cred = {a: _arm_credit(x) for a, x in rows.items()}
    diff = cred["flush"] - cred["shared"]
    rng, boots = random.Random(seed), []
    if rows["flush"] and rows["shared"]:
        for _ in range(n_boot):
            boots.append(_arm_credit([rng.choice(rows["flush"]) for _ in rows["flush"]])
                         - _arm_credit([rng.choice(rows["shared"]) for _ in rows["shared"]]))
    se = statistics.pstdev(boots) if len(boots) > 1 else math.inf
    days = sum(max(0.0, float(r["closed"]) - float(r["ts"])) for r in rounds) / 86400 or math.nan
    fam = {a: len({v.led.nodes[n].family for n in ns}) / days if days == days else 0.0
           for a, ns in (("flush", arm), ("shared", ctl))}
    if len(rounds) < MIN_ROUNDS or not rows["flush"] or not rows["shared"]:
        verdict = "insufficient"
    elif diff > 0 and diff >= 2 * se:
        verdict = "adopt"
    else:
        verdict = "keep testing"
    return {"rounds": len(rounds), "winners": [r.get("winner") or "-" for r in rounds],
            "credit_per_gpu_h": {a: round(c, 8) for a, c in cred.items()}, "diff": round(diff, 8),
            "se": round(se, 8) if se != math.inf else None, "nodes": {"flush": len(arm), "shared": len(ctl)},
            "gpu_h": {a: round(sum(h for _, h in x), 1) for a, x in rows.items()},
            "tested": {a: sum(1 for n in ns if n in gain) for a, ns in (("flush", arm), ("shared", ctl))},
            "families_per_day": {a: round(x, 2) for a, x in fam.items()}, "verdict": verdict, "share": SHARE}


def report_text(fnf: Mapping[str, Any]) -> list[str]:
    """The meta-report's fork-and-flush section."""
    out = []
    d = fnf.get("diversity") or {}
    if d:
        ba = d.get("by_arm") or {}
        out.append(f"diversity: median NN similarity of new ideas (last {WINDOW_H:g} h) {d.get('median', 0):.2f} over "
                   f"{d.get('n', 0)} (trigger > {NN_SIM_MAX:g}); flush {ba.get('flush', {}).get('median')}, shared "
                   f"{ba.get('shared', {}).get('median')}")
    if a := fnf.get("ab"):
        c, g = a["credit_per_gpu_h"], a["gpu_h"]
        out.append(f"A/B (share {a['share']:g}): {a['rounds']} closed rounds (winners {', '.join(a['winners']) or '-'}); "
                   f"credit/GPU-h flush {c['flush']:.2e} ({a['nodes']['flush']} nodes, {a['tested']['flush']} tested, "
                   f"{g['flush']} GPU-h) vs shared {c['shared']:.2e} ({a['nodes']['shared']} nodes, "
                   f"{a['tested']['shared']} tested, {g['shared']} GPU-h); diff {a['diff']:+.2e} (SE "
                   f"{a['se'] if a['se'] is not None else '-'}); families/day {a['families_per_day']}; {a['verdict']}")
    if o := fnf.get("open"):
        out.append(f"open round {o['id']} ({o['trigger']}): branches {', '.join(o['branches'])}, cap {o['cap']:.0f} GPU-h "
                   f"each, sessions {o.get('sessions') or {}}")
    return out
