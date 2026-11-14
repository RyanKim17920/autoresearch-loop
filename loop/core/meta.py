"""loop.core.meta — the 3-hour meta-cycle (docs/loop-core-v3.md §7): measure the loop, let one Opus call propose,
accept only what a held-out replay can show is better.

1. :func:`report` -> :class:`MetaReport` (computed, no LLM): yield per source (test-block gains per GPU-h), calibration
   of the selection-adjusted test belief against the fresh outcome (and the refit max-of-K β), selection survival,
   test precision, GPU-h by purpose, engagement hit rates, intake reject reasons, LLM sessions, plateau, noise vs the
   last report. Written to ``<state>/meta/<ts>.md`` (+ ``.json``) every ``core_meta.every_h`` of loop wall time.
2. Meta-review: one Opus call (no tools; ``prompts/meta_review.md``) on the report + the TUNABLE config + the patchable
   prompts -> ``[{kind: config|prompt, path, value|patch, rationale, predicted_effect}]``.
3. Acceptance: a config change only through :func:`backtest_config` -- the posterior replay (``core.backtest.replay``,
   each point fitted strictly before its fresh outcome) evaluated on the last ``split_h`` hours: replay value (fresh
   panel gain of the tests the config would open), with ranking ρ, calibration, coverage and robustness harm as
   guards; accepted iff the value gain >= 2 bootstrap SE and no guard is worse by more than one SE. A prompt change
   becomes an A/B arm on ``ab_share`` of the research sessions, judged by fresh-seed credit after
   ``ab_min_sessions`` sessions per arm.
4. Locks: the meta process writes only ``<state>/loop_config/v<N>.yaml`` and ``<state>/prompts/v<N>/``; a proposal
   naming a LOCKED key / file / the adapter's locked surface is refused (:func:`vet`). Every tick :class:`Locks`
   compares the git-blob hashes of ``spec.LOCKED_FILES`` with the deployed HEAD; a mismatch trips the breaker (no
   launch, no session) until a clean redeploy.
5. Versions: ``config_versions`` rows; the next report auto-reverts a config version whose guard went out of bounds;
   ``loop revert v<N>`` by hand. One summary line per cycle goes to the loop log."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence

import yaml

from loop import project as P
from loop import spec as SP
from loop.core import backtest as B
from loop.core import ideation as ID
from loop.core import model as M
from loop.core import test as T
from loop.core.data import PANEL, ROOT, Ledger

PURPOSES = ("idea", "seed", "wave", "test", "rebase", "port", "fill", "void", "crash")
#: guard metric -> (lo, hi): the next report reverts a config version that moved one out of bounds
GUARD_BOUNDS = {"slope": (0.5, 1.5), "coverage80": (0.6, 0.95), "rho": (0.15, 1.0), "false_pass": (0.0, 0.2),
                "harm_rate": (0.0, 0.25), "type1_sim": (0.0, 0.15)}


def guard_bounds() -> dict[str, tuple[float, float]]:
    """GUARD_BOUNDS with the simulated type-I's upper bound from ``test.type1_guard_max`` (``test.TYPE1_GUARD_MAX``):
    gs2 spends exactly alpha at the fitted sigma_d, so its 1.25x-misfit stress sits at ~.155 -- a fixed .15 would
    flip in / out on the sim's own noise and auto-revert unrelated config versions."""
    return {**GUARD_BOUNDS, "type1_sim": (0.0, float(T.TYPE1_GUARD_MAX))}
MIN_EVAL = 12                   # held-out points a config backtest needs (fewer in split_h: the last MIN_EVAL)
N_BOOT = 400
HARM_YARDSTICK = 0.004          # the robustness loss the harm guard counts (fixed: not the tuned knob)


def harm_yardstick() -> float:
    """The guard-axis loss the harm metric counts: half the project's harm limit on its guard axis (nanopath: rob
    .008 / 2 = ``HARM_YARDSTICK`` .004 exactly; a project's own HARM_H scales it); no guard: ``HARM_YARDSTICK``
    (unused: no guard means no harm term)."""
    g = P.guard()
    h = P.HARM_H.get(g) if g is not None else None
    return HARM_YARDSTICK if h is None else float(h) / 2.0
MAX_GAP_S = 600.0               # loop wall time: a gap longer than this between ticks (the loop was down) not counted
K_CLOCK, K_LAST, K_BREAKER, K_AB, K_CHECK = "meta.clock", "meta.last", "meta.breaker", "meta.ab", "meta.check"
K_REVIEW_PENDING = "meta.review_pending"   # a meta-review the LLM pause deferred: {ts, report}
K_NOBT = "meta.not_backtestable"   # v3-c6a: {key: {ts, n, reason}} config keys the replay cannot evaluate (gain 0, SE 0)
PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def _prompt(name: str) -> Path:
    """A prompt file: the project's domain overlay (loop.project.prompt_path) first, else :data:`PROMPTS`."""
    p = P.prompt_path(name)
    return p if p.parent != P.PROMPTS else PROMPTS / p.name
REPO = Path(__file__).resolve().parents[2]

REVIEW_SCHEMA = {"type": "object", "required": ["diagnosis", "proposals"], "properties": {
    "diagnosis": {"type": "string", "minLength": 1},
    "proposals": {"type": "array", "items": {
        "type": "object", "required": ["kind", "path", "rationale", "predicted_effect"], "properties": {
            "kind": {"type": "string"}, "path": {"type": "string"}, "value": {},
            "patch": {"type": "object"}, "rationale": {"type": "string"}, "predicted_effect": {"type": "string"}}}}}}


# ---------------------------------------------------------------------------------------------- 1. the report
@dataclass
class MetaReport:
    ts: float
    since: float                                   # the previous report (window start)
    incumbent: str = ""
    epoch: str = ""
    sources: dict[str, dict[str, float]] = field(default_factory=dict)
    calibration: dict[str, float] = field(default_factory=dict)
    survival: dict[str, float] = field(default_factory=dict)
    tests: dict[str, float] = field(default_factory=dict)
    gpu_h: dict[str, list[float]] = field(default_factory=dict)          # purpose -> [window, all]
    engagement: dict[str, list[int]] = field(default_factory=dict)       # source -> [engaged, measured]
    intake: dict[str, list[int]] = field(default_factory=dict)           # reason -> [window, all]
    llm: dict[str, float] = field(default_factory=dict)
    plateau: dict[str, float] = field(default_factory=dict)
    noise: dict[str, dict[str, float]] = field(default_factory=dict)     # target -> {sigma, rho[, d_sigma, d_rho]}
    config: dict[str, float] = field(default_factory=dict)
    fnf: dict[str, Any] = field(default_factory=dict)                    # fork-and-flush: diversity, A/B, open round
    bar: dict[str, Any] = field(default_factory=dict)                    # the external bar (core.bar): bar, best, beats
    bar_blind: bool = False                                              # bar was ranked on the official ledger != visible
    bar_vis: dict[str, Any] = field(default_factory=dict)                # the same on the VISIBLE ledger: the LLM copy's
    switches: list[str] | None = None                                    # incumbent switches in the window ("node +gain"); None = not computed
    holdout: dict[str, Any] = field(default_factory=dict)                # v3-c6a: the blind holdout audit (core.holdout)
    alloc: dict[str, Any] = field(default_factory=dict)                  # v3-c7i: budget allocation (core.alloc)
    originality: dict[str, Any] = field(default_factory=dict)            # v3-c8b: the originality mix (core.bold)
    alt_base: dict[str, Any] = field(default_factory=dict)               # v3-c9 C2: the alt-base arm vs incumbent-based
    top_new: list[str] = field(default_factory=list)                     # the window's best test verdicts ("node KIND +gain (n=)"): the digest
    components: dict[str, Any] = field(default_factory=dict)             # 10-22: the causal component audit (loop.components.v3_section)
    errors: dict[str, str] = field(default_factory=dict)

    def guards(self) -> dict[str, float]:
        g = {k: self.calibration[k] for k in ("slope", "coverage80", "rho") if k in self.calibration}
        g.update({k: self.tests[k] for k in ("false_pass", "harm_rate", "type1_sim") if k in self.tests})
        return g

    def out_of_bounds(self) -> list[str]:
        gb = guard_bounds()
        return [f"{k} {v:.3g} not in [{gb[k][0]}, {gb[k][1]}]" for k, v in self.guards().items()
                if not gb[k][0] <= v <= gb[k][1]]

    def markdown(self, llm: bool = False) -> str:
        """The report (``llm``: the LLM reviewer's copy -- v3-c6r H2: without the holdout audit's numbers)."""
        return render_markdown(self, llm=llm)


def purpose_of(run: Any, led: Ledger, rank: Mapping[str, int], idea_runs: int = 3,
               fill: Collection[str] = frozenset()) -> str:
    """idea / seed / wave / test / rebase / port / fill / void / crash (``rank``: run id -> its index among the node's
    selection runs, by submission; ``fill``: the run ids launched as idle fill, the ``run_fill`` events)."""
    node = led.nodes.get(run.node)
    if run.status in ("crash", "void"):
        return run.status
    if run.id in fill:
        return "fill"
    if run.stage in ("confirm", "audit", "remint"):
        return "test"
    if node is not None and node.port_of:
        return "port"
    if run.stage in ("baseline", "calibrate") or node is None or node.parent is None:
        return "rebase"
    if node.lane == "factorial" or "wave=" in node.notes:
        return "wave"
    if run.stage in ("screen", "promote") and rank.get(run.id, idea_runs) < idea_runs:
        return "idea"
    return "seed"


def gpu_by_purpose(led: Ledger, since: float, fill: Collection[str] = frozenset()) -> dict[str, list[float]]:
    rank: dict[str, int] = {}
    seen: dict[str, int] = {}
    for r in sorted(led.runs, key=lambda r: r.submitted_ts):
        if r.stage in ("screen", "promote"):
            rank[r.id] = seen.get(r.node, 0)
            seen[r.node] = rank[r.id] + 1
    out = {p: [0.0, 0.0] for p in PURPOSES}
    for r in led.runs:
        p = purpose_of(r, led, rank, fill=fill)
        out[p][1] += r.gpu_h
        if r.done_ts >= since:
            out[p][0] += r.gpu_h
    return {p: [round(a, 1), round(b, 1)] for p, (a, b) in out.items()}


def siblings_k(led: Ledger, node: str, ts: float) -> int:
    """K of the max-of-K adjustment: the node's siblings (same parent) with selection data before ``ts`` (<= 16)."""
    parent = led.nodes[node].parent
    have = {o.node for o in led.obs if o.ts < ts}
    return max(1, min(16, sum(1 for n in led.nodes.values() if n.parent == parent and n.id in have)))


@dataclass(frozen=True)
class EvalPoint:
    """A backtest point with what a config decision needs: the max-of-K K and the robustness harm belief."""
    p: B.Point
    k: int
    rob: tuple[float, float]           # selection-data CUPED robustness increment (mean, var)
    fresh_rob: float


def eval_points(led: Ledger, points: Sequence[B.Point]) -> list[EvalPoint]:
    """Robustness estimates use the full-history noise (σ / ρ only; each point's means stay pre-outcome)."""
    noise, _ = B.fit_at(led, led.change_parents())
    by = led.by_node()
    g = P.guard()                                     # the guard axis (nanopath: rob); None = no harm term
    out = []
    for p in points:
        par = led.nodes[p.node].parent
        sel = [o for o in by.get(p.node, []) if o.ts < p.ts]
        fresh = [o for o in by.get(p.node, []) if o.stage in ("confirm", "audit", "remint")]
        rob = M.cuped(sel, [o for o in by.get(par, []) if o.ts < p.ts], g, noise) if sel and g else None
        fr = M.cuped(fresh, by.get(par, []), g, noise) if fresh and g else None
        out.append(EvalPoint(p, siblings_k(led, p.node, p.ts), (rob.delta, rob.var) if rob else (0.0, 1.0),
                             fr.delta if fr else 0.0))
    return out


def m_sel(e: EvalPoint, beta: float) -> float:
    return e.p.m_c - beta * math.sqrt(e.p.v_c) * emax(e.k)          # = model.max_of_k, E_K cached


@functools.lru_cache(maxsize=None)
def emax(k: int) -> float:
    return M.expected_max(k)


def calibration(pts: Sequence[EvalPoint], beta: float) -> dict[str, float]:
    """Selection-adjusted composite belief (m_sel at the live β) vs the fresh composite / panel outcome, and the β
    that would have made m_sel unbiased (least squares on a = √v E_K)."""
    if len(pts) < 3:
        return {"n": len(pts)}
    ms = [m_sel(e, beta) for e in pts]
    blk = B.block(ms, [e.p.v_c for e in pts], [e.p.fresh_c for e in pts], [e.p.fresh_c_var for e in pts],
                  [e.p.fresh for e in pts])
    a = [math.sqrt(e.p.v_c) * emax(e.k) for e in pts]
    num = sum((e.p.m_c - e.p.fresh_c) * ai for e, ai in zip(pts, a))
    den = sum(ai * ai for ai in a)
    return {**blk, "beta": beta, "beta_fit": num / den if den > 0 else beta}


def _since(store: Any, ev: str, since: float) -> list[dict[str, Any]]:
    return [e for e in store.events(ev) if e["ts"] >= since]


def llm_sessions(store: Any, since: float) -> dict[str, float]:
    start = {e["request_id"]: e["ts"] for e in store.events("research_started")}
    end: dict[str, tuple[float, str]] = {}
    for ev in ("intake", "research_failed", "research_error"):
        for e in store.events(ev):
            rid = e.get("request_id")
            # a re-issued request has several starts and ends (live 10-21: "-8.18 h wall"): a session ends at the first
            # end AT OR AFTER its (latest) start, never at an end of an earlier attempt
            if rid in start and e["ts"] >= start[rid] and (rid not in end or e["ts"] < end[rid][0]):
                end[rid] = (e["ts"], ev)
    win = [r for r, t in start.items() if t >= since]
    wall = [end[r][0] - start[r] for r in win if r in end]
    return {"started": len(win), "finished": sum(r in end for r in win),
            "failed": sum(r in end and end[r][1] != "intake" for r in win), "started_all": len(start),
            "wall_h": round(sum(wall) / 3600, 2), "median_min": round(sorted(wall)[len(wall) // 2] / 60, 1) if wall else 0}


def intake_reasons(store: Any, since: float) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    rows = [(e["ts"], str(e.get("reason") or "?").split(":")[0]) for e in store.events("intake") if not e.get("admitted")]
    rows += [(e["ts"], f"failed:{e.get('status')}") for e in store.events("research_failed")]
    for ts, reason in rows:
        c = out.setdefault(reason[:60], [0, 0])
        c[1] += 1
        c[0] += ts >= since
    return dict(sorted(out.items(), key=lambda kv: -kv[1][1]))


def noise_fit(led: Ledger, post: Any) -> dict[str, dict[str, float]]:
    cur = led.in_epoch()
    out = {t: {"sigma": round(math.sqrt(post.noise.sigma2[t]), 5), "rho": round(post.noise.rho[t], 3)}
           for t in (*post.noise.sigma2,)}
    try:
        far, _ = B.fit_at(cur, cur.far_parents())
        out["far"] = {"sigma": round(far.sd(PANEL), 5), "rho": round(far.rho[PANEL], 3)}
    except (ValueError, ZeroDivisionError, KeyError):
        pass
    return out


#: a pair's σ_d over its change class's pooled σ_d in the simulated type-I (heterogeneity within a class: the sim saw
#: ρ .4 and .93 code changes pooled into one far fit)
TYPE1_MISFIT = 1.25


def type1_sim(led: Ledger, n_sims: int = 1000, misfit: float = TYPE1_MISFIT) -> dict[str, float]:
    """The simulated type-I of the test AS THE LOOP RUNS IT now: a far-class candidate's decision noise on the current
    epoch (``backtest.epoch_noise``), its σ_d re-drawn from its estimation uncertainty (``df``) and the pair's true
    σ_d ``misfit`` x the class's (``test.simulate_fit``) -- not the assumed value. ``type1_sim`` at ``misfit``,
    ``type1_sim_exact`` with the class σ_d exact."""
    nz = B.epoch_noise(led, math.inf, led.epoch)
    if nz is None:
        raise ValueError("no decision noise")
    sd, df = nz.sd_d(PANEL), nz.df.get(PANEL, math.inf)
    return {"type1_sim": round(T.simulate_fit(0.0, sd, df, misfit, n_sims, seed=1)["pass"], 3),
            "type1_sim_exact": round(T.simulate_fit(0.0, sd, df, 1.0, n_sims, seed=1)["pass"], 3),
            "type1_sd_d": round(sd, 5), "type1_df": df}


def report(led: Ledger, store: Any, *, now: float, since: float = 0.0, post: Any = None, blocks: Any = None,
           values: Mapping[str, float] | None = None, points: Sequence[EvalPoint] | None = None,
           last: Mapping[str, Any] | None = None, bar: Any = None,
           bar_led: Ledger | None = None) -> tuple[MetaReport, list[EvalPoint]]:
    """The report and the backtest's evaluation points (reused by acceptance). Each section is computed on its own: a
    failure is recorded in ``errors``, never raised. ``bar``: the ``labless`` config (``bar_node`` / ``bar_score`` /
    ``bar_alpha``) for the external-bar section; ``bar_led``: the OFFICIAL full ledger that section ranks on (``led`` may be
    the holdout's visible-composite one, whose panel is on another scale: live 10-14 the report's best was .6715 while the
    heartbeat's, on the full ledger, was a node .6805), else ``led``."""
    from loop.core.acquire import Posterior
    values = SP.tunable_values(values)
    rep = MetaReport(ts=now, since=since, epoch=led.epoch[:7], config=dict(values))
    blocks = blocks if blocks is not None else ID.blocks_of(store, led)
    v = None

    def section(name: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - a section that cannot be computed is reported, not fatal
            rep.errors[name] = f"{type(exc).__name__}: {exc}"[:200]
            return None
    post = post if post is not None else section("posterior", lambda: Posterior(led, blocks))
    if post is not None:
        v = section("view", lambda: ID.view(led, blocks, post))
    if v is not None:
        rep.incumbent = v.incumbent
        rep.sources = section("sources", lambda: {k: {a: round(b, 6) for a, b in s.items()}
                                                   for k, s in ID.credit(v).items()}) or {}
        judged = [j for j in v.judged]
        rep.tests = {"closed": len(judged), "passed": sum(j.verdict.kind is T.Kind.PASS for j in judged),
                     "window": sum(j.ts >= since for j in judged),
                     "mean_n": round(sum(j.verdict.n for j in judged) / len(judged), 2) if judged else 0.0,
                     "harm_rate": round(sum(bool(j.verdict.harm) for j in judged) / len(judged), 3) if judged else 0.0}
        rep.switches = [f"{j.node} {j.verdict.mean:+.4f} (n={j.verdict.n})" for j in judged
                        if j.switched and j.ts >= since and j.verdict.kind is T.Kind.PASS]
        rep.top_new = [f"{j.node} {getattr(j.verdict.kind, 'name', j.verdict.kind)} {j.verdict.mean:+.4f} (n={j.verdict.n})"
                       for j in sorted((j for j in judged if j.ts >= since), key=lambda j: -j.verdict.mean)[:5]]
        rep.tests["pass_rate"] = round(rep.tests["passed"] / len(judged), 3) if judged else 0.0
        # test.local_reroot: passes of the local-incumbent tier (outside gs2's alpha; type1_sim covers gs2 only)
        rep.tests["local_passes"] = sum(j.verdict.kind is T.Kind.PASS and str(j.verdict.reason).startswith("local re-root")
                                        for j in judged)
        rep.tests["gs2_passed"] = rep.tests["passed"] - rep.tests["local_passes"]   # the passes type1_sim prices
        cc = section("false_pass", lambda: B.check_c(led, v.noise))       # each audit: its own epoch's noise
        if cc is not None:
            rep.tests["false_pass"] = round(cc["false_pass_rate"], 3)
            rep.tests["audited_passes"] = cc["passes"]
        sim = section("type1", lambda: type1_sim(led))
        if sim is not None:
            rep.tests.update(sim)
        p = section("plateau", lambda: ID.plateau(led, v.judged))
        if p is not None:                  # median inf = no PASS yet (a thin / fresh ledger): None, not round(inf)
            rep.plateau = {"active": p.active, "gpu_h": round(p.gpu_h),
                           "median": round(p.median) if math.isfinite(p.median) else None, "passes": p.passes}
        rep.noise = section("noise", lambda: noise_fit(led, post)) or {}
        for t, row in rep.noise.items():
            old = ((last or {}).get("noise") or {}).get(t)
            if old:
                row["d_sigma"], row["d_rho"] = round(row["sigma"] - old["sigma"], 5), round(row["rho"] - old["rho"], 3)
    if points is None:
        points = section("calibration", lambda: eval_points(led, B.replay(led))) or []
    rep.calibration = section("calibration", lambda: calibration(points, values["acquire.max_of_k_beta"])) or {}
    rep.calibration["window"] = sum(e.p.ts >= since for e in points)
    if points:
        sel = [e.p for e in points if e.p.raw > 0]
        rep.survival = {"n": len(sel), "ratio": round(sum(p.fresh for p in sel) / sum(p.raw for p in sel), 3)
                        if sel else 0.0, "positive": round(sum(p.fresh > 0 for p in sel) / len(sel), 3) if sel else 0.0}
    fill = section("fill", lambda: frozenset(str(e.get("run_id")) for e in store.events("run_fill"))) or frozenset()
    rep.gpu_h = section("gpu_h", lambda: gpu_by_purpose(led, since, fill)) or {}
    rep.engagement = section("engagement", lambda: {k: list(x) for k, x in ID.engagement_rates(
        ID.engagements(store, led), led).items()}) or {}
    rep.intake = section("intake", lambda: intake_reasons(store, since)) or {}
    rep.llm = section("llm", lambda: llm_sessions(store, since)) or {}
    rep.fnf = section("fnf", lambda: fnf_section(led, store, v, now)) or {}
    if v is not None:
        from loop.core import altbase as AB
        rep.alt_base = section("alt_base", lambda: AB.section(None, led, store, v)) or {}
    from loop.core.holdout import K_AUDIT
    rep.holdout = section("holdout", lambda: dict(store.get_state(K_AUDIT) or {})) or {}
    from loop.core import bold as BOLD
    rep.originality = section("originality", lambda: BOLD.mix(None, store, since)) or {}
    if getattr(bar, "bar_node", "") or getattr(bar, "leader_median", None) is not None:
        rep.bar = section("bar", lambda: bar_section(bar_led if bar_led is not None else led, bar,
                                                   bar_incumbent(store, rep.incumbent))) or {}
        if bar_led is not None and bar_led is not led:      # blind holdout: the LLM copy never sees the official panel
            rep.bar_blind = True
            rep.bar_vis = section("bar_vis", lambda: bar_section(led, bar, bar_incumbent(store, rep.incumbent))) or {}
    return rep, list(points)


def bar_incumbent(store: Any, view_incumbent: str = "") -> str | None:
    """The incumbent the bar section ranks against: the tick's status-derived one (state ``tick.incumbent``, what the
    heartbeat's ``best=`` uses), else the view's replay-derived one (they differed live 06:30Z: the report showed a node
    .6698 while the heartbeat's best was a node .6808), else None (the ledger's)."""
    got = store.get_state("tick.incumbent") if hasattr(store, "get_state") else None
    return (got if isinstance(got, str) and got else None) or view_incumbent or None


def bar_section(led: Ledger, cfg: Any, incumbent: str | None = None) -> dict[str, Any]:
    """The external bar (``core.bar``, Labless's acceptance rule): the target median, the local mean needed, the best
    current candidate with its P(promote), and every node that beats it -- the operator's "real improvement". The
    best is ``bar.ranked``'s first (the paired posterior, v3-c7a A1): the heartbeat's ``best=`` is the same ranking."""
    from loop.core import bar as BR
    b = BR.bar_from(led, cfg)
    if b is None:
        return {"text": f"bar={cfg.bar_node} (not measurable)", "best": "", "beats": []}
    ranked = BR.ranked(led, b, incumbent=incumbent or None,
                       eb_min_nodes=int(getattr(cfg, "eb_min_nodes", BR.EB_MIN_NODES)))
    try:
        sig = BR.epoch_sigma(led)
    except ValueError:
        sig = None
    return {"text": BR.bar_text(b, sig) + (" (Labless median of 3 fresh seeds)" if b.lab else " (the bar node's local mean)"),
            "alpha": b.p, "best": BR.beat_text(ranked[0], b) if ranked else "",
            "beats": [BR.beat_text(x, b) for x in ranked if x.beats]}


def fnf_section(led: Ledger, store: Any, v: Any, now: float) -> dict[str, Any]:
    """Fork-and-flush (``core.fnf``): the diversity metric (median NN similarity of the last 48 h of new ideas,
    overall and per arm), the arm-vs-control A/B over the closed rounds, the open round."""
    from loop.core import fnf as FN
    st = FN.state(store)
    out: dict[str, Any] = {"diversity": FN.nn_similarity(store, led, now)}
    if v is not None:
        out["ab"] = FN.ab(v, st)
    if st.get("open"):
        out["open"] = {k: st["open"].get(k) for k in ("id", "trigger", "branches", "cap", "sessions")}
    return out


def _utc(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))


def render_markdown(r: MetaReport, llm: bool = False) -> str:
    c, s, t = r.calibration, r.survival, r.tests
    out = [f"# Meta report {_utc(r.ts)} (window since {_utc(r.since) if r.since else 'start'})",
           f"incumbent {r.incumbent or '?'} on base {r.epoch or '?'}; config "
           + ", ".join(f"{k}={v:g}" for k, v in r.config.items()), "",
           "## Yield per source (fresh-seed test gains per GPU-h)", "source | nodes | tested | gain | GPU-h | gain/GPU-h",
           "---|---|---|---|---|---"]
    top = sorted(r.sources.items(), key=lambda kv: (-kv[1].get("per_gpu_h", 0), -kv[1].get("gpu_h", 0)))[:12]
    out += [f"{k} | {int(x['nodes'])} | {int(x['tested'])} | {x['gain']:+.4f} | {x['gpu_h']:.0f} | {x['per_gpu_h']:.2e}"
            for k, x in top]
    if c.get("n", 0) >= 3:
        out += ["", "## Calibration (selection-adjusted belief m_sel vs fresh outcome)",
                f"n {c['n']} ({c.get('window', 0)} in window): rho {c['rho']:.3f} vs fresh panel, slope {c['slope']:.2f}, "
                f"80% coverage {c['coverage80']:.2f}, bias {c['bias']:+.5f}; beta {c['beta']:g} -> refit {c['beta_fit']:.2f}"]
    if s:
        out.append(f"Selection survival: fresh/selected panel gain {s['ratio']:.2f} over {s['n']} selected nodes; "
                   f"{s['positive']:.0%} positive on fresh seeds")
    if t:
        out += ["", "## Tests", f"closed {t['closed']} ({t['window']} in window), passed {t['passed']} "
                f"(rate {t['pass_rate']:.2f}), mean seeds {t['mean_n']}, robustness-harm {t['harm_rate']:.2f}; "
                f"audited false-pass {t.get('false_pass', float('nan')):.2f}, simulated type-I {t.get('type1_sim', float('nan')):.3f} (sigma_d x{TYPE1_MISFIT:g}; "
                f"exact sigma_d {t.get('type1_sim_exact', float('nan')):.3f})"]
        if t.get("local_passes"):
            out.append(f"Of the {t['passed']} passes, {t['local_passes']} came from the LOCAL re-root tier "
                       f"(test.local_reroot) and {t.get('gs2_passed', t['passed'] - t['local_passes'])} from gs2: the "
                       "simulated type-I above prices gs2 only; local passes carry their own, unpriced, extra type-I "
                       "(fresh pairs z >= fresh_z on top of an EB-shrunk gate; simulated ~3% on a selected true-zero)")
    out += ["", "## GPU-h by purpose (window / all)",
            " | ".join(f"{p} {a:.0f}/{b:.0f}" for p, (a, b) in r.gpu_h.items())]
    if r.engagement:
        out += ["", "## Mechanism engagement (engaged / measured)",
                "; ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(r.engagement.items()))]
    if r.intake:
        out += ["", "## Intake rejects / failed sessions (window / all)",
                "; ".join(f"{k} {a}/{b}" for k, (a, b) in list(r.intake.items())[:10])]
    if r.llm:
        out += ["", f"## LLM sessions: {r.llm['started']} started in window ({r.llm['finished']} finished, "
                    f"{r.llm['failed']} failed), {r.llm['wall_h']} h wall, median {r.llm['median_min']} min; "
                    f"{r.llm['started_all']} all time"]
    bar = (r.bar_vis if r.bar_blind else r.bar) if llm else r.bar   # LLM copy: the visible-ledger section, else none
    if bar:
        out += ["", f"## External bar (Labless acceptance rule; needs P(promote) >= {bar.get('alpha', '?')})",
                *([f"(VISIBLE-task composite only: the held-out tasks are withheld, so every mean and the candidate "
                   "ranking here are on a different scale and order than the official heartbeat / operator figures "
                   "-- never compare them across, e.g. a 'best candidate' here is not the heartbeat's best=)"]
                  if llm and r.bar_blind else []),
                bar["text"], f"best current candidate: {bar['best'] or 'none'}",
                "pause-rule improvement (incumbent switches in window): " + ("; ".join(r.switches or []) or "none"),
                "beats the Labless bar (submission gate): " + ("; ".join(bar["beats"]) or "none")]
    if r.plateau:
        p = r.plateau
        out += ["", f"## Plateau: {'ACTIVE' if p['active'] else 'no'} -- {p['gpu_h']} GPU-h since the last pass "
                    + (f"(median {p['median']} between {p['passes']} passes)" if p['median'] is not None else "(no pass yet)")]
    if r.noise:
        out += ["", "## Noise (current epoch; d = change since the last report)",
                "; ".join(f"{k} sd {x['sigma']:.4f} rho {x['rho']:.2f}" + (
                    f" (d {x['d_sigma']:+.4f}/{x['d_rho']:+.2f})" if "d_sigma" in x else "") for k, x in r.noise.items())]
    if r.alloc:
        from loop.core.alloc import meta_text
        out += ["", "## Budget allocation (v3-c7i: yield per GPU-h by kind, GPU split, backlog, clusters)",
                *meta_text(r.alloc)]
    if r.originality:
        from loop.core.bold import mix_text
        out += ["", "## Originality mix (new research nodes in window: share per structural category; knob-only = "
                    "diff changes only config values / constants of existing code)", *mix_text(r.originality)]
    if (r.components or {}).get("rows"):
        from loop.components import v3_text
        out += ["", "## Component audit (causal: optional v3 context parts bypassed on a seeded share of sessions)",
                *v3_text(r.components)]
    if r.fnf:
        from loop.core.fnf import report_text
        out += ["", "## Fork-and-flush (flushed branches vs the shared context)", *report_text(r.fnf)]
    if r.alt_base and ((r.alt_base.get("alt_base") or {}).get("nodes") or (r.alt_base.get("root") or {}).get("nodes")):
        from loop.core.altbase import report_text as alt_text
        out += ["", "## Alt-base arm (sessions started from a non-incumbent base vs incumbent-based sessions)",
                *alt_text(r.alt_base)]
    if r.holdout and not llm:                  # v3-c6r H2: operator-only (the LLM reviewer never sees held-out numbers)
        from loop.core.holdout import report_text as holdout_text
        out += ["", "## Holdout audit (visible vs held-out probe tasks of each passed fresh-seed test)",
                *holdout_text(r.holdout)]
    if bad := r.out_of_bounds():
        out += ["", "## GUARDS OUT OF BOUNDS: " + "; ".join(bad)]
    if r.errors:
        out += ["", "## Not computed: " + "; ".join(f"{k}: {v}" for k, v in r.errors.items())]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------- 3. acceptance
def _policy(pts: Sequence[EvalPoint], values: Mapping[str, float], V: float) -> list[int]:
    """The tests a config would open among the eval points: harm-allowed, then the better half by EI(m_sel)."""
    from loop.core.acquire import ei, harm_ok
    h, q, beta = values["acquire.harm_h_rob"], values["acquire.harm_q"], values["acquire.max_of_k_beta"]
    g = P.guard()                                     # no guard axis: every point is harm-allowed
    ok = [i for i, e in enumerate(pts) if g is None or harm_ok({g: e.rob}, {g: h}, q)]
    ok.sort(key=lambda i: -ei(m_sel(pts[i], beta), pts[i].p.v_c, V, 0.0))
    return ok[:max(1, math.ceil(len(pts) / 2))]


def metrics(pts: Sequence[EvalPoint], values: Mapping[str, float], V: float) -> dict[str, float]:
    """value: Σ max(0, fresh panel) of the opened tests per test slot (budget = half the points); guards: ranking ρ,
    calibration error |slope − 1|, coverage error |cov80 − .8|, robustness harm among the opened tests."""
    picks, budget = _policy(pts, values, V), max(1, math.ceil(len(pts) / 2))
    beta = values["acquire.max_of_k_beta"]
    cal = calibration(pts, beta)
    return {"value": sum(max(0.0, pts[i].p.fresh) for i in picks) / budget,
            "rho": cal.get("rho", 0.0), "cal_err": abs(cal.get("slope", 1.0) - 1), "cov_err": abs(cal.get("coverage80", .8) - .8),
            "harm": sum(pts[i].fresh_rob < -harm_yardstick() for i in picks) / len(picks) if picks else 0.0}


#: guard -> +1 if larger is better, -1 if smaller is better
GUARD_SIGN = {"rho": 1, "cal_err": -1, "cov_err": -1, "harm": -1}


def split(pts: Sequence[EvalPoint], split_h: float) -> list[EvalPoint]:
    if not pts:
        return []
    t_end = max(e.p.ts for e in pts)
    held = [e for e in pts if e.p.ts >= t_end - split_h * 3600]
    return held if len(held) >= MIN_EVAL else sorted(pts, key=lambda e: e.p.ts)[-MIN_EVAL:]


def backtest_config(pts: Sequence[EvalPoint], cur: Mapping[str, float], new: Mapping[str, float], *,
                    split_h: float = 72.0, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Accept iff the replay value of ``new`` beats ``cur`` by >= 2 bootstrap SE (and > 0) and no guard is worse by
    more than one SE, on the held-out split (the last ``split_h`` hours of fresh outcomes, at least MIN_EVAL)."""
    ev = split(pts, split_h)
    if len(ev) < 3:
        return {"accept": False, "reason": f"only {len(ev)} held-out points", "n": len(ev)}
    V = sorted(e.p.fresh_c_var for e in ev)[len(ev) // 2]
    a, b = metrics(ev, cur, V), metrics(ev, new, V)
    rng, diffs = random.Random(seed), {k: [] for k in a}
    for _ in range(n_boot):
        s = [ev[rng.randrange(len(ev))] for _ in ev]
        ma, mb = metrics(s, cur, V), metrics(s, new, V)
        for k in a:
            diffs[k].append(mb[k] - ma[k])
    se = {k: (sum((x - sum(d) / len(d)) ** 2 for x in d) / (len(d) - 1)) ** .5 for k, d in diffs.items()}
    gain = b["value"] - a["value"]
    worse = [k for k, sg in GUARD_SIGN.items() if sg * (b[k] - a[k]) < -max(se[k], 1e-12)]
    ok = gain > 0 and gain >= 2 * se["value"] and not worse
    reason = (f"replay value {a['value']:.5f} -> {b['value']:.5f} (gain {gain:+.5f}, 2 SE {2 * se['value']:.5f})"
              + (f"; guard worse: {', '.join(worse)}" if worse else ""))
    return {"accept": ok, "reason": reason, "n": len(ev), "t_from": min(e.p.ts for e in ev), "cur": a, "new": b,
            "se": se, "worse": worse}


# ---------------------------------------------------------------------------------------------- 4. locks and vetting
def blob_hash(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Locks:
    """Git-blob hashes of the LOCKED files in the deployed HEAD (read once) vs the working tree (every check)."""

    def __init__(self, repo: Path = REPO, files: Sequence[str] = SP.LOCKED_FILES) -> None:
        self.repo, self.files, self.head, self.blobs, self.error = Path(repo), tuple(files), "", {}, ""
        try:
            self.head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                                       check=True, timeout=30).stdout.strip()
            out = subprocess.run(["git", "-C", str(repo), "ls-tree", self.head, "--", *self.files], capture_output=True,
                                 text=True, check=True, timeout=30).stdout
            self.blobs = {ln.split("\t", 1)[1]: ln.split()[2] for ln in out.splitlines() if "\t" in ln}
        except (OSError, subprocess.SubprocessError) as exc:
            self.error = f"locks unavailable: {exc}"[:200]

    def verify(self) -> list[str]:
        """The LOCKED files whose working-tree content differs from the deployed HEAD (or are missing from either)."""
        if self.error:
            return []
        bad = []
        for f in self.files:
            try:
                h = blob_hash((self.repo / f).read_bytes())
            except OSError:
                h = "missing"
            if self.blobs.get(f) != h:
                bad.append(f)
        return bad


@dataclass(frozen=True)
class Proposal:
    kind: str
    path: str
    value: Any = None
    patch: Mapping[str, Any] | None = None
    rationale: str = ""
    predicted_effect: str = ""


def _norm(path: str) -> str:
    return re.sub(r"^(\./)+", "", str(path).strip().replace("\\", "/"))


def vet(p: Proposal, surface_locked: Sequence[str] = ()) -> str:
    """'' if the proposal may go to acceptance, else why it is refused: an unknown kind, a LOCKED key, file or
    surface (adapter eval / probe locks), a path outside the meta-writable places, an out-of-bounds or unwired
    value, a malformed patch."""
    path = _norm(p.path)
    file = path.split("::")[0]
    if ".." in path.split("/") or file in SP.LOCKED_FILES or any(fnmatch(file, g) or fnmatch(Path(file).name, g)
                                                                  for g in surface_locked):
        return f"{path}: LOCKED path"
    if p.kind == "config":
        key = re.sub(r"^(state/)?loop_config/(v\d+\.yaml[:#.]?)?", "", path)
        if why := SP.check_tunable(key, p.value):
            return why
        target = SP.TUNABLE[key].target
        if ":" in target and not target.startswith(("rank:", "costs:")):
            mod, attr = target.split(":")
            import importlib
            if not hasattr(importlib.import_module(mod), attr):
                return f"{key}: not wired in this build ({mod} has no {attr})"
        return ""
    if p.kind == "prompt":
        name = Path(path).name
        if not re.fullmatch(r"((loop/)?prompts/|(state/)?prompts/v\d+/)?[\w.-]+\.md", path) or name not in SP.META_PROMPTS:
            return f"{path}: not a patchable prompt (allowed: {', '.join(SP.META_PROMPTS)})"
        pt = p.patch or {}
        if not isinstance(pt.get("find"), str) or not pt["find"] or not isinstance(pt.get("replace"), str):
            return f"{path}: a prompt patch is {{find, replace}} with a non-empty find"
        return ""
    return f"{path}: unknown proposal kind {p.kind!r} (config | prompt)"


def proposals_of(answer: Mapping[str, Any]) -> list[Proposal]:
    return [Proposal(str(x.get("kind", "")), str(x.get("path", "")), x.get("value"), x.get("patch"),
                     str(x.get("rationale", ""))[:600], str(x.get("predicted_effect", ""))[:300])
            for x in answer.get("proposals") or [] if isinstance(x, Mapping)]


# ---------------------------------------------------------------------------------------------- 5. versions
class Versions:
    """``config_versions`` rows + their files: config v<N> -> ``<state>/loop_config/v<N>.yaml`` (every TUNABLE value),
    prompt v<N> -> ``<state>/prompts/v<N>/`` (every patchable prompt). v0 is the defaults / the repo's prompts."""

    def __init__(self, store: Any, state_dir: str | Path, clock: Callable[[], float] = time.time) -> None:
        self.store, self.dir, self.clock = store, Path(state_dir), clock

    def rows(self, kind: str) -> list[dict[str, Any]]:
        return self.store.config_versions(kind)

    def active(self, kind: str) -> dict[str, Any]:
        rows = self.rows(kind)
        if not rows:
            self.store.put_config_version(kind, 0, parent=None, ts=self.clock(), diff={}, reason="defaults",
                                          backtest=None, state="active")
            if kind == "config":
                self._write_config(0, SP.tunable_values())
            rows = self.rows(kind)
        return next((r for r in reversed(rows) if r["state"] == "active"), rows[0])

    def values(self, v: int | None = None) -> dict[str, float]:
        v = self.active("config")["v"] if v is None else v
        path = self.dir / "loop_config" / f"v{v}.yaml"
        return SP.tunable_values((yaml.safe_load(path.read_text()) or {}).get("values") if path.exists() else None)

    def _write_config(self, v: int, values: Mapping[str, float], parent: int | None = None) -> Path:
        path = self.dir / "loop_config" / f"v{v}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"version": v, "parent": parent, "values": dict(values)}, sort_keys=True))
        return path

    def accept_config(self, diff: Mapping[str, float], reason: str, backtest: Mapping[str, Any]) -> int:
        cur = self.active("config")
        v = max(r["v"] for r in self.rows("config")) + 1
        self._write_config(v, {**self.values(cur["v"]), **diff}, cur["v"])
        self.store.set_config_version_state("config", cur["v"], "superseded")
        self.store.put_config_version("config", v, parent=cur["v"], ts=self.clock(), diff=diff, reason=reason,
                                      backtest=backtest, state="active")
        return v

    def prompt_dir(self, v: int) -> Path | None:
        """The dir serving prompt version v (None for v0 = the repo's prompts). A version is stored as its patch chain
        on top of the CURRENT repo prompts: the files are re-derived whenever they differ from that (a base prompt
        edited after the arm was minted is picked up on the next call, never served stale)."""
        if v == 0:
            return None
        d = self.dir / "prompts" / f"v{v}"
        d.mkdir(parents=True, exist_ok=True)
        for n in SP.META_PROMPTS:
            text = self.prompt_text(v, n)
            if not (d / n).exists() or (d / n).read_text() != text:
                (tmp := d / f".{n}.tmp").write_text(text)
                tmp.replace(d / n)                    # atomic: a session reading the file never sees half of it
        return d

    def skipped(self, v: int) -> list[str]:
        """The versions / files of v's patch chain whose find text no longer occurs exactly once (patch NOT applied)."""
        rows, out = {r["v"]: r for r in self.rows("prompt")}, []
        while v and (row := rows.get(v)) is not None:
            df = row["diff"] or {}
            if df.get("file") in SP.META_PROMPTS and df.get("find") and \
                    self.prompt_text(row["parent"] or 0, df["file"], rows).count(df["find"]) != 1:
                out.append(f"v{v}:{df['file']}")
            v = row["parent"] or 0
        return out

    def prompt_text(self, v: int, name: str, _rows: Mapping[int, Any] | None = None) -> str:
        """Prompt ``name`` at version v = the current repo file with the chain of version patches applied (a patch
        whose find text no longer occurs exactly once in the current text is skipped)."""
        if v == 0:
            return _prompt(name).read_text()
        rows = _rows or {r["v"]: r for r in self.rows("prompt")}
        row = rows.get(v)
        if row is None:
            return _prompt(name).read_text()
        text = self.prompt_text(row["parent"] or 0, name, rows)
        df = row["diff"] or {}
        if df.get("file") == name and df.get("find") and text.count(df["find"]) == 1:
            text = text.replace(df["find"], df.get("replace", ""))
        return text

    def new_prompt(self, p: Proposal, reason: str) -> int | str:
        """A prompt version = the active one with ``p.patch`` applied (the find text must occur exactly once)."""
        base = self.active("prompt")["v"]
        name = Path(_norm(p.path)).name
        text = self.prompt_text(base, name)
        if text.count(p.patch["find"]) != 1:
            return f"{name}: the patch's find text occurs {text.count(p.patch['find'])} times (needs exactly 1)"
        v = max(r["v"] for r in self.rows("prompt")) + 1
        d = self.dir / "prompts" / f"v{v}"
        d.mkdir(parents=True, exist_ok=True)
        for n in SP.META_PROMPTS:
            (d / n).write_text(text.replace(p.patch["find"], p.patch["replace"]) if n == name else self.prompt_text(base, n))
        self.store.put_config_version("prompt", v, parent=base, ts=self.clock(),
                                      diff={"file": name, **dict(p.patch)}, reason=reason, backtest=None, state="arm")
        return v

    def revert(self, kind: str, v: int, why: str) -> str:
        rows = {r["v"]: r for r in self.rows(kind)}
        if v not in rows:
            raise KeyError(f"no {kind} version v{v} (have {sorted(rows)})")
        cur = self.active(kind)
        if cur["v"] == v:
            return f"{kind} v{v} is already active"
        self.store.set_config_version_state(kind, cur["v"], "reverted")
        self.store.set_config_version_state(kind, v, "active")
        self.store.log("meta_revert", kind=kind, frm=cur["v"], to=v, why=why)
        return f"{kind} v{cur['v']} -> v{v} ({why})"


def apply(values: Mapping[str, float], core: Any) -> list[str]:
    """Install the TUNABLE values: rank keywords and Costs on the core, loop.core constants on their modules."""
    import importlib
    harm: dict[str, Any] = {}
    out = []
    for key, val in values.items():
        target = SP.TUNABLE[key].target
        kind, _, rest = target.partition(":")
        if kind == "rank":
            arg, _, sub = rest.partition(".")
            if arg == "harm_h" and sub == "rob":      # "the guard axis": nanopath rob; no guard = no harm limit
                if (g := P.guard()) is None:
                    continue
                sub = g
            if sub:
                harm.setdefault(arg, {})[sub] = val
            else:
                harm[arg] = val
        elif kind == "costs":
            if core is not None and hasattr(core, "costs"):
                core.costs = replace(core.costs, **{rest: type(getattr(core.costs, rest))(val)})
        else:
            mod = importlib.import_module(kind)
            if hasattr(mod, rest):
                setattr(mod, rest, val)
            else:
                out.append(f"{key} not wired ({kind}.{rest} missing)")
    if core is not None:
        core.harm = harm
    return out


# ---------------------------------------------------------------------------------------------- the hook
class Meta:
    """The tick's ``meta`` hook (and ``guard``: the lock check before any launch)."""

    def __init__(self, ctx: Any, store: Any, *, llm: Any = None, clock: Callable[[], float] = time.time,
                 dry: bool = False, locks: Locks | None = None) -> None:
        self.ctx, self.store, self.llm, self.clock, self.dry = ctx, store, llm, clock, dry
        self.cfg = getattr(getattr(ctx, "cfg", None), "core_meta", None) or SP.CoreMetaCfg()
        self.versions = Versions(store, getattr(ctx, "state_dir", "") or ".", clock)
        self.locks = locks if locks is not None else (Locks() if self.cfg.locks else None)
        self.applied: int | None = None
        self.points: list[EvalPoint] = []
        self._review: Any = None
        self._pool: Any = None                      # the meta-review's own one-worker pool (never a research slot)
        self._pending: MetaReport | None = self._load_pending()   # a review the LLM pause deferred (kept across a restart)
        self._lock_said = ""                        # the lock-guard state last announced

    PENDING_MAX_AGE_S = 48 * 3600.0     # a deferred review older than this is judged on numbers too stale to tune from

    def _load_pending(self) -> MetaReport | None:
        """The review ``K_REVIEW_PENDING`` recorded when the LLM pause deferred it, read back at start (10-22: it was
        written and never read, so a restart during a pause lost the review and ``meta-review deferred`` was a lie).
        The record names the report's ``.md``; the rebuilt report is its ``.json`` twin (:meth:`write`). Gone, unreadable
        or older than ``PENDING_MAX_AGE_S``: dropped (state cleared, ``meta_review_pending_dropped`` logged)."""
        rec = self.store.get_state(K_REVIEW_PENDING)
        if not rec:
            return None
        why = ""
        try:
            if self.clock() - float(rec.get("ts") or 0.0) > self.PENDING_MAX_AGE_S:
                why = "stale"
            elif float((self.store.get_state(K_LAST) or {}).get("ts") or 0.0) > float(rec.get("ts") or 0.0) + 1.0:
                why = "superseded"      # 10-22: a 33 h-old deferred report (empty window) was reviewed after a newer report
            else:
                raw = json.loads(Path(str(rec.get("report") or "")).with_suffix(".json").read_text())
                names = {f.name for f in fields(MetaReport)}
                return MetaReport(**{k: v for k, v in raw.items() if k in names})
        except Exception as exc:  # noqa: BLE001 - an unreadable record is dropped, never fatal at start
            why = f"unreadable: {type(exc).__name__}: {exc}"[:200]
        self.store.set_state(K_REVIEW_PENDING, None)
        self.store.log("meta_review_pending_dropped", why=why, report=str((rec or {}).get("report", "")))
        return None

    # -- locks -------------------------------------------------------------------------------------------------
    def guard(self, tick: Any = None) -> str | None:
        """None, or why nothing may launch: a LOCKED file differs from the deployed HEAD (the breaker trips).
        Loud: a guard that could not read the deployed HEAD at start (git failed) is DISABLED -- logged
        ``lock_guard_disabled`` once, an ALERT on stderr, ``lock=DISABLED`` in every heartbeat; a breach is an ALERT
        too. A clean check (a successful redeploy restart) clears both."""
        if self.locks is None:
            return None
        b = self.store.get_state(K_BREAKER) or {}
        if self.locks.error:
            if b.get("disabled") != self.locks.error:
                self.store.set_state(K_BREAKER, {**b, "disabled": self.locks.error, "ts": self.clock()})
                self.store.log("lock_guard_disabled", error=self.locks.error)
            self._say("DISABLED", f"ALERT: lock guard DISABLED ({self.locks.error}): LOCKED files are NOT checked "
                                  "against the deployed HEAD; restart the service once the checkout is readable")
            return None
        bad = self.locks.verify()
        if bad:
            if not b.get("active") or b.get("files") != bad:
                self.store.set_state(K_BREAKER, {"active": True, "files": bad, "head": self.locks.head, "ts": self.clock()})
                self.store.log("lock_breach", files=bad, head=self.locks.head)
            self._say("BREACH " + ",".join(bad), f"ALERT: lock BREACH: {', '.join(bad)} differ from the deployed HEAD "
                                                 f"{self.locks.head[:8]}: nothing launches until a clean redeploy")
            return f"BREAKER: locked file(s) differ from the deployed HEAD {self.locks.head[:8]}: {', '.join(bad)}"
        if b.get("active") or b.get("disabled"):
            self.store.set_state(K_BREAKER, {**{k: v for k, v in b.items() if k != "disabled"}, "active": False,
                                             "cleared": self.clock()})
            self.store.log("lock_guard_cleared", head=self.locks.head)
        self._lock_said = ""
        return None

    def _say(self, key: str, msg: str) -> None:
        if self._lock_said != key:
            self._lock_said = key
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.clock()))} {msg}", file=sys.stderr,
                  flush=True)

    def lock_state(self) -> str:
        """For the heartbeat: off (core_meta.locks false) | ok | DISABLED | BREACH(<files>)."""
        if self.locks is None:
            return "off"
        if self.locks.error:
            return "DISABLED"
        b = self.store.get_state(K_BREAKER) or {}
        return f"BREACH({','.join(b.get('files') or [])})" if b.get("active") else "ok"

    def _llm_blocked(self) -> str:
        """Why the meta-review may not call the LLM now ('' = it may): the operator / outage pause, or the probe."""
        from loop import runner as R
        if R.llm_paused(self.store):
            return "paused"
        return "probing" if R.llm_probing(self.store) else ""

    def review_pool(self, tick: Any = None) -> Any:
        """The meta-review's own pool (one worker, of the tick pool's executor class): it never takes a research
        session's worker or slot."""
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            cls = type(getattr(tick, "pool", None) or ThreadPoolExecutor(max_workers=1))
            self._pool = cls(max_workers=1)
        return self._pool

    # -- prompt A/B --------------------------------------------------------------------------------------------
    def arm_for(self, request_id: str) -> Path | None:
        """The prompt dir of a new research session: the A/B arm for ``ab_share`` of them, else the active version."""
        ab = self.store.get_state(K_AB) or {}
        if ab.get("arm") and (bad := self.versions.skipped(ab["arm"])):    # its patch no longer applies: end it
            self.store.set_config_version_state("prompt", ab["arm"], "stale")
            self.store.set_state(K_AB, {})
            self.store.log("meta_ab_stale", arm=ab["arm"], control=ab.get("control"), skipped=bad)
            ab = {}
        if not ab.get("arm"):
            return self.versions.prompt_dir(self.versions.active("prompt")["v"])
        side = "arm" if int(hashlib.sha1(request_id.encode()).hexdigest(), 16) % 1000 < 1000 * ab["share"] else "control"
        ab.setdefault("sessions", {})[request_id] = side
        self.store.set_state(K_AB, ab)
        return self.versions.prompt_dir(ab[side])

    def judge_ab(self, led: Ledger, judged: Sequence[Any], post: Any = None) -> str:
        """The prompt A/B: fresh-seed test credit per GPU-h of each side's nodes after ``ab_min_sessions`` sessions
        per arm. v3-c6a (meta.py:816 retired an arm on 0-vs-0 credit): equal test credit (0 vs 0 above all) is no
        information -- the SCREEN posterior credit (sum of each node's positive shrunk own mean, ``post.belief``) per
        GPU-h decides instead, once both sides have >= ``ab_min_screens`` screened nodes; until then nothing is
        retired or adopted. Still uninformative after ``ab_max_sessions_mult`` x ab_min_sessions: ended
        INCONCLUSIVE (the arm's version state ``inconclusive``; the control stays active)."""
        ab = self.store.get_state(K_AB) or {}
        if not ab.get("arm"):
            return ""
        done = {e.get("request_id") for e in self.store.events("intake")}
        n = {s: sum(1 for r, side in ab.get("sessions", {}).items() if side == s and r in done) for s in ("arm", "control")}
        if min(n.values()) < self.cfg.ab_min_sessions:
            return f"prompt A/B v{ab['arm']} vs v{ab['control']}: {n['arm']}/{n['control']} sessions"
        gain = {j.node: max(0.0, j.verdict.mean) for j in judged}
        req = {nid: m.group(1) for nid, nd in led.nodes.items() if (m := re.search(r"\brequest=(\S+)", nd.notes))}
        cred, scr, n_scr, gpus = {}, {}, {}, {}
        est = getattr(post, "est", None) or {}
        for s in ("arm", "control"):
            nodes = [nid for nid, r in req.items() if ab["sessions"].get(r) == s]
            gpu = sum(r.gpu_h for r in led.runs if r.node in nodes)
            gpus[s] = gpu
            cred[s] = sum(gain.get(x, 0.0) for x in nodes) / gpu if gpu else 0.0
            seen = [x for x in nodes if x in est]
            n_scr[s] = len(seen)
            scr[s] = sum(max(0.0, post.belief(x)[0]) for x in seen) / gpu if gpu and seen else 0.0
        basis = "fresh-seed test"
        min_scr = int(getattr(self.cfg, "ab_min_screens", 6))
        if cred["arm"] == cred["control"]:             # no test information (0 vs 0): the screen posterior decides
            if min(n_scr.values()) < min_scr or scr["arm"] == scr["control"]:
                cap = int(getattr(self.cfg, "ab_max_sessions_mult", 10)) * self.cfg.ab_min_sessions
                why = (f"{n_scr['arm']}/{n_scr['control']} screened nodes (need {min_scr} per arm)"
                       if min(n_scr.values()) < min_scr else "equal screen credit")
                if min(n.values()) < cap:
                    return (f"prompt A/B v{ab['arm']}: not enough information yet -- equal test credit, {why} "
                            f"({n['arm']}/{n['control']} sessions)")
                self.store.set_config_version_state("prompt", ab["arm"], "inconclusive")
                self.store.set_state(K_AB, {})
                self.store.log("meta_ab", arm=ab["arm"], control=ab["control"], credit=cred, screen=scr,
                               screened=n_scr, sessions=n, adopted=False, inconclusive=True)
                return (f"prompt A/B: v{ab['arm']} ended INCONCLUSIVE after {n['arm']}/{n['control']} sessions "
                        f"({why}); control v{ab['control']} stays")
            cred, basis = scr, "screen posterior"
        win = cred["arm"] > cred["control"]
        self.store.set_config_version_state("prompt", ab["arm"], "active" if win else "retired")
        if win:
            self.store.set_config_version_state("prompt", ab["control"], "superseded")
        self.store.set_state(K_AB, {})
        self.store.log("meta_ab", arm=ab["arm"], control=ab["control"], credit=cred, sessions=n, adopted=win,
                       basis=basis, screened=n_scr)
        return (f"prompt A/B: v{ab['arm']} {'ADOPTED' if win else 'retired'} ({basis} credit/GPU-h {cred['arm']:.2e} vs "
                f"{cred['control']:.2e}, {n['arm']}/{n['control']} sessions)")

    def judge_fnf(self, rep: MetaReport) -> str:
        """Fork-and-flush: hand the report's diversity metric to the round trigger (``fnf.K_DIV``) and adopt the
        flushed arm as the explore-lane default (``ideation.fnf_share`` -> ``fnf.ADOPT_SHARE``, a new config version)
        when its A/B verdict is ``adopt``: fresh-seed credit per GPU-h ahead of the shared-context control by >= 2
        bootstrap SE after >= ``fnf.MIN_ROUNDS`` closed rounds. Never by the replay (it cannot see the arm)."""
        from loop.core import fnf as FN
        f = rep.fnf or {}
        if f.get("diversity"):
            self.store.set_state(FN.K_DIV, f["diversity"])
        a = f.get("ab") or {}
        if not a:
            return ""
        cur = self.versions.values()
        if a["verdict"] != "adopt" or cur["ideation.fnf_share"] >= FN.ADOPT_SHARE:
            return (f"fork-and-flush A/B: {a['verdict']} ({a['rounds']} rounds, credit/GPU-h flush "
                    f"{a['credit_per_gpu_h']['flush']:.2e} vs shared {a['credit_per_gpu_h']['shared']:.2e})")
        v = self.versions.accept_config({"ideation.fnf_share": FN.ADOPT_SHARE},
                                         f"fork-and-flush A/B: flushed branches ahead by {a['diff']:.2e} credit/GPU-h "
                                         f"(SE {a['se']}) over {a['rounds']} rounds", {"fnf": a})
        self.store.log("meta_fnf", adopted=True, version=v, ab=a)
        return (f"fork-and-flush A/B: ADOPTED as the explore-lane default (config v{v}: ideation.fnf_share "
                f"{cur['ideation.fnf_share']:g} -> {FN.ADOPT_SHARE:g}; diff {a['diff']:.2e}, SE {a['se']})")

    # -- the cycle ---------------------------------------------------------------------------------------------
    def due(self, now: float) -> bool:
        """Loop wall time since the last report >= every_h (gaps > MAX_GAP_S between ticks are not loop time)."""
        c = self.store.get_state(K_CLOCK) or {"wall_s": 0.0, "last": now}
        c["wall_s"] += min(max(0.0, now - c["last"]), MAX_GAP_S)
        c["last"] = now
        due = c["wall_s"] >= self.cfg.every_h * 3600
        if due:
            c["wall_s"] = 0.0
        self.store.set_state(K_CLOCK, c)
        return due

    def __call__(self, tick: Any, data: Ledger, statuses: Mapping[str, Any]) -> list[str]:
        acts: list[str] = []
        core = getattr(tick, "core", None)
        if getattr(tick, "research", None) is not None and getattr(tick.research, "prompt_dir", 0) is None:
            tick.research.prompt_dir = self.arm_for
        try:
            act = self.versions.active("config")
            if act["v"] != self.applied:
                acts += [f"meta: {w}" for w in apply(self.versions.values(act["v"]), core)]
                self.applied = act["v"]
                acts.append(f"meta: config v{act['v']} applied")
            if self._review is not None and self._review.done():
                fut, self._review = self._review, None
                acts += self.accept(fut.result(), data)
            llm = self.llm or getattr(getattr(getattr(tick, "research", None), "researcher", None), "llm", None)
            if self._pending is not None and self._review is None and not self._llm_blocked() and llm is not None:
                rep, self._pending = self._pending, None          # the review the LLM pause deferred
                self.store.set_state(K_REVIEW_PENDING, None)
                acts += [f"meta: {x}" for x in self._start_review(rep, llm, self.review_pool(tick), wait=False)]
            if self.due(self.clock()):
                acts += self.cycle(data, post=getattr(tick, "post", None), pool=self.review_pool(tick), llm=llm,
                                   bar_led=getattr(tick, "data", None))     # the official ledger, as the hb's best=
        except Exception as exc:  # noqa: BLE001 - the meta-cycle never stops the loop
            self.store.log("meta_error", error=repr(exc)[:500])
            acts.append(f"meta: error {exc!r}"[:200])
        return acts

    def cycle(self, data: Ledger, *, post: Any = None, pool: Any = None, llm: Any = None,
              wait: bool = False, bar_led: Ledger | None = None) -> list[str]:
        """Report (+ file), auto-revert check, A/B judgement, then the review (in ``pool``; ``wait``: inline)."""
        now, last = self.clock(), self.store.get_state(K_LAST) or {}
        values = self.versions.values()
        since = max(float(last.get("ts", 0.0)), data.bases[-1][0] if data.bases else 0.0)   # this base epoch at most
        rep, self.points = report(data, self.store, now=now, since=since, post=post,
                                  values=values, last=last, bar_led=bar_led,
                                  bar=getattr(getattr(self.ctx, "cfg", None), "labless", None))
        ap = post
        try:                                           # v3-c7i: the allocation section, its weights logged each meta
            from loop.core import alloc as AL
            from loop.core.acquire import Posterior
            ap = post if post is not None else Posterior(data, ID.blocks_of(self.store, data))
            rep.alloc = AL.meta_section(self.ctx, self.store, data, ap, since, now)
            self.store.log("alloc_meta", **rep.alloc)
        except Exception as exc:  # noqa: BLE001 - a section that cannot be computed is reported, not fatal
            rep.errors["alloc"] = f"{type(exc).__name__}: {exc}"[:200]
        try:                                           # 10-22: the causal component audit (loop.components.v3_section)
            from loop import components as CG
            from loop.core.acquire import Posterior
            beliefs: dict[str, Any] = {}
            passed: set[str] = set()
            gpu: dict[str, float] = {}
            try:                                       # no fit yet (a fresh ledger): the draws are still counted, outcomes unscored
                ap = ap if ap is not None else Posterior(data, ID.blocks_of(self.store, data))
                beliefs = ap.beliefs()
                passed = {j.node for j in ID.view(data, ID.blocks_of(self.store, data), ap).judged
                          if j.verdict.kind is T.Kind.PASS}
            except Exception:  # noqa: BLE001
                pass
            for r in data.runs:
                gpu[r.node] = gpu.get(r.node, 0.0) + r.gpu_h
            rep.components = CG.v3_section(self.ctx, self.store, self.store_nodes(), gpu, beliefs, passed)
            from loop import root_audit as RA
            if RA.enabled(self.ctx):                   # 11-08: what each root-audit arm became (the ledger's audit)
                rep.components["root_audit_arms"] = RA.outcomes(self.store)
        except Exception as exc:  # noqa: BLE001 - a section that cannot be computed is reported, not fatal
            rep.errors["components"] = f"{type(exc).__name__}: {exc}"[:200]
        path = self.write(rep)
        try:                                           # 10-22: the 3-hour milestone digest (state/digests/), fed from the report
            from loop import digest as DG
            DG.write_report(Path(self.versions.dir), self.store, asdict(rep))
        except Exception as exc:  # noqa: BLE001 - a digest is not worth a meta cycle
            self.store.log("digest_error", error=repr(exc)[:300])
        self.store.set_state(K_LAST, {"ts": now, "noise": rep.noise, "guards": rep.guards(), "path": str(path)})
        lines = [f"meta report {path}: " + (f"guards out of bounds: {'; '.join(rep.out_of_bounds())}"
                                            if rep.out_of_bounds() else "guards in bounds")]
        if rep.bar:
            lines.append(f"{rep.bar['text']}; best {rep.bar['best'] or 'none'}; pause-rule improvement: "
                         + ("; ".join(rep.switches or []) or "none") + "; beats the Labless bar: "
                         + ("; ".join(rep.bar["beats"]) or "none"))
        lines += self.check_revert(rep)
        try:
            from loop.core.acquire import Posterior
            post = post if post is not None else Posterior(data, ID.blocks_of(self.store, data))
            if msg := self.judge_ab(data, ID.view(data, ID.blocks_of(self.store, data), post).judged, post):
                lines.append(msg)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"prompt A/B not judged: {exc!r}"[:160])
        try:
            if msg := self.judge_fnf(rep):
                lines.append(msg)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"fork-and-flush A/B not judged: {exc!r}"[:160])
        if self.cfg.review and llm is not None and not self.dry:
            if (why := self._llm_blocked()) and not wait:      # the LLM pause / outage holds the review too
                self._pending = rep
                self.store.set_state(K_REVIEW_PENDING, {"ts": now, "report": str(path)})
                lines.append(f"meta-review deferred: LLM {why} (starts once the LLM is back)")
            elif self._review is not None and not wait:
                lines.append("meta-review skipped: the previous one is still running")
            else:
                self._pending = None            # this report supersedes a deferred one (memory AND the stored record)
                self.store.set_state(K_REVIEW_PENDING, None)
                lines += self._start_review(rep, llm, pool, wait=wait, data=data)
        self.summary(lines)
        return [f"meta: {x}" for x in lines]

    def store_nodes(self) -> dict[str, Any]:
        return {n.id: n for n in self.store.nodes() if n.id != ROOT}

    def _start_review(self, rep: MetaReport, llm: Any, pool: Any, *, wait: bool, data: Ledger | None = None
                      ) -> list[str]:
        kw = {"system": "You tune an automated ML experiment loop. You change only its tunable knobs and prompts; "
                        "cite only numbers from the input.", "prompt": self.review_prompt(rep), "schema": REVIEW_SCHEMA}
        if wait or pool is None:
            return self.accept(llm.complete_json("review", **kw), data)
        self._review = pool.submit(llm.complete_json, "review", **kw)
        return ["meta-review started"]

    def summary(self, lines: Sequence[str]) -> None:
        self.store.log("meta_cycle", summary=" | ".join(lines)[:2000])

    def write(self, rep: MetaReport) -> Path:
        d = Path(self.versions.dir) / "meta"
        d.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(rep.ts))
        (d / f"{stem}.json").write_text(json.dumps(asdict(rep), sort_keys=True, default=str))
        path = d / f"{stem}.md"
        path.write_text(rep.markdown())
        return path

    def check_revert(self, rep: MetaReport) -> list[str]:
        """The first report after a config version was accepted: a guard it moved out of bounds reverts it."""
        chk = self.store.get_state(K_CHECK) or {}
        if not chk:
            return []
        self.store.set_state(K_CHECK, {})
        before, now = chk.get("guards", {}), rep.guards()
        gb = guard_bounds()
        broke = [k for k, v in now.items() if not gb[k][0] <= v <= gb[k][1]
                 and (k not in before or gb[k][0] <= before[k] <= gb[k][1])]
        act = self.versions.active("config")
        if not broke or act["v"] != chk.get("v") or act["parent"] is None:
            return [f"config v{chk.get('v')} kept: guards in bounds"] if not broke else []
        return ["AUTO-REVERT " + self.versions.revert("config", act["parent"], f"guard out of bounds: {', '.join(broke)}")]

    def not_backtestable(self) -> dict[str, Any]:
        """Config keys the replay cannot evaluate (v3-c6a: a backtest with replay gain 0 and SE 0 -- the policy the
        replay scores never reads the key, e.g. ideation.fnf_every_gpu_h): neither listed for the review nor
        re-backtested when proposed. A mark lapses once the eval points have doubled (the replay may see it then)."""
        marks = dict(self.store.get_state(K_NOBT) or {})
        n = len(self.points or ())
        return {k: m for k, m in marks.items() if not n or n < 2 * max(1, int(m.get("n") or 0))}

    def review_prompt(self, rep: MetaReport) -> str:
        cur = self.versions.values()
        nobt = self.not_backtestable()
        tun = "\n".join(f"- {k} = {cur[k]:g} (bounds [{t.lo:g}, {t.hi:g}]): {t.doc}" for k, t in SP.visible_tunables().items()
                        if k not in nobt)
        if nobt:
            tun += ("\n(not backtestable -- the replay cannot evaluate them, do not propose: "
                    + ", ".join(sorted(nobt)) + ")")
        pv = self.versions.active("prompt")["v"]
        prompts = "\n\n".join(f"### {n} (v{pv})\n```\n{self.versions.prompt_text(pv, n)[:5000]}\n```" for n in SP.META_PROMPTS)
        hist = [f"- {e['kind']} {e['path']}: {e['verdict']} ({str(e.get('reason', ''))[:160]})"
                for e in self.store.events("meta_proposal")[-12:]]
        ab = self.store.get_state(K_AB) or {}
        text = _prompt("meta_review.md").read_text()
        values = {"report": rep.markdown(llm=True), "tunables": tun, "prompts": prompts, "history": "\n".join(hist) or "(none)",
                  "ab": f"prompt A/B running: v{ab['arm']} vs v{ab['control']}" if ab.get("arm") else "no prompt A/B running",
                  "locked": ", ".join(SP.LOCKED_FILES)}
        return re.sub(r"\{([a-z_]+)\}", lambda m: str(values.get(m.group(1), m.group(0))), text)

    def accept(self, answer: Mapping[str, Any], data: Ledger) -> list[str]:
        """Vet every proposal; at most ONE config change (the best that passes its backtest) and ONE prompt arm."""
        surface = tuple(getattr(getattr(getattr(self.ctx, "spec", None), "surface", None), "locked", ()) or ())
        cur, out, cands = self.versions.values(), [], []
        ab_busy = bool((self.store.get_state(K_AB) or {}).get("arm"))
        for p in proposals_of(answer):
            why = vet(p, surface)
            if not why and p.kind == "prompt" and ab_busy:
                why = "a prompt A/B is already running"
            if not why and p.kind == "config":
                key = re.sub(r"^(state/)?loop_config/(v\d+\.yaml[:#.]?)?", "", _norm(p.path))
                if cur[key] == p.value:
                    why = f"{key} is already {p.value}"
                elif key in self.not_backtestable():
                    why = f"{key} is not backtestable (the replay cannot evaluate it)"
                else:
                    if not self.points:
                        self.points = eval_points(data, B.replay(data))
                    bt = backtest_config(self.points, cur, {**cur, key: p.value}, split_h=self.cfg.split_h)
                    cands.append((bt["new"]["value"] - bt["cur"]["value"] if "new" in bt else -1, key, p, bt))
                    why = "" if bt["accept"] else f"backtest: {bt['reason']}"
                    if "new" in bt and bt["new"] == bt["cur"] and not any((bt.get("se") or {}).values()):
                        marks = dict(self.store.get_state(K_NOBT) or {})   # gain 0, SE 0: the replay is blind to it
                        marks[key] = {"ts": self.clock(), "n": len(self.points), "reason": bt["reason"][:200]}
                        self.store.set_state(K_NOBT, marks)
                        why = f"{key} is not backtestable (replay value unchanged, SE 0): marked, not proposed again"
            if why:
                self.store.log("meta_proposal", kind=p.kind, path=p.path, value=p.value, verdict="refused", reason=why,
                               rationale=p.rationale)
                out.append(f"refused {p.kind} {p.path}: {why}")
            elif p.kind == "prompt":
                v = self.versions.new_prompt(p, p.rationale)
                if isinstance(v, str):
                    out.append(f"refused prompt {p.path}: {v}")
                    self.store.log("meta_proposal", kind=p.kind, path=p.path, verdict="refused", reason=v)
                    continue
                ctl = self.versions.active("prompt")["v"]
                self.store.set_state(K_AB, {"arm": v, "control": ctl, "share": self.cfg.ab_share, "started": self.clock(),
                                            "sessions": {}})
                self.store.log("meta_proposal", kind=p.kind, path=p.path, verdict="ab_arm", version=v, rationale=p.rationale)
                out.append(f"prompt v{v} ({p.path}) is an A/B arm on {self.cfg.ab_share:.0%} of sessions vs v{ctl}")
                ab_busy = True
        best = max((c for c in cands if c[3]["accept"]), key=lambda c: c[0], default=None)
        for gain, key, p, bt in cands:
            if not bt["accept"]:
                continue
            if (gain, key) != best[:2]:
                self.store.log("meta_proposal", kind="config", path=p.path, value=p.value, verdict="refused",
                               reason="another config change was accepted this cycle")
                out.append(f"refused config {key}={p.value}: another change accepted this cycle")
                continue
            v = self.versions.accept_config({key: p.value}, p.rationale, bt)
            self.store.set_state(K_CHECK, {"v": v, "guards": (self.store.get_state(K_LAST) or {}).get("guards", {})})
            self.store.log("meta_proposal", kind="config", path=p.path, value=p.value, verdict="accepted", version=v,
                           reason=bt["reason"])
            out.append(f"ACCEPTED config v{v}: {key} {cur[key]:g} -> {p.value:g} ({bt['reason']})")
        if diag := str(answer.get("diagnosis", "")).strip():
            out.insert(0, f"review: {diag[:300]}")
        self.summary(out or ["review: no proposals"])
        return out or ["review: no proposals"]
