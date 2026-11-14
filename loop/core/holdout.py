"""loop.core.holdout — the BLIND HOLDOUT (v3-c6a; DAPS arXiv 2609.00077 / REUSE 2609.33180: a loop that selects on
one reused benchmark overfits it).

A few probe TASKS are held out of every number the loop SELECTS on, and used only to audit what it selected:

- ``visible``  the ledger the selection side reads (posterior, acquisition, knob banks / waves, program ladders, briefs,
               levers): every observation's per-axis value recomputed from ``summary.json``'s per-task results WITHOUT the
               held tasks (an axis is the mean over its tasks, exactly as ``probe.py`` aggregates). The official
               ``panel`` of each observation is left as it is: the TEST verdict, the Labless bar and the target gap read
               the full locked panel. ``probe.py`` is never touched; the numbers come from the run's ``summary.json``.
- ``audit``    each passed fresh-seed TEST (a closed block's frozen PASS verdict): the candidate's same-seed paired gain
               over its control on the block's seeds, VISIBLE (the held axes' visible tasks) vs HOLDOUT (the held tasks),
               both as official-weight averages over the held axes (apples to apples), plus the visible panel gain.
               The meta report's "holdout audit" section shows them, the gap ratio over time, and an ALERT when the mean
               visible gain exceeds ``holdout.alert_ratio`` x the mean holdout gain over >= ``holdout.min_nodes`` nodes.

Tasks (per ``summary.json``, ``final_probe_`` keys): cls ``<t>_{linear,knn,fewshot}_val_f1`` (a task = the mean of its
three heads: ``classification_mean_f1`` is the mean over tasks x heads), seg ``<t>_seg_val_f1``, surv ``<t>_val_cindex``,
rob ``pathorob_<t>_robustness`` (``robustness_mean`` = the pathorob mean over its two subsets). slide and auc have one task
each: never held out, their visible value is the official one.

The holdout set (``holdout.tasks``, "axis:task"; () = the deterministic pick, :func:`default_tasks`): per axis with >= 2
tasks, the task minimising ``sha1("holdout:<axis>:<task>")`` -- on the 10-13 suite cls:spider_colorectal, seg:pannuke,
surv:leopard_bcr, rob:tolkach_esca. The pick is made ONCE (on the first task set seen) and frozen in state
``holdout.tasks``, so a new probe task never reshuffles it.

Known limits (v3-c6r review H2): rob:tolkach_esca is the robustness view of the ESCA data, so it correlates with the
VISIBLE cls:esca task (the rob holdout is not independent of a visible task); the VISIBLE rob axis is camelyon alone (a
one-task axis: noisier than the official two-task mean). The eval-only harm guard reads the FULL panel (a guard, not
an ideator-facing number). Leak closure: the meta report / A/B replay the LLM reviewer reads, the brief's target gap
and the calibration reference gap use the VISIBLE composite (:func:`visible_panel`); the audit's numbers are
OPERATOR-ONLY (``report_text`` goes to the operator log / dashboard, never into an LLM prompt)."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from loop.core.data import OFFICIAL_WEIGHTS, TEST, Ledger, Obs

HEADS = ("linear", "knn", "fewshot")
#: axis -> (the summary key naming a task, the task's score from the summary)
PATTERNS: dict[str, tuple[re.Pattern[str], Any]] = {
    "cls": (re.compile(r"final_probe_(\w+)_fewshot_val_f1"),
            lambda d, t: sum(float(d[f"final_probe_{t}_{h}_val_f1"]) for h in HEADS) / len(HEADS)),
    "seg": (re.compile(r"final_probe_(\w+)_seg_val_f1"), lambda d, t: float(d[f"final_probe_{t}_seg_val_f1"])),
    "surv": (re.compile(r"final_probe_(\w+)_val_cindex"), lambda d, t: float(d[f"final_probe_{t}_val_cindex"])),
    "rob": (re.compile(r"final_probe_pathorob_(\w+)_robustness"),
            lambda d, t: float(d[f"final_probe_pathorob_{t}_robustness"])),
}
K_TASKS = "holdout.tasks"        # the frozen pick (list of "axis:task")
K_AUDIT = "holdout.audit"        # {"rows": [...], "summary": {...}, "alert": bool}
CACHE = "holdout/tasks.jsonl"    # under the state dir: {"run": id, "t": {"axis:task": score} | null}


def tasks_of(summary: Mapping[str, Any]) -> dict[str, float]:
    """"axis:task" -> score for every per-task result of one run's ``summary.json`` ({} for none)."""
    out: dict[str, float] = {}
    for axis, (pat, score) in PATTERNS.items():
        for k in summary:
            if (m := pat.fullmatch(k)) is not None:
                try:
                    out[f"{axis}:{m.group(1)}"] = float(score(summary, m.group(1)))
                except (KeyError, TypeError, ValueError):
                    continue
    return out


def default_tasks(universe: Iterable[str]) -> tuple[str, ...]:
    """The deterministic holdout: per axis with >= 2 tasks in ``universe`` ("axis:task"), the one minimising
    sha1("holdout:<axis>:<task>")."""
    by: dict[str, list[str]] = {}
    for x in universe:
        a, _, t = x.partition(":")
        by.setdefault(a, []).append(t)
    return tuple(sorted(f"{a}:{min(ts, key=lambda t: hashlib.sha1(f'holdout:{a}:{t}'.encode()).hexdigest())}"
                        for a, ts in by.items() if len(set(ts)) >= 2))


def split(tasks: Mapping[str, float], held: frozenset[str]) -> tuple[dict[str, float], dict[str, float]]:
    """(visible axis means, holdout axis means) of one run's per-task scores: axes with no task here are absent."""
    vis: dict[str, list[float]] = {}
    ho: dict[str, list[float]] = {}
    for k, v in tasks.items():
        (ho if k in held else vis).setdefault(k.partition(":")[0], []).append(v)
    return ({a: statistics.fmean(xs) for a, xs in vis.items()}, {a: statistics.fmean(xs) for a, xs in ho.items()})


class TaskCache:
    """run id -> its per-task scores, read once from ``<output_dir>/summary.json`` (outputs are immutable once done)
    and appended to a JSONL file under the state dir; an unreadable summary is recorded as None (retried once per
    process)."""

    def __init__(self, path: str | Path | None, workers: int = 16) -> None:
        self.path = Path(path) if path else None
        self.workers = max(1, int(workers))
        self.t: dict[str, dict[str, float] | None] = {}
        self._retried: set[str] = set()
        self._lock = threading.Lock()
        if self.path is not None and self.path.is_file():
            for line in self.path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and "run" in row:
                    self.t[str(row["run"])] = row.get("t")

    def get(self, run_id: str) -> dict[str, float] | None:
        return self.t.get(run_id)

    def update(self, led: Ledger) -> int:
        """Read the summaries of ``led``'s observations not cached yet (threaded); returns how many were read."""
        out_dir = {r.id: r.output_dir for r in led.runs}
        want = [o.run_id for o in led.obs if o.run_id not in self.t
                or (self.t[o.run_id] is None and o.run_id not in self._retried)]
        want = list(dict.fromkeys(want))
        if not want:
            return 0

        def one(rid: str) -> tuple[str, dict[str, float] | None]:
            d = out_dir.get(rid) or ""
            try:
                got = tasks_of(json.loads((Path(d) / "summary.json").read_text())) if d else {}
            except (OSError, ValueError):
                got = {}
            return rid, (got or None)
        if len(want) > 1 and self.workers > 1:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(want)), thread_name_prefix="holdout") as ex:
                rows = list(ex.map(one, want))
        else:
            rows = [one(r) for r in want]
        with self._lock:
            for rid, t in rows:
                if rid in self.t:
                    self._retried.add(rid)
                self.t[rid] = t
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    for rid, t in rows:
                        f.write(json.dumps({"run": rid, "t": t}, sort_keys=True) + "\n")
        return len(rows)


def visible(led: Ledger, cache: TaskCache, held: frozenset[str]) -> tuple[Ledger, int]:
    """(``led`` with every observation's axes recomputed without the ``held`` tasks, observations without per-task
    data left as they are). The official ``panel`` is unchanged."""
    if not held:
        return led, 0
    axes_held = {h.partition(":")[0] for h in held}
    out, missing = [], 0
    for o in led.obs:
        t = cache.get(o.run_id)
        if not t:
            missing += 1
            out.append(o)
            continue
        vis, _ = split(t, held)
        ax = dict(o.axes)
        for a in axes_held:
            if a in vis and a in ax:
                ax[a] = vis[a]
        out.append(replace(o, axes=ax))
    return replace(led, obs=tuple(out)), missing


def visible_panel(led: Ledger) -> Ledger:
    """``led`` (a VISIBLE ledger) with each observation's ``panel`` replaced by the official-weight composite of its
    (visible) axes -- the number the LLM-facing texts may show (v3-c6r H2: the meta report, the brief's gap); an
    observation missing an axis keeps its panel. Cached for the last ledger object."""
    got = _VP.get("v")
    if got is not None and got[0] is led:
        return got[1]
    out = []
    for o in led.obs:
        if all(a in o.axes for a in OFFICIAL_WEIGHTS):
            o = replace(o, panel=sum(w * float(o.axes[a]) for a, w in OFFICIAL_WEIGHTS.items()))
        out.append(o)
    vl = replace(led, obs=tuple(out))
    _VP["v"] = (led, vl)
    return vl


def panel_offset(vis: Ledger) -> float:
    """The current epoch's mean (visible-composite panel - official panel) over a VISIBLE ledger's observations (its
    ``panel`` is still the official one): the population offset between the two scales -- a target stated on the
    official panel moves by it; no node's own holdout result is in it beyond its share of the mean."""
    full = {o.run_id: o.panel for o in vis.in_epoch().obs}
    xs = [o.panel - full[o.run_id] for o in visible_panel(vis).in_epoch().obs if o.run_id in full]
    return statistics.fmean(xs) if xs else 0.0


_VP: dict[str, Any] = {}


# -- the audit ----------------------------------------------------------------------------------------------------
def _seed_means(obs: Sequence[Obs], cache: TaskCache, held: frozenset[str]) -> dict[int, dict[str, float]]:
    """seed -> {visible axes "v:<a>", holdout axes "h:<a>", the visible panel "vpanel"} (runs at a seed averaged)."""
    acc: dict[int, dict[str, list[float]]] = {}
    for o in obs:
        t = cache.get(o.run_id)
        if not t:
            continue
        vis, ho = split(t, held)
        row = acc.setdefault(o.seed, {})
        full = dict(o.axes)
        full.update({a: v for a, v in vis.items() if a in full})
        row.setdefault("vpanel", []).append(sum(w * full[a] for a, w in OFFICIAL_WEIGHTS.items() if a in full))
        for a, v in vis.items():
            row.setdefault(f"v:{a}", []).append(v)
        for a, v in ho.items():
            row.setdefault(f"h:{a}", []).append(v)
    return {s: {k: statistics.fmean(xs) for k, xs in r.items()} for s, r in acc.items()}


def gains(led: Ledger, c: str, i: str, seeds: Iterable[int], cache: TaskCache, held: frozenset[str]
          ) -> dict[str, float] | None:
    """The same-seed paired gains of ``c`` over ``i`` on ``seeds``: visible / holdout as official-weight averages over
    the held axes, and the visible panel; None without a pair."""
    want = set(seeds)
    by_c = _seed_means([o for o in led.obs if o.node == c and o.seed in want], cache, held)
    by_i = _seed_means([o for o in led.obs if o.node == i and o.seed in want], cache, held)
    both = sorted(set(by_c) & set(by_i))
    axes = sorted({h.partition(":")[0] for h in held})
    w = {a: OFFICIAL_WEIGHTS.get(a, 0.0) for a in axes}
    if not both or not sum(w.values()):
        return None

    def avg(prefix: str, s: int) -> float | None:
        xs = [(w[a], by_c[s][f"{prefix}:{a}"] - by_i[s][f"{prefix}:{a}"]) for a in axes
              if f"{prefix}:{a}" in by_c[s] and f"{prefix}:{a}" in by_i[s]]
        tot = sum(x for x, _ in xs)
        return sum(x * d for x, d in xs) / tot if tot else None
    vis = [x for s in both if (x := avg("v", s)) is not None]
    ho = [x for s in both if (x := avg("h", s)) is not None]
    if not vis or not ho:
        return None
    return {"n": len(both), "visible": statistics.fmean(vis), "holdout": statistics.fmean(ho),
            "visible_panel": statistics.fmean(by_c[s]["vpanel"] - by_i[s]["vpanel"] for s in both)}


def select_seeds(led: Ledger, c: str, i: str) -> list[int]:
    """The seeds both ``c`` and its control ``i`` ran on OUTSIDE the TEST pool: the selection seeds, where the loop
    chose ``c`` on the visible tasks only (the TEST verdict reads the full panel, holdout tasks included)."""
    def sel(n: str) -> set[int]:
        return {o.seed for o in led.obs if o.node == n and o.pool != TEST}
    return sorted(sel(c) & sel(i))


def audit(led: Ledger, blocks: Sequence[Any], cache: TaskCache, held: frozenset[str]) -> list[dict[str, Any]]:
    """One row per passed fresh-seed TEST (a closed TEST block's frozen PASS verdict), oldest first.

    v3-c6r (review H1): the row's visible / holdout gains are measured on the SELECTION seeds (``basis`` "select"):
    selection there read only the visible tasks, so a visible >> holdout gap is the overfit signal. On the TEST seeds
    the verdict read the full panel (holdout tasks included): its numbers are selection-biased toward the holdout too
    and are kept only as ``test_*`` (flagged), never in the ALERT. A node without a select-seed pair keeps the TEST
    numbers with ``basis`` "test" (shown, excluded from the summary)."""
    rows = []
    for b in blocks:
        if getattr(b, "pool", "") != TEST or getattr(b, "closed_ts", None) is None or not b.verdicts:
            continue
        for v in b.verdicts:
            if str(v.get("kind")) != "pass":
                continue
            c, i = str(v["node"]), str(v["vs"])
            gt = gains(led, c, i, b.seeds, cache, held)
            gs = gains(led, c, i, select_seeds(led, c, i), cache, held)
            g = gs if gs is not None else gt
            if g is None:
                continue
            row = {"node": v["node"], "vs": v["vs"], "block": b.id, "ts": float(b.closed_ts),
                   "official": float(v.get("mean") or 0.0), "basis": "select" if gs is not None else "test",
                   **{k: round(x, 6) if isinstance(x, float) else x for k, x in g.items()}}
            if gt is not None:
                row.update(test_visible=round(gt["visible"], 6), test_holdout=round(gt["holdout"], 6),
                           test_n=gt["n"], test_biased=True)
            rows.append(row)
    return sorted(rows, key=lambda r: (r["ts"], r["node"]))


def summarize(rows: Sequence[Mapping[str, Any]], alert_ratio: float, min_nodes: int) -> dict[str, Any]:
    """n, the mean visible / holdout gains, their ratio, the cumulative ratio after each pass (over time), and the
    ALERT (>= min_nodes passes and the mean visible gain > alert_ratio x the mean holdout gain)."""
    def ratio(v: float, h: float) -> float:
        return v / h if h > 0 else (math.inf if v > 0 else 0.0)
    series, vs, hs = [], [], []
    rows = [r for r in rows if r.get("basis", "select") == "select"]   # TEST-seed numbers: biased, never alerted on
    for r in rows:
        vs.append(float(r["visible"]))
        hs.append(float(r["holdout"]))
        series.append({"ts": r["ts"], "node": r["node"], "ratio": round(ratio(statistics.fmean(vs),
                                                                              statistics.fmean(hs)), 3)})
    if not rows:
        return {"n": 0, "visible": 0.0, "holdout": 0.0, "ratio": 0.0, "alert": False, "series": []}
    mv, mh = statistics.fmean(vs), statistics.fmean(hs)
    return {"n": len(rows), "visible": round(mv, 6), "holdout": round(mh, 6), "ratio": round(ratio(mv, mh), 3),
            "alert": len(rows) >= int(min_nodes) and mv > 0 and mv > float(alert_ratio) * mh, "series": series,
            "alert_ratio": float(alert_ratio), "min_nodes": int(min_nodes)}


def report_text(st: Mapping[str, Any]) -> list[str]:
    """The OPERATOR's "holdout audit" lines from state ``holdout.audit`` (never put into an LLM-facing text: the
    holdout numbers would leak the held tasks into selection)."""
    s, rows = st.get("summary") or {}, st.get("rows") or []
    out = [f"held-out tasks: {', '.join(st.get('tasks') or []) or 'none'} (official panel unchanged); "
           f"visible-composite noise: {st.get('noise') or 'not measured'}"]
    if not s.get("n"):
        return out + ["no passed fresh-seed test with per-task data yet"]
    out.append(f"{s['n']} passed tests: mean visible gain {s['visible']:+.5f} vs holdout {s['holdout']:+.5f} "
               f"(ratio {s['ratio']:.2f}; alert above {s.get('alert_ratio', 2.0):g} over >= {s.get('min_nodes', 5)})")
    out += [f"- {r['node']} vs {r['vs']} (n={r['n']}, {r.get('basis', 'select')} seeds): visible {r['visible']:+.5f}, "
            f"holdout {r['holdout']:+.5f}, visible panel {r['visible_panel']:+.5f}, official {r['official']:+.5f}"
            + (f"; TEST seeds (biased: the verdict read the holdout) visible {r['test_visible']:+.5f} holdout "
               f"{r['test_holdout']:+.5f}" if "test_visible" in r else "") for r in rows[-8:]]
    out.append("gap ratio over time: " + ", ".join(f"{x['node']} {x['ratio']:.2f}" for x in s.get("series", [])[-12:]))
    if s.get("alert"):
        out.append(f"ALERT: visible gains exceed {s.get('alert_ratio', 2.0):g}x the holdout gains -- the loop is "
                   "overfitting the visible tasks")
    return out


# -- the tick's side -----------------------------------------------------------------------------------------------
class Holdout:
    """The tick's holdout: the task cache, the frozen held set, the visible ledger and the audit."""

    def __init__(self, ctx: Any, store: Any) -> None:
        self.ctx, self.store = ctx, store
        state = getattr(ctx, "state_dir", "") or ""
        self.cache = TaskCache(Path(state) / CACHE if state else None, int(getattr(self.cfg, "cache_workers", 16)))
        self.missing = 0

    @property
    def cfg(self) -> Any:
        return getattr(getattr(self.ctx, "cfg", None), "holdout", None)

    @property
    def on(self) -> bool:
        return bool(getattr(self.cfg, "enabled", False))

    def held(self) -> frozenset[str]:
        """The configured tasks, else the frozen pick (made once from the cache's task universe)."""
        if tuple(getattr(self.cfg, "tasks", ()) or ()):
            return frozenset(self.cfg.tasks)
        got = self.store.get_state(K_TASKS)
        if got:
            return frozenset(got)
        sets: dict[tuple[str, ...], int] = {}
        for t in self.cache.t.values():
            if t:
                sets[tuple(sorted(t))] = sets.get(tuple(sorted(t)), 0) + 1
        if not sets:
            return frozenset()
        pick = default_tasks(max(sets, key=lambda k: (sets[k], k)))
        self.store.set_state(K_TASKS, list(pick))
        self.store.log("holdout_tasks", tasks=list(pick), rule="sha1 min per axis with >= 2 tasks")
        return frozenset(pick)

    def visible(self, led: Ledger) -> Ledger:
        if not self.on:
            return led
        self.cache.update(led)
        vis, self.missing = visible(led, self.cache, self.held())
        return vis

    def audit_tick(self, led: Ledger, blocks: Sequence[Any], acts: list[str], vis: Ledger | None = None,
                   now: float | None = None) -> dict[str, Any]:
        """Re-derive the audit (cheap: passed tests only); log new rows and an alert transition; keep it in state
        (with the visible-vs-full composite noise note, refitted every ``holdout.noise_every_h``)."""
        if not self.on:
            return {}
        held = self.held()
        rows = audit(led, blocks, self.cache, held)
        s = summarize(rows, float(self.cfg.alert_ratio), int(self.cfg.min_nodes))
        old = self.store.get_state(K_AUDIT) or {}
        seen = {(r.get("node"), r.get("block")) for r in old.get("rows") or ()}
        for r in rows:
            if (r["node"], r["block"]) not in seen:
                self.store.log("holdout_audit", **r)
                acts.append(f"holdout audit {r['node']}: visible {r['visible']:+.5f} vs holdout {r['holdout']:+.5f} "
                            f"(n={r['n']})")
        if s["alert"] and not (old.get("summary") or {}).get("alert"):
            self.store.log("holdout_alert", ratio=s["ratio"], n=s["n"], visible=s["visible"], holdout=s["holdout"])
            acts.append(f"ALERT holdout: visible gains {s['ratio']:.2f}x the holdout gains over {s['n']} passed tests")
        noise, noise_ts = old.get("noise", ""), float(old.get("noise_ts") or 0.0)
        if vis is not None and now is not None and now - noise_ts >= 3600.0 * float(getattr(self.cfg, "noise_every_h", 3.0)):
            noise, noise_ts = noise_note(led, vis) or noise, now
        st = {"tasks": sorted(held), "rows": rows, "summary": s, "missing": self.missing, "noise": noise,
              "noise_ts": noise_ts}
        if st != old:
            self.store.set_state(K_AUDIT, st)
        return st


def noise_note(led: Ledger, vis: Ledger) -> str:
    """'composite sd .00xx (full .00yy), sd_d ... ' of the current epoch: how much noisier the visible composite is."""
    from loop.core.backtest import fit_at
    from loop.core.model import COMPOSITE
    try:
        a = fit_at(led.in_epoch(), led.change_parents())[0]
        b = fit_at(vis.in_epoch(), vis.change_parents())[0]
    except (ValueError, ZeroDivisionError, KeyError):
        return ""
    return (f"composite sd {b.sd(COMPOSITE):.5f} (full {a.sd(COMPOSITE):.5f}), paired sd_d {b.sd_d(COMPOSITE):.5f} "
            f"(full {a.sd_d(COMPOSITE):.5f}), rho {b.rho[COMPOSITE]:.3f} (full {a.rho[COMPOSITE]:.3f})")
