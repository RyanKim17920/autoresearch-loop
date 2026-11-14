"""loop.design — DESIGN mode (docs/loop/DESIGN.md §4.6). A two-level resolution-IV fractional factorial
over K (4..10) config knobs: 16 or 32 config-only points (family ``design``) screened on ONE shared seed
block against root, analysed by OLS with a per-seed intercept, then ONE stacked node (every significantly
better level) that takes the normal promote/confirm/audit path. Pure python, no new dependencies."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from loop import components as CG
from loop import novelty as NV
from loop.adapters import flatten
from loop.agents.researcher import config_file, diff_hash, set_yaml_value
from loop.contracts import (CLOSED_STATUSES, DESIGN_FAMILY, PANEL, ROOT_NODE_ID, AxisPrediction, Candidate,
                            HandoffNote, HypothesisKind, HypothesisRecord, LeverTags, Node, NodeStatus,
                            Operator, RunStatus, SmokeResult, Stage, Strategy, VoidType, is_oom,
                            next_mitigation, with_oom_note, design_point)
from loop.select import intake
from loop.spec import LoopContext
from loop.world.predictive import invert

K_DESIGN = "design"   # {"current": batch | None, "batches": [results], "request": knobs | None, "idle_at": evals,
                      #  "bad_levels": [{key, level, batch, why}] (a level that crashed: never designed again)}
#: runs -> K -> generator words over the base factors A.. (standard minimum-aberration resolution >= IV
#: generators); K = log2(runs) is the full factorial. An even word is negated so row 0 is all-low.
GENERATORS: dict[int, dict[int, tuple[str, ...]]] = {
    16: {4: (), 5: ("ABCD",), 6: ("ABC", "BCD"), 7: ("ABC", "BCD", "ACD"), 8: ("BCD", "ACD", "ABC", "ABD")},
    32: {5: (), 6: ("ABCDE",), 7: ("ABCD", "ABDE"), 8: ("ABC", "ABD", "BCDE"),
         9: ("ABCD", "ABCE", "ABDE", "ACDE"), 10: ("ABCD", "ABCE", "ABDE", "ACDE", "BCDE")}}


def n_runs(k: int, want: int = 16) -> int:
    return 32 if k > 8 or (want == 32 and k >= 5) else 16


def matrix(k: int, runs: int) -> list[tuple[int, ...]]:
    """The design's rows (+1 = high); row 0 has every knob low."""
    if k not in GENERATORS[runs]:
        raise ValueError(f"no {runs}-run resolution-IV design for {k} knobs")
    base = runs.bit_length() - 1
    rows = []
    for r in range(runs):
        b = {chr(65 + j): 1 if r >> (base - 1 - j) & 1 else -1 for j in range(base)}
        rows.append(tuple(b.values()) + tuple(math.prod(b[c] for c in w) * (-1) ** (len(w) % 2 == 0)
                                              for w in GENERATORS[runs][k]))
    return rows


def _basis(cols: Sequence[Sequence[float]], tol: float = 1e-9) -> list[int]:
    """Indices of a maximal linearly independent subset of ``cols``, taken in order (modified
    Gram-Schmidt): a column whose residual on the kept ones is ~0 lies in their span."""
    basis: list[list[float]] = []
    kept: list[int] = []
    for j, c in enumerate(cols):
        v, n0 = [float(x) for x in c], sum(float(x) * float(x) for x in c)
        for q in basis:
            d = sum(a * b for a, b in zip(v, q))
            v = [a - d * b for a, b in zip(v, q)]
        n = sum(x * x for x in v)
        if n0 > 0 and n > tol * n0:
            basis.append([x / math.sqrt(n) for x in v])
            kept.append(j)
    return kept


def _why_not(keys: Sequence[str], cols: Sequence[Sequence[float]], j: int, kept: Sequence[int]) -> str:
    if not cols[j]:
        return "no scored point"
    if len(set(cols[j])) == 1:
        return f"constant: every scored point ran its {'high' if cols[j][0] > 0 else 'low'} level"
    for i in kept:
        if list(cols[i]) == list(cols[j]) or [-x for x in cols[i]] == list(cols[j]):
            return f"aliased with {keys[i]} over the scored points"
    return "confounded with the seed intercepts / other knobs over the scored points"


def analyze(keys: Sequence[str], rows: Sequence[Sequence[int]], obs: Sequence[tuple[int, int, float]],
            z: float) -> dict[str, Any]:
    """OLS of the panel score on a per-seed intercept + the K main-effect columns. Every point at a seed
    shares that seed's control run, so the per-seed intercept IS the per-control correction (exact in a
    balanced design); root's own run is the all-low point. effect = high - low = 2 beta. Each 2FI alias
    chain's contrast on the residuals is flagged when |c| >= z se (res IV: chains alias 2FIs only).

    Never raises on a degenerate design: only the scored rows count (a crashed / voided / missing point
    is simply absent), and a knob whose column is not identifiable from them (constant — e.g. every
    point at its high level crashed —, aliased with another, a duplicate key, or too few rows) is
    reported ``estimable: False`` with a ``reason`` while the identifiable knobs are still estimated."""
    seeds, k = sorted({s for _, s, _ in obs}), len(keys)
    y = [v for *_, v in obs]
    scols = [[float(s == t) for _, s, _ in obs] for t in seeds]
    kcols = [[float(rows[i][j]) for i, _, _ in obs] for j in range(k)]
    dup = {j for j in range(k) if keys[j] in keys[:j]}
    kept = [j - len(seeds) for j in _basis(scols + [c if j not in dup else [0.0] * len(c)
                                                    for j, c in enumerate(kcols)]) if j >= len(seeds)]
    p = len(seeds) + len(kept)
    df = len(y) - p
    reasons = {j: "duplicate key" if j in dup else _why_not(keys, kcols, j, kept) for j in range(k) if j not in kept}
    if df < 1 or not kept:
        why = f"{len(y)} scored observations for {p} parameters" if df < 1 else "no identifiable knob"
        reasons.update({j: reasons.get(j, why) for j in range(k)})
        kept = []
    effects: list[dict[str, Any]] = [{"key": keys[j], "effect": None, "se": None, "significant": False,
                                      "estimable": False, "reason": reasons.get(j, "")} for j in range(k)]
    out: dict[str, Any] = {"effects": effects, "interactions": [], "n": len(y), "df": max(df, 0)}
    if not kept:
        return out
    x = [[c[r] for c in scols] + [kcols[j][r] for j in kept] for r in range(len(y))]
    cov = invert([[sum(r[a] * r[b] for r in x) for b in range(p)] for a in range(p)])
    xty = [sum(r[a] * v for r, v in zip(x, y)) for a in range(p)]
    beta = [sum(cov[a][b] * xty[b] for b in range(p)) for a in range(p)]
    res = [v - sum(r[a] * beta[a] for a in range(p)) for r, v in zip(x, y)]
    s2 = sum(e * e for e in res) / df
    for m, j in enumerate(kept):
        a = len(seeds) + m
        eff, se = 2 * beta[a], 2 * math.sqrt(max(0.0, s2 * cov[a][a]))
        effects[j] = {"key": keys[j], "effect": eff, "se": se, "significant": abs(eff) >= z * se > 0,
                      "estimable": True}
        # missing rows can drop the scored subset below resolution IV: a main then carries a 2FI
        al = [f"{keys[a]}*{keys[b]}" for a, b in itertools.combinations([i for i in kept if i != j], 2)
              if (w := [kcols[a][r] * kcols[b][r] for r in range(len(y))]) == kcols[j] or [-v for v in w] == kcols[j]]
        if al:
            effects[j]["aliased_2fi"] = al
    chains: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for a, b in itertools.combinations(range(k), 2):
        chains.setdefault(tuple(r[a] * r[b] for r in rows), []).append((a, b))
    out["chains"] = []
    for col, pairs in chains.items():
        w0 = [float(col[i]) for i, _, _ in obs]
        g = [sum(cov[a][b] * sum(r[b] * v for r, v in zip(x, w0)) for b in range(p)) for a in range(p)]
        w = [v - sum(r[a] * g[a] for a in range(p)) for r, v in zip(x, w0)]     # its part the fit cannot absorb
        ww = sum(v * v for v in w)
        # a chain in the span of the seed intercepts (a blocked design's block words) / the mains has a residual
        # contrast of exactly 0 with a fake se: confounded, never an estimate
        if ww <= 1e-9 * sum(v * v for v in w0):
            continue
        c, se = 2 * sum(v * e for v, e in zip(w, res)) / ww, 2 * math.sqrt(s2 / ww)
        names = [f"{keys[a]}*{keys[b]}" for a, b in pairs]
        out["chains"].append({"pairs": [[keys[a], keys[b]] for a, b in pairs], "c": c, "se": se})
        if abs(c) >= z * se > 0:
            out["interactions"].append(f"{' = '.join(names)}: {c:+.5f}±{se:.5f}")
    return out


def fmt_effect(e: Mapping[str, Any]) -> str:
    if not e.get("estimable", e.get("effect") is not None):
        return f"not estimable ({e.get('reason', '?')})"
    return f"{e['effect']:+.5f} ± {e['se']:.5f}{' *' if e['significant'] else ''}" + (
        f" (aliased with {' = '.join(e['aliased_2fi'])} over the scored rows)" if e.get("aliased_2fi") else "")


def stacked_levels(knobs: Sequence[Mapping[str, Any]], effects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """key -> the significantly better level; a knob without a significant effect stays at root."""
    return {kb["key"]: kb["high"] if e["effect"] > 0 else kb["low"]
            for kb, e in zip(knobs, effects) if e["significant"]}


# ------------------------------------------------------------------------------------ knob choice
def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _sign(v: float) -> int:
    return (v > 0) - (v < 0)


def _same_shape(root: Any, v: Any) -> bool:
    """``v`` is the kind of value root holds: a number for a number, a bool for a bool, a list of the
    same length (element-wise) for a list, a str for a str (None / other types: anything)."""
    if isinstance(root, bool) or isinstance(v, bool):
        return isinstance(root, bool) and isinstance(v, bool)
    if _num(root):
        return _num(v)
    if isinstance(root, (list, tuple)):
        return isinstance(v, (list, tuple)) and len(v) == len(root) and all(map(_same_shape, root, v))
    if isinstance(root, str):
        return isinstance(v, str)
    return True


def _knob_ok(ctx: LoopContext, text: str, base: Mapping[str, Any], kb: Mapping[str, Any]) -> bool:
    key = kb["key"]
    if key not in base or kb["high"] is None or kb["low"] == kb["high"] or not all(
            _same_shape(base[key], kb[lv]) for lv in ("low", "high")) or not (fam := intake.family_of(ctx, key)) \
            or ctx.cfg.families[fam].init_check or any(   # review-requested knobs too: no init_check family
            key == pk or key.startswith(pk + ".") for pk in ctx.spec.surface.protected_keys):
        return False
    try:
        return all(flatten(yaml.safe_load(set_yaml_value(text, key, v)) or {}).get(key) == v
                   for v in (kb["low"], kb["high"]))
    except (ValueError, yaml.YAMLError):
        return False


def _step(v: Any, d: int, step: float) -> Any:
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        if v == 0:                                     # 0 is a count / width / sentinel: no x-step, and
            return None                                # 0 - 1 = -1 crashed every d2 point at its high level
        new = round(v * step ** d)
        new = new if new != v else v + d
        return new if _sign(new) == _sign(v) else None        # never 1 -> 0, never across 0
    if isinstance(v, float) and 0.5 < v < 1:           # a beta / momentum: step the distance to 1
        return float(f"{1 - (1 - v) / step ** d:.6g}")
    return float(f"{v * step ** d:.4g}") if isinstance(v, float) and v else None


SCREENED_MAX = 3   # a knob level that many design / tuning points set and were screened out is not tried again


def screened_out(store: Any) -> dict[tuple[str, str], int]:
    """(key, level as JSON) -> how many design / tuning points that set it were screened out (dino.lr=2.344e-4: 11
    points, 9 screened out over t3..t9, and t12 tried it again). A level some CONFIRMED node's config carries is
    not counted (d1's factorial points carried kde_concentration 6.667, now in a node's confirmed stack)."""
    won = {(k, json.dumps(v)) for c in store.nodes(status=[NodeStatus.CONFIRMED])
           for k, v in intake.parent_config(store, c.id).items()}
    out: dict[tuple[str, str], int] = {}
    for n in store.nodes(status=[NodeStatus.SCREENED_OUT]):
        if design_point(n) and "campaign_of=" not in (n.notes or ""):     # a campaign's levels are its node's own
            for k, v in (store.get_state(f"config_diff:{n.id}") or {}).items():
                if (k, json.dumps(v)) not in won:
                    out[(k, json.dumps(v))] = out.get((k, json.dumps(v)), 0) + 1
    return out


def fresh_level(store: Any, kb: Mapping[str, Any], memo: dict | None = None) -> bool:
    """The knob's high level was screened out fewer than SCREENED_MAX times (``memo``: a screened_out() table)."""
    tab = screened_out(store) if memo is None else memo
    return tab.get((str(kb["key"]), json.dumps(kb["high"])), 0) < SCREENED_MAX


def pick_knobs(ctx: LoopContext, store: Any, snapshot: Any, text: str, base: Mapping[str, Any],
               requested: Sequence[Mapping[str, Any]] = (), auto: bool = True) -> list[dict[str, Any]]:
    """Up to ``design.max_knobs`` knobs {key, low, high, levers}: the review's (low defaults to root's value;
    a review states their mechanism), then -- only MAJOR knobs, backed by the lever regression (see
    :func:`backed`) -- redirected sweeps and the most-probed keys not designed before (outside init_check
    families). An auto knob's high level is a
    ``design.step`` move from root's value toward the side its probes' shrunk U favours (x or / step; a
    float in (0.5, 1) moves its distance to 1 instead; an int moves >= 1; a bool flips;
    other types take the best positive probe's value); its levers are the probes' registered levers,
    signed by that move."""
    vocab, out, cap = ctx.cfg.levers.vocabulary, [], ctx.cfg.design.max_knobs
    bad = {(e["key"], json.dumps(e["level"])) for e in store.get_state(K_DESIGN, {}).get("bad_levels", ())}
    seen = screened_out(store)

    def ok(kb: Mapping[str, Any], auto_pick: bool = True) -> bool:   # no level crashed, high not screened out 3x
        return not any((kb["key"], json.dumps(kb[lv])) in bad for lv in ("low", "high")) \
            and fresh_level(store, kb, seen) \
            and _knob_ok(ctx, text, base, kb) and (not auto_pick or backed(ctx, snapshot, kb["levers"]))
    for r in requested:
        kb = {"key": str(r.get("key", "")), "low": r.get("low", base.get(str(r.get("key")))),
              "high": r.get("high"), "levers": {l: d for l, d in dict(r.get("levers") or {}).items()
                                                if l in vocab and d in (-1, 1)}}
        if ok(kb, False) and kb["key"] not in {o["key"] for o in out}:
            out.append(kb)
    if not auto:                                 # loop.components bypassed the auto top-up: review knobs only
        return out[:cap]
    for r in reversed(NV.pool_entries(store)):  # redirected ideation sweeps (loop.novelty), newest first
        kb = {"key": r["key"], "low": base.get(r["key"]), "high": r["high"],
              "levers": {l: d for l, d in dict(r.get("levers") or {}).items() if l in vocab and d in (-1, 1)}}
        if len(out) < cap and kb["key"] not in {o["key"] for o in out} and ok(kb):
            out.append(kb)
    done = {kb["key"] for b in store.get_state(K_DESIGN, {}).get("batches", []) for kb in b.get("knobs", ())}
    probes: dict[str, list[tuple[Any, float, Node]]] = {}
    for n in store.nodes():
        post = snapshot.nodes.get(n.id)
        if post is None or n.id == ROOT_NODE_ID or n.strategy is Strategy.DESIGN:
            continue
        for key, v in (store.get_state(f"config_diff:{n.id}") or {}).items():
            probes.setdefault(key, []).append((v, post.utility.mean, n))
    for key, ps in sorted(probes.items(), key=lambda kv: (-len(kv[1]), -abs(sum(u for _, u, _ in kv[1])), kv[0])):
        fam = intake.family_of(ctx, key)
        if len(out) >= cap or key in done or key in {o["key"] for o in out} or not fam \
                or ctx.cfg.families[fam].init_check or key not in base:
            continue
        root = base[key]
        moves = [(_sign(v - root) if _num(v) and _num(root) else 1, u, n) for v, u, n in ps]
        d = _sign(sum(m * u for m, u, _ in moves)) or 1
        high = _step(root, d, ctx.cfg.design.step) if _num(root) or isinstance(root, bool) else \
            max(((v, u) for v, u, _ in ps if u > 0), key=lambda t: t[1], default=(None, 0))[0]
        lev = {l: _sign(sum(m * d * n.levers.directions.get(l, 0) for m, _, n in moves)) for l in vocab}
        kb = {"key": key, "low": root, "high": high, "levers": {l: v for l, v in lev.items() if v}}
        if high is not None and ok(kb):
            out.append(kb)
    return out[:cap]


def backed(ctx: LoopContext, snapshot: Any, levers: Mapping[str, int]) -> bool:
    """A MAJOR knob (``design.lever_p`` > 0): one of its levers has panel P(effect in the knob's direction)
    >= design.lever_p in the world model's lever regression. Always true at lever_p 0."""
    p = ctx.cfg.design.lever_p
    eff = {e.lever: e.p_positive for e in getattr(snapshot, "levers", ()) or () if e.target == PANEL}
    return p <= 0 or any((eff[l] if d > 0 else 1 - eff[l]) >= p for l, d in levers.items() if l in eff)


# ----------------------------------------------------------------------------------- the batch
def request(store: Any, knobs: Sequence[Mapping[str, Any]]) -> None:
    """A review asked for a designed batch over ``knobs`` (the next tick starts it)."""
    book = store.get_state(K_DESIGN, {})
    store.set_state(K_DESIGN, {**book, "request": list(knobs)})


def tick(ctx: LoopContext, store: Any, adapter: Any, snapshot: Any, *, evals: int, evals_at_confirm: int,
         new_id: Any, paused: bool = False) -> list[str]:
    """Finish the running batch once every point is closed; else start one on a review request or when
    ``design.stall_closes`` closes passed without a confirmation (and since the last batch). ``paused`` (the
    driver's scarce mode): no new batch starts (a review request waits)."""
    book = {"current": None, "batches": [], "request": None, "idle_at": 0, **store.get_state(K_DESIGN, {})}
    if acts := _reanalyse(ctx, store, adapter, book, evals, new_id):
        return acts
    cur, stall = book["current"], ctx.cfg.design.stall_closes
    try:
        if cur is not None:
            pts = [store.get_node(i) for i in cur["points"].values() if i != ROOT_NODE_ID]
            if any(n is None or n.status not in CLOSED_STATUSES for n in pts):
                return []
            return _finish(ctx, store, adapter, book, evals, new_id)
        if paused or not book["request"] and not (stall and evals - max(evals_at_confirm, book["idle_at"]) >= stall):
            return []
        req, book["request"], book["idle_at"] = book["request"] or (), None, evals
        store.set_state(K_DESIGN, book)
        return _start(ctx, store, adapter, snapshot, book, req, new_id)
    except Exception as exc:  # noqa: BLE001 — a git / adapter failure closes the batch, never the loop
        if cur is not None:
            book["batches"] = book["batches"] + [{**cur, "error": repr(exc)[:300], "stacked": ""}]
        book["current"] = None
        store.set_state(K_DESIGN, book)
        store.log("design_error", error=repr(exc)[:500])
        return [f"design error: {exc!r}"[:300]]


def _reanalyse(ctx: LoopContext, store: Any, adapter: Any, book: dict, evals: int, new_id: Any) -> list[str]:
    """One closed batch that has no analysis (closed 'singular matrix' before the rank-checked OLS, or by
    an adapter error) is re-analysed from its runs — once, beside any running batch."""
    for slot, old in enumerate(book["batches"]):
        if "effects" in old or old.get("reanalysed"):
            continue
        book["batches"][slot] = {**old, "reanalysed": True}      # once, even if this attempt fails
        store.set_state(K_DESIGN, book)
        try:
            if (again := rebuild(store, old)) is None:
                store.log("design_reanalysis_skipped", batch=old.get("batch"), reason="points not recoverable")
                return [f"design d{old.get('batch')}: re-analysis skipped (points not recoverable)"]
            return _finish(ctx, store, adapter, book, evals, new_id, slot=slot, cur=again)
        except Exception as exc:  # noqa: BLE001 — never touches the running batch
            store.log("design_error", batch=old.get("batch"), error=repr(exc)[:500], reanalysis=True)
            return [f"design d{old.get('batch')} re-analysis error: {exc!r}"[:300]]
    return []


class _Tree:
    """A worktree on the base node's commit (root's, or a moonshot's for loop.tune: the config text and
    values) that writes config-only nodes whose parent is that base."""

    def __init__(self, ctx: LoopContext, store: Any, adapter: Any, name: str, base_id: str = ROOT_NODE_ID) -> None:
        self.ctx, self.store, self.adapter = ctx, store, adapter
        self.root = store.get_node(base_id)
        self.diff_text = ""
        self.wt = adapter.prepare_worktree(parent_commit=self.root.commit, name=name)
        self.rel = config_file(ctx, adapter)
        self.text = (Path(self.wt.path) / self.rel).read_text()
        self.base = flatten(yaml.safe_load(self.text) or {})

    def __enter__(self) -> "_Tree":
        return self

    def __exit__(self, *_: Any) -> None:
        try:
            self.adapter.reap_worktree(self.wt)
        except Exception:  # noqa: BLE001 — best effort; the driver re-reaps orphans
            pass

    def node(self, levels: Mapping[str, Any], *, knobs: Sequence[Mapping[str, Any]], tag: str,
             operator: Operator | None, new_id: Any, hyp: HypothesisRecord | None = None,
             notes: str = "") -> str:
        """A SCREENING node with the base's config at ``levels``: its id, the base's when that IS the base's
        config, '' when an intake rule (surface, protected, exact dup, family map, forbid, init-compat) fires."""
        changed = {k: v for k, v in levels.items() if v != self.base[k]}
        if not changed:
            return self.root.id
        text = self.text
        for k, v in changed.items():
            text = set_yaml_value(text, k, v)
        (Path(self.wt.path) / self.rel).write_text(text)
        diff_text, paths, cfg_diff = self.adapter.diff(self.wt)
        self.diff_text = diff_text
        now, tid = time.time(), self.ctx.task_id
        h = hyp or HypothesisRecord(id="", task_id=tid, kind=HypothesisKind.TRIAL, claim=f"design point {tag}",
                                    mechanism_tag=tag, family=DESIGN_FAMILY, predictions={}, falsification="")
        salt = "" if self.root.id == ROOT_NODE_ID else f"{self.root.commit}\n"   # same text, another base
        cand = Candidate(tag, tid, "", self.root.commit, diff_text, diff_hash(salt + diff_text), tuple(paths),
                         dict(cfg_diff), h, HandoffNote("", tid, tag, "", "", ""))
        rule, _ = intake.diagnose(self.ctx, self.store, cand, operator=operator, embedder=None,
                                  strategy=Strategy.DESIGN)
        commit = self.adapter.commit(self.wt, message=f"loop design {tag}: {changed}") if not rule else ""
        if not rule and intake.needs_init_compat(self.ctx, cand, intake.derive_families(self.ctx, cand)[0]) \
                and not (ic := self.adapter.init_compat(self.wt, parent_commit=self.root.commit)).ok:
            rule = ic.rule
        if rule:
            self.store.log("design_point_rejected", tag=tag, reason=rule, levels=dict(changed))
            return ""
        lev: dict[str, int] = {}                  # the moved knobs' levers, signed by the move
        for kb in (kb for kb in knobs if kb["key"] in changed):
            for l, d in kb["levers"].items():
                lev[l] = lev.get(l, 0) + d * (1 if changed[kb["key"]] == kb["high"] else -1)
        nid = new_id()
        if hyp is not None:
            self.store.put_hypothesis(dataclasses.replace(hyp, id=f"h-{nid}", created_ts=now))
            self.store.link(f"h-{nid}", nid, sign=1, auto=False)
        self.store.put_node(Node(
            id=nid, task_id=tid, commit=commit, parent_id=self.root.id, strategy=Strategy.DESIGN,
            operator=operator, family=DESIGN_FAMILY, mechanism_tag=tag, diff_hash=cand.diff_hash,
            status=NodeStatus.SCREENING, hypothesis_id=f"h-{nid}" if hyp is not None else None,
            levers=LeverTags({l: _sign(d) for l, d in lev.items() if d}), created_ts=now, updated_ts=now,
            notes=(notes + " " if notes else "") + f"design {', '.join(f'{k}={v}' for k, v in changed.items())}"))
        self.store.set_state(f"config_diff:{nid}", dict(cfg_diff))
        return nid


def aliased(rows: Sequence[Sequence[int]]) -> list[int]:
    """Columns of ``rows`` that are constant or +-equal to an earlier column (never, for a proper
    resolution-IV matrix: a guard against a bad generator / knob list)."""
    cols = [tuple(c) for c in zip(*rows)]
    return [j for j, c in enumerate(cols) if len(set(c)) < 2 or any(
        d in (c, tuple(-x for x in c)) for d in cols[:j])]


def _start(ctx: LoopContext, store: Any, adapter: Any, snapshot: Any, book: dict, req: Sequence[Mapping],
           new_id: Any) -> list[str]:
    b = len(book["batches"]) + 1
    with _Tree(ctx, store, adapter, f"design{b}") as t:
        auto = CG.use(ctx, store, "design_auto", "top_up", f"d{b}", tag=f"d{b}stacked")
        knobs = pick_knobs(ctx, store, snapshot, t.text, t.base, req, auto)   # no dup key, low != high, root's shape
        if len(knobs) < 4:
            store.log("design_skipped", knobs=[kb["key"] for kb in knobs])
            return [f"design: only {len(knobs)} usable knob(s) {[kb['key'] for kb in knobs]}; need 4"]
        rows = matrix(len(knobs), n_runs(len(knobs), ctx.cfg.design.runs))
        if bad := aliased(rows):
            raise ValueError(f"design matrix aliases {[knobs[j]['key'] for j in bad]}")
        points = {}
        for i, row in enumerate(rows):
            nid = t.node({kb["key"]: kb["high"] if x > 0 else kb["low"] for kb, x in zip(knobs, row)},
                         knobs=knobs, tag=f"d{b}pt{i:02d}", operator=None, new_id=new_id)
            if nid:
                points[str(i)] = nid
    keys = [kb["key"] for kb in knobs]
    NV.consume(store, keys)
    sm = [points[str(i)] for i in smoke_rows(knobs, rows, t.base, [int(i) for i, n in points.items()
                                                                   if n != ROOT_NODE_ID])]
    smoke = {"point": sm[0], "queue": sm[1:], "mitigation": "", "status": "pending"} if sm else None
    plan = analyze(keys, rows, [(int(i), 0, 0.0) for i in points], z=1.0)   # identifiable from the points made?
    lost = {e["key"]: e["reason"] for e in plan["effects"] if not e["estimable"]}
    book["current"] = {"batch": b, "knobs": knobs, "rows": [list(r) for r in rows], "points": points,
                       **({"smoke": smoke} if smoke else {})}
    store.set_state(K_DESIGN, book)
    store.log("design_started", batch=b, knobs=knobs, runs=len(rows), points=len(points), unestimable=lost)
    n_pts = sum(p != ROOT_NODE_ID for p in points.values())
    return [f"design d{b} started: {len(knobs)} knobs x {len(rows)} runs ({n_pts} point nodes): "
            f"{', '.join(keys)}" + (f"; NOT estimable from the points made: {lost}" if lost else "")
            + (f"; held for a training smoke of {', '.join([smoke['point'], *smoke['queue']])}" if smoke else "")]


# ------------------------------------------------------------------------ memory pre-smoke
#: config keys whose larger level costs GPU memory (crop sizes, views, batch, width, depth)
MEMORY_KEY = re.compile(r"global_size|local_size|batch_size|local_views|views|width|depth|blocks|hidden_dim", re.I)
#: keys whose LOWER level costs memory (a smaller patch is more tokens)
INVERSE_MEMORY_KEY = re.compile(r"patch_size", re.I)


def _mass(v: Any) -> float | None:
    if isinstance(v, bool) or _num(v):
        return float(v)
    if isinstance(v, (list, tuple)) and v and all(_num(x) for x in v):
        return float(sum(v))
    return None


def _mem_score(knobs: Sequence[Mapping[str, Any]], rows: Sequence[Sequence[int]]) -> list[int]:
    """Per row: how many memory knobs sit at their larger level."""
    big = [(j, (1 if hi > lo else -1) * (-1 if INVERSE_MEMORY_KEY.search(leaf) else 1)) for j, kb in enumerate(knobs)
           if (leaf := kb["key"].rsplit(".", 1)[-1]) and (MEMORY_KEY.search(leaf) or INVERSE_MEMORY_KEY.search(leaf))
           and (lo := _mass(kb["low"])) is not None and (hi := _mass(kb["high"])) is not None and hi != lo]
    return [sum(row[j] == s for j, s in big) for row in rows]


def heaviest(knobs: Sequence[Mapping[str, Any]], rows: Sequence[Sequence[int]]) -> int | None:
    """The row with the most memory knobs at their larger level (None: no memory knob, or no row
    heavier than root's all-low row)."""
    score = _mem_score(knobs, rows)
    best = max(range(len(rows)), key=lambda i: (score[i], -i))
    return best if score[best] > score[0] else None


def smoke_rows(knobs: Sequence[Mapping[str, Any]], rows: Sequence[Sequence[int]], base: Mapping[str, Any],
               avail: Sequence[int]) -> list[int]:
    """The rows (of ``avail``, the rows that got a point node) to smoke before ANY point launches: together
    they run every knob level that is not root's own value (root's level is known to train), the first
    one the heaviest in memory. Usually one row, else its mirror too. d3 had no memory knob, was never
    smoked, and crashed 32/32 on a level the smoke would have hit in minutes."""
    mem = _mem_score(knobs, rows)
    moved = {i: {j for j, kb in enumerate(knobs) if kb["high" if rows[i][j] > 0 else "low"] != base.get(kb["key"])}
             for i in avail}
    left, out = set().union(*moved.values()), []
    while left:
        i = max(sorted(moved), key=lambda i: (mem[i] if not out else 0, len(moved[i] & left), mem[i], -i))
        out.append(i)
        left -= moved[i]
    return out


#: a smoke that died in training code (a traceback), not in the queue / infra: the POINT's config crashes
CRASHED = re.compile(r"^Traceback|\b\w+(Error|Exception): ", re.M)


def _abort(store: Any, book: dict, sm: dict, res: SmokeResult) -> list[str]:
    """The smoked point crashed (not OOM): the batch is withdrawn before any point launched. The levels
    to blame are the point's non-root levels whose key's leaf is named in the traceback (every one of
    them when none is); each is recorded in ``bad_levels`` (never picked again at that level) and the
    batch's knobs are re-requested, so the next tick designs the batch without the offending levels."""
    cur = book["current"]
    b, tail = cur["batch"], res.log_tail or ""
    moved = store.get_state(f"config_diff:{sm['point']}") or {}
    hit = {k: v for k, v in moved.items() if re.search(rf"\b{re.escape(k.rsplit('.', 1)[-1])}\b", tail)}
    err = next((ln.strip() for ln in reversed(tail.splitlines()) if re.search(r"(Error|Exception)\b", ln)),
               res.reason)[:200]
    why = f"d{b} smoke of {sm['point']} crashed{'' if hit else ' (unattributed: every moved level)'}: {err}"
    known = {(e["key"], json.dumps(e["level"])) for e in book.get("bad_levels", [])}
    book["bad_levels"] = book.get("bad_levels", []) + [
        {"key": k, "level": v, "batch": b, "why": why} for k, v in (hit or moved).items()
        if (k, json.dumps(v)) not in known]
    for nid in cur["points"].values():
        if nid != ROOT_NODE_ID and (n := store.get_node(nid)) is not None and n.status not in CLOSED_STATUSES:
            store.update_node(nid, status=NodeStatus.REJECTED, reject_reason="design_smoke_crash")
    book["batches"] = book["batches"] + [{"batch": b, "knobs": cur["knobs"], "stacked": "", "error": why,
                                          "aborted": True, "reanalysed": True}]
    book.update(current=None, request=list(cur["knobs"]))
    store.set_state(K_DESIGN, book)
    store.log("design_smoke", batch=b, point=sm["point"], result="crashed", blamed=hit or moved,
              attributed=bool(hit), reason=err)
    return [f"design d{b} withdrawn before launch: {why}; dropped levels "
            f"{', '.join(f'{k}={v!r}' for k, v in (hit or moved).items())}; re-designing without them"]


def _smoke(store: Any) -> tuple[dict, dict | None]:
    book = store.get_state(K_DESIGN, {}) or {}
    return book, (book.get("current") or {}).get("smoke")


def current_points(store: Any) -> set[str]:
    return set(((store.get_state(K_DESIGN, {}) or {}).get("current") or {}).get("points", {}).values())


def held(store: Any) -> set[str]:
    """The running batch's points while its memory smoke is pending: nothing of theirs launches."""
    _, sm = _smoke(store)
    return current_points(store) - {ROOT_NODE_ID} if sm and sm["status"] == "pending" else set()


def smoke_due(store: Any) -> tuple[str, str] | None:
    """(heaviest point, OOM step) to smoke next; None when no smoke is pending."""
    _, sm = _smoke(store)
    return (sm["point"], sm["mitigation"]) if sm and sm["status"] == "pending" else None


def smoke_point(adapter: Any, commit: str, name: str, mitigation: str) -> SmokeResult:
    """A memory smoke (the point's own batch shape) of ``commit`` at an OOM step; worktree reaped."""
    wt = adapter.prepare_worktree(parent_commit=commit, name=f"memsmoke-{name}")
    try:
        return adapter.smoke(wt, memory=True, mitigation=mitigation)
    finally:
        try:
            adapter.reap_worktree(wt)
        except Exception:  # noqa: BLE001 — the driver re-reaps orphans
            pass


#: a design smoke that never got a slot (queue timeout, preempted) is re-smoked this many times, then released
MAX_SMOKE_UNAVAILABLE = 3


def smoke_done(store: Any, res: SmokeResult, ladder: Sequence[str]) -> list[str]:
    """Passed: the next queued smoke row runs, else the batch is released, every point at the step that
    passed. OOM: the next ladder step is smoked; the ladder spent, every point runs at its strongest step
    (each still gets the run-level OOM retry). A training crash (a traceback) withdraws the batch
    (``_abort``). Unavailable (no slot in time: no verdict) is re-smoked up to MAX_SMOKE_UNAVAILABLE times.
    Any other failure (infra, a vanished node, unavailable past the cap) releases the batch as is."""
    book, sm = _smoke(store)
    if not sm or sm["status"] != "pending":
        return []
    cur, b = book["current"], book["current"]["batch"]
    oom = not res.ok and (is_oom(res.log_tail) or "OOM" in res.reason)
    if not res.ok and not oom and CRASHED.search(res.log_tail or ""):
        return _abort(store, book, sm, res)
    if getattr(res, "unavailable", False) and sm.get("unavailable", 0) < MAX_SMOKE_UNAVAILABLE:
        sm["unavailable"] = sm.get("unavailable", 0) + 1      # no slot in time: no verdict; smoked again next
        store.set_state(K_DESIGN, book)                       # tick, placed afresh (RunBook.smoke_pool)
        store.log("design_smoke", batch=b, point=sm["point"], result="unavailable", retry=sm["unavailable"],
                  reason=res.reason[:300])
        return [f"design d{b}: smoke of {sm['point']} got no slot in time ({res.reason}); "
                f"re-smoking ({sm['unavailable']}/{MAX_SMOKE_UNAVAILABLE})"]
    if res.ok and sm.get("queue"):              # the next smoke row runs the levels this one did not
        was, nxt = sm["point"], sm["queue"][0]
        sm.update(point=nxt, queue=sm["queue"][1:], last="")
        store.set_state(K_DESIGN, book)
        store.log("design_smoke", batch=b, point=was, result="passed", next=nxt, mitigation=sm["mitigation"])
        return [f"design d{b}: smoke of {was} passed; smoking {nxt} for the remaining levels"]
    if oom and (nxt := next_mitigation(ladder, sm["mitigation"])):
        was = sm["mitigation"] or "no mitigation"
        sm.update(mitigation=nxt, last=res.reason[:200])
        store.set_state(K_DESIGN, book)
        store.log("design_smoke", batch=b, point=sm["point"], result="oom", at=was, next=nxt)
        return [f"design d{b}: heaviest point {sm['point']} OOMs with {was}; smoking again with {nxt}"]
    step = sm["mitigation"] or (ladder[-1] if oom and ladder else "")
    sm.update(status="passed" if res.ok else "oom" if oom else "failed", mitigation=step, last=res.reason[:200])
    for nid in cur["points"].values():
        if step and nid != ROOT_NODE_ID and (n := store.get_node(nid)) is not None:
            store.update_node(nid, notes=with_oom_note(n.notes, step))
    store.set_state(K_DESIGN, book)
    store.log("design_smoke", batch=b, point=sm["point"], result=sm["status"], mitigation=step, reason=res.reason[:300])
    return [f"design d{b}: memory smoke {sm['status']} ({res.reason or 'ok'}); batch released"
            + (f", every point runs with {step}" if step else "")]


def rebuild(store: Any, res: Mapping[str, Any]) -> dict[str, Any] | None:
    """A closed batch's {batch, knobs, rows, points} for re-analysis: as recorded, else recovered from
    its point nodes (tag ``d{b}ptNN``) — the 16- or 32-run matrix every point's config diff agrees with;
    a missing row 0 (all low) is root's baseline."""
    if res.get("rows") and res.get("points"):
        return {k: res[k] for k in ("batch", "knobs", "rows", "points")}
    b, knobs = res.get("batch"), res.get("knobs") or []
    pat = re.compile(rf"d{b}pt(\d+)$")
    pts = {int(m.group(1)): n.id for n in store.nodes()
           if n.strategy is Strategy.DESIGN and n.operator is None and (m := pat.match(n.mechanism_tag or ""))}
    if not pts or not knobs:
        return None

    def agrees(nid: str, row: Sequence[int]) -> bool:
        diff = store.get_state(f"config_diff:{nid}") or {}
        return all(diff[kb["key"]] == (kb["high"] if x > 0 else kb["low"]) if kb["key"] in diff else x < 0
                   for kb, x in zip(knobs, row))
    for runs in (16, 32):
        if len(knobs) in GENERATORS[runs] and max(pts) < runs:
            rows = matrix(len(knobs), runs)
            if all(agrees(nid, rows[i]) for i, nid in pts.items()):
                points = {str(i): nid for i, nid in sorted(pts.items())}
                points.setdefault("0", ROOT_NODE_ID)
                return {"batch": b, "knobs": list(knobs), "rows": [list(r) for r in rows], "points": points}
    return None


def panel_of(ctx: LoopContext, r: Any) -> float | None:
    """A finished run's weighted panel score (None unless done with every axis)."""
    return sum(ctx.cfg.weights[a] * r.scores[a] for a in ctx.cfg.axes) \
        if r.status is RunStatus.DONE and all(a in r.scores for a in ctx.cfg.axes) else None


def _obs(ctx: LoopContext, store: Any, cur: Mapping[str, Any]) -> list[tuple[int, int, float]]:
    """(row, seed, panel) of every finished point run; the base's control runs at those seeds for its row
    (root's baselines; a tuning pass's moonshot: its PARENT runs, ``cur['base']``)."""
    base = cur.get("base", ROOT_NODE_ID)
    out, seeds = [], set()
    for i, nid in cur["points"].items():
        for r in store.runs(node_id=nid, stage=Stage.SCREEN) if nid != base else ():
            if (y := panel_of(ctx, r)) is not None:
                out.append((int(i), r.seed, y))
                seeds.add(r.seed)
    root_row = next((int(i) for i, nid in cur["points"].items() if nid == base), None)
    for r in store.runs(node_id=base, stage=Stage.BASELINE if base == ROOT_NODE_ID else Stage.PARENT) \
            if root_row is not None else ():
        if r.seed in seeds and (y := panel_of(ctx, r)) is not None:
            out.append((root_row, r.seed, y))
    return out


DESIGN_RETRIES = 1   # re-runs of a batch's voided points when nothing is estimable


def _level_failures(store: Any, cur: Mapping[str, Any], j: int) -> str:
    """Which of knob j's levels never scored: '; high scored 0/16 points (crashed 16)'."""
    out = []
    for lv, sign in (("high", 1), ("low", -1)):
        pts = [nid for i, nid in cur["points"].items() if cur["rows"][int(i)][j] == sign and nid != ROOT_NODE_ID]
        nodes = [store.get_node(nid) for nid in pts]
        scored = sum(any(r.status is RunStatus.DONE for r in store.runs(node_id=nid, stage=Stage.SCREEN)) for nid in pts)
        if pts and scored < len(pts):
            st: dict[str, int] = {}
            for n in nodes:
                if n is not None and n.status is not NodeStatus.SCREENED_OUT:
                    st[n.status.value] = st.get(n.status.value, 0) + 1
            out.append(f"{lv} scored {scored}/{len(pts)} points"
                       + (f" ({', '.join(f'{k} {v}' for k, v in sorted(st.items()))})" if st else ""))
    return "; ".join(out)


def _dead_levels(store: Any, cur: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Knob levels whose every point crashed / voided (at least one crash) while the knob's other level
    scored: that LEVEL crashes (d2 jepa_pred_width=-1, d6 local_size=192 OOM), not the batch."""
    scored: dict[tuple[int, int], bool] = {}
    crashed: dict[tuple[int, int], bool] = {}
    for i, nid in cur["points"].items():
        runs = store.runs(node_id=nid, stage=Stage.BASELINE if nid == ROOT_NODE_ID else Stage.SCREEN)
        ok = any(r.status is RunStatus.DONE for r in runs)
        bad = nid != ROOT_NODE_ID and any(r.status is RunStatus.CRASH for r in runs)
        for j, x in enumerate(cur["rows"][int(i)]):
            scored[j, x] = scored.get((j, x), False) or ok
            crashed[j, x] = crashed.get((j, x), False) or bad
    return [{"key": kb["key"], "level": kb[lv], "batch": cur["batch"],
             "why": f"d{cur['batch']}: every {lv}-level point crashed/voided, the {'low' if x > 0 else 'high'} level scored"}
            for j, kb in enumerate(cur["knobs"]) for lv, x in (("high", 1), ("low", -1))
            if crashed.get((j, x)) and not scored.get((j, x)) and scored.get((j, -x))]


def _reopen_voids(store: Any, cur: Mapping[str, Any]) -> list[str]:
    """Put the batch's VOID points (infra / preempted voids only: a crash is deterministic) back to
    SCREENING with their void screen runs' attempts reset, so the driver's owed-retry path re-runs them."""
    out = []
    for nid in cur["points"].values():
        node = store.get_node(nid) if nid != ROOT_NODE_ID else None
        if node is None or node.status is not NodeStatus.VOID:
            continue
        voids = [r for r in store.runs(node_id=nid, stage=Stage.SCREEN) if r.status is RunStatus.VOID]
        if not voids or any(r.void_type not in (None, VoidType.INFRA) for r in voids):
            continue
        for r in voids:
            store.update_run(r.id, attempt=0, void_type=VoidType.INFRA)
        store.update_node(nid, status=NodeStatus.SCREENING)
        store.log("node_transition", node_id=nid, frm=node.status.value, to=NodeStatus.SCREENING.value,
                  reason=f"design d{cur['batch']} retry: nothing estimable yet")
        out.append(nid)
    return out


def _finish(ctx: LoopContext, store: Any, adapter: Any, book: dict, evals: int, new_id: Any,
            slot: int | None = None, cur: Mapping[str, Any] | None = None) -> list[str]:
    """Analyse the closed batch on its scored rows. Nothing estimable: re-run its voided points once
    (the batch stays current) — else close it with the reason logged. ``slot``: a re-analysis of the
    recorded batch ``cur`` at that index (replaced in place; never retried; the running batch untouched)."""
    cur, z = cur or book["current"], ctx.cfg.design.z
    knobs, b = cur["knobs"], cur["batch"]
    ana = analyze([kb["key"] for kb in knobs], cur["rows"], _obs(ctx, store, cur), z)
    for j, e in enumerate(ana["effects"]):
        if not e["estimable"] and (lf := _level_failures(store, cur, j)):
            e["reason"] += f"; {lf}"
    lost = {e["key"]: e["reason"] for e in ana["effects"] if not e["estimable"]}
    known = {(e["key"], json.dumps(e["level"])) for e in book.get("bad_levels", [])}
    if dead := [d for d in _dead_levels(store, cur) if (d["key"], json.dumps(d["level"])) not in known]:
        book["bad_levels"] = book.get("bad_levels", []) + dead
        store.log("design_bad_levels", batch=b, levels=dead)
    if len(lost) == len(knobs) and slot is None and cur.get("retries", 0) < DESIGN_RETRIES \
            and (again := _reopen_voids(store, cur)):
        book["current"] = {**cur, "retries": cur.get("retries", 0) + 1}
        store.set_state(K_DESIGN, book)
        store.log("design_retry", batch=b, points=again, unestimable=lost, n=ana["n"])
        return [f"design d{b}: nothing estimable from {ana['n']} scored runs; re-running {len(again)} voided "
                f"point(s) {again}"]
    result: dict[str, Any] = {"batch": b, "knobs": knobs, "rows": cur["rows"], "points": cur["points"],
                              "stacked": "", **ana}
    if slot is not None:
        result.update(reanalysed=True, prior_error=book["batches"][slot].get("error", ""))
    if lost:
        store.log("design_unestimable", batch=b, unestimable=lost, n=ana["n"], df=ana["df"])
    if len(lost) == len(knobs):
        result["error"] = f"no estimable effect from {ana['n']} scored runs"
        store.log("design_no_estimate", batch=b, n=ana["n"], unestimable=lost)
    acts = [f"design d{b} {'re-analysed' if slot is not None else 'done'}: " + (result.get("error") or "; ".join(
        f"{e['key']} {e['effect']:+.5f}±{e['se']:.5f}{'*' if e['significant'] else ''}"
        for e in ana["effects"] if e["estimable"])) + (f"; not estimable: {lost}" if lost else "")]
    levels = stacked_levels(knobs, ana["effects"])
    if levels:
        gain = sum(abs(e["effect"]) for e in ana["effects"] if e["significant"])
        hyp = HypothesisRecord(
            id="", task_id=ctx.task_id, kind=HypothesisKind.TRIAL, family=DESIGN_FAMILY,
            claim=f"stacked optimum of design d{b}: {levels} (each a significant main effect)",
            mechanism_tag=f"d{b}stacked", predictions={PANEL: AxisPrediction(1, gain)},
            falsification="the stacked node does not beat root on fresh seeds")
        try:                                     # a git / adapter failure must not lose the analysis
            with _Tree(ctx, store, adapter, f"design{b}-stacked") as t:
                nid = t.node(levels, knobs=knobs, tag=f"d{b}stacked", operator=Operator.DRAFT, new_id=new_id,
                             hyp=hyp)
            result["stacked"] = "" if nid == ROOT_NODE_ID else nid      # root: every better level is root's
            acts.append(f"design d{b}: stacked node {result['stacked'] or '(none: rejected or root)'} {levels}")
        except Exception as exc:  # noqa: BLE001
            result["stacked_error"] = repr(exc)[:300]
            store.log("design_error", batch=b, error=repr(exc)[:500], stacked=levels)
            acts.append(f"design d{b}: stacked node {levels} failed: {exc!r}"[:300])
    batches = list(book["batches"])
    if slot is None:
        batches.append(result)
        book.update(current=None, idle_at=evals)
    else:
        batches[slot] = result
    book["batches"] = batches
    store.set_state(K_DESIGN, book)
    store.log("design_done", **{k: v for k, v in result.items() if k not in ("rows", "points")})
    return acts


def lines(store: Any) -> str:
    """Per-knob main effects (± se, * = significant) of every finished batch; the running one's progress."""
    book = store.get_state(K_DESIGN, {})
    out = []
    for r in book.get("batches", []):
        if "effects" not in r:
            out.append(f"design d{r['batch']}: no estimate ({r.get('error', '?')})")
            continue
        out.append(f"design d{r['batch']} (n={r['n']}, df={r['df']}; effect = high - low on the panel, "
                   f"* = |effect| >= z se; stacked node {r['stacked'] or 'none'})"
                   + (f" — {r['error']}" if r.get("error") else "") + ":")
        out += [f"  {e['key']}: {kb['low']!r} -> {kb['high']!r}: {fmt_effect(e)}"
                for kb, e in zip(r["knobs"], r["effects"])]
        out += [f"  aliased 2FI chain flagged: {f}" for f in r["interactions"]]
    if pool := NV.pool_entries(store):
        out.append("design knob pool (redirected ideation sweeps, next batch): "
                   + ", ".join(f"{e['key']}={e['high']!r}" for e in pool[-12:]))
    if cur := book.get("current"):
        pts = [store.get_node(i) for i in cur["points"].values() if i != ROOT_NODE_ID]
        out.append(f"design d{cur['batch']} running: {sum(n.status in CLOSED_STATUSES for n in pts if n)}/"
                   f"{len(pts)} points closed over {', '.join(kb['key'] for kb in cur['knobs'])}")
    return "\n".join(out)
