"""loop.evalonly — EVAL-ONLY READOUT RUNS. A queued candidate whose change lies entirely in the readout
family's spans (``evalonly.family``: forward / encode_image / probe_features / _suppress and train.py's
post-training fit; plus added fitted buffers) with no config change is not retrained: it is grafted onto
the code of trained MODELS (the incumbent, root, the other CONFIRMED nodes) and run with
``train.resume=<that model's run>/latest.pt`` -- the training loop is skipped, the calibration draw and the
child's fit are redone, the locked probe scores it (~23 min instead of ~55).

A readout is good when it generalises across models, not just seeds of one: a node is SCREENED on
``screen_ckpts`` checkpoints spanning >= ``min_models`` distinct models, then CONFIRMED on
``confirm_ckpts`` FRESH checkpoints (other runs, other models first). Each delta pairs the eval run with
the very run whose checkpoint it resumed (same weights, same seed). Noise is the empirical spread of those
deltas across checkpoints (pooled over eval-only nodes, floored at ``sigma_floor``), never the training-seed
sd. Screen: P(mean > 0) >= promotion.p_promote and every model's mean delta > 0; confirm: the same
likelihood-ratio e-value as a normal confirmation (frozen alt/sigma, 1/alpha_confirm), every model still
> 0; the panel-average veto (loop.world.veto_sigmas) applies at both. A winner closes EVAL_PASSED and its
readout, grafted onto the incumbent, is queued as a NORMAL candidate (full retrain, the usual
promote/confirm/audit path) before anything reaches Labless. Eval-only runs hold at most ``max_share`` of
the slots.

NULL CONTROL: when first enabled, each of ``null_models`` models' own code is run eval-only on one of its
own checkpoints; every |final_score - the source run's| must be < ``null_tol`` or eval-only scoring stays
blocked (waiting nodes go back to the queue as normal candidates).

Eval-only runs are ordinary run rows at synthetic seeds (>= EO_SEED; a null control's >= NULL_SEED, stage
AUDIT on the model node), so no world-model pairing, block or audit ever sees them."""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from loop.agents.researcher import diff_hash
from loop.contracts import (PANEL, ROOT_NODE_ID, Node, NodeStatus, Operator, Run, RunStatus, Stage,
                            VoidType, design_point)
from loop.seeds import run_failed
from loop.spec import LoopContext
from loop.world import prob_greater, veto_sigmas
from loop.world.evalues import confirm_delta, lr_evalue, phi

EO_SEED = 900_000         # eval-only run seeds: EO_SEED + k (k = the node's k-th checkpoint)
NULL_SEED = 990_000       # the null control's runs: NULL_SEED + k on the model node, stage AUDIT
K_EO = "evalonly"         # {"null": {...}, "nodes": {id: {...}}, "requeued": [ids], "errors": {id: n}}
MAX_ERRORS = 3            # a node whose lane step raised this many times goes back to the queue as a full run
_PHASE = {"screen": Stage.PROMOTE, "confirm": Stage.CONFIRM}
STAGE_ORDER = {s: i for i, s in enumerate((Stage.BASELINE, Stage.CALIBRATE, Stage.SCREEN, Stage.PROMOTE,
                                             Stage.CONFIRM, Stage.PARENT, Stage.AUDIT))}


def is_eval_run(run: Run) -> bool:
    return run.seed >= EO_SEED


def _complete(ctx: LoopContext, run: Run | None) -> bool:
    return run is not None and run.status is RunStatus.DONE and all(a in run.scores for a in ctx.cfg.axes)


def deltas(ctx: LoopContext, run: Run, src: Run) -> dict[str, float]:
    """target -> eval run minus the run whose checkpoint it resumed (panel = the weighted axes)."""
    out = {a: run.scores[a] - src.scores[a] for a in ctx.cfg.axes}
    out[PANEL] = sum(ctx.cfg.weights[a] * out[a] for a in ctx.cfg.axes)
    return out


def models(ctx: LoopContext, store: Any, snapshot: Any, exists: Callable[[Path], bool] = Path.is_file) -> dict[str, list[Run]]:
    """model node -> its finished scored runs whose latest.pt is on disk, best first: the incumbent, root,
    then the other CONFIRMED nodes by age; runs by stage then seed."""
    inc = snapshot.incumbent_id if snapshot is not None else ROOT_NODE_ID
    ids = list(dict.fromkeys([inc, ROOT_NODE_ID, *(n.id for n in sorted(
        store.nodes(status=[NodeStatus.CONFIRMED]), key=lambda n: (n.created_ts, n.id)))]))
    out = {}
    for mid in ids:
        runs = [r for r in store.runs(node_id=mid, status=[RunStatus.DONE]) if not is_eval_run(r)
                and r.seed < EO_SEED and _complete(ctx, r) and r.output_dir and exists(Path(r.output_dir) / "latest.pt")]
        if runs:
            out[mid] = sorted(runs, key=lambda r: (STAGE_ORDER.get(r.stage, 9), r.seed))
    return out


def pick(pool: Mapping[str, Sequence[Run]], n: int, *, skip: set[str], uses: Counter, min_models: int,
         prefer_new: Sequence[str] = ()) -> list[tuple[str, Run]]:
    """``n`` checkpoints round-robin over the models (models in ``prefer_new``'s complement first), each
    model's least-used run first (rotation: siblings do not all pair with one checkpoint); [] when fewer
    than ``n`` exist or they span fewer than ``min_models`` models."""
    order = sorted(pool, key=lambda m: (m in prefer_new, list(pool).index(m)))
    left = {m: sorted((r for r in pool[m] if r.id not in skip), key=lambda r: uses[r.id]) for m in order}
    out: list[tuple[str, Run]] = []
    while len(out) < n and any(left.values()):
        for m in order:
            if left[m] and len(out) < n:
                out.append((m, left[m].pop(0)))
    return out if len(out) == n and len({m for m, _ in out}) >= min(min_models, n) else []


def sigma(ctx: LoopContext, samples: Mapping[str, Sequence[float]]) -> float:
    """Pooled within-node sd of panel deltas across checkpoints (node -> deltas), floored."""
    ss = df = 0.0
    for xs in samples.values():
        if len(xs) >= 2:
            m = statistics.fmean(xs)
            ss, df = ss + sum((x - m) ** 2 for x in xs), df + len(xs) - 1
    return max(ctx.cfg.evalonly.sigma_floor, math.sqrt(ss / df) if df else 0.0)


def per_model(obs: Sequence[tuple[str, Mapping[str, float]]]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for m, d in obs:
        by.setdefault(m, []).append(d[PANEL])
    return {m: statistics.fmean(xs) for m, xs in by.items()}


def vetoed(ctx: LoopContext, obs: Sequence[tuple[str, Mapping[str, float]]], sig: float) -> tuple[str, ...]:
    """Axes the panel-average veto fires on (loop.world.veto_sigmas), each axis's spread floored like the panel's."""
    n = len(obs)
    panel = (statistics.fmean(d[PANEL] for _, d in obs), sig * sig / n)
    out = []
    for a in ctx.cfg.axes:
        xs = [d[a] for _, d in obs]
        sd = max(ctx.cfg.evalonly.sigma_floor, statistics.stdev(xs) if n > 1 else 0.0)
        if 1.0 - prob_greater((statistics.fmean(xs), sd * sd / n), -veto_sigmas(ctx, panel) * ctx.cfg.sigma(a)) \
                >= ctx.cfg.veto.prob:
            out.append(a)
    return tuple(out)


def screen_verdict(ctx: LoopContext, obs: Sequence[tuple[str, Mapping[str, float]]], sig: float) -> tuple[NodeStatus | None, str]:
    """None = passes to confirmation; else the closing status and why."""
    xs, pm = [d[PANEL] for _, d in obs], per_model(obs)
    p = phi(statistics.fmean(xs) / (sig / len(xs) ** 0.5))
    why = f"eval-only screen mean {statistics.fmean(xs):+.5f} P>0={p:.3f} sigma={sig:.5f} per-model " + \
        ",".join(f"{m}:{v:+.5f}" for m, v in pm.items())
    if bad := vetoed(ctx, obs, sig):
        return NodeStatus.VETOED, f"{why}; harm veto on {list(bad)}"
    if p < ctx.cfg.promotion.p_promote or min(pm.values()) <= 0.0:
        return NodeStatus.SCREENED_OUT, why
    return None, why


def confirm_verdict(ctx: LoopContext, obs: Sequence[tuple[str, Mapping[str, float]]], conf: Sequence[float],
                    alt: float, sig: float) -> tuple[NodeStatus, str, float]:
    e = lr_evalue(conf, alt, sig)
    pm = per_model(obs)
    why = f"eval-only confirm E={e:.3g} n={len(conf)} per-model " + ",".join(f"{m}:{v:+.5f}" for m, v in pm.items())
    if bad := vetoed(ctx, obs, sig):
        return NodeStatus.VETOED, f"{why}; harm veto on {list(bad)}", e
    ok = e >= 1.0 / ctx.cfg.promotion.alpha_confirm and min(pm.values()) > 0.0
    return (NodeStatus.EVAL_PASSED if ok else NodeStatus.SCREENED_OUT), why, e


class EvalOnly:
    """One tick of the eval-only lane (the driver calls :meth:`tick` before filling slots)."""

    def __init__(self, ctx: LoopContext, store: Any, adapter: Any, book: Any, new_id: Callable[[], str],
                 clock: Callable[[], float], exists: Callable[[Path], bool] = Path.is_file) -> None:
        self.ctx, self.store, self.adapter, self.book = ctx, store, adapter, book
        self.new_id, self.clock, self.exists = new_id, clock, exists

    @property
    def cfg(self) -> Any:
        return self.ctx.cfg.evalonly

    def _state(self) -> dict[str, Any]:
        st = self.store.get_state(K_EO) or {}
        return {"null": st.get("null") or {}, "nodes": st.get("nodes") or {}, "requeued": st.get("requeued") or [],
                "errors": st.get("errors") or {}}

    def _save(self, st: dict[str, Any]) -> None:
        """Persisted after every step: a tick that dies midway (SIGTERM on deploy) keeps what it launched."""
        if self.store.get_state(K_EO) != st:
            self.store.set_state(K_EO, st)

    def _guard(self, what: str, fn: Callable[[], Any], acts: list[str]) -> bool:
        """Run one lane step; any exception is logged, never raised into the driver. False = it raised."""
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001 - one bad step (git, disk, a node) never kills the driver
            self.store.log("evalonly_error", step=what, error=repr(exc)[:500])
            acts.append(f"eval-only {what} failed: {repr(exc)[:300]}")
            return False

    def _set(self, node: Node, status: NodeStatus, reason: str, acts: list[str], **fields: Any) -> Node:
        new = self.store.update_node(node.id, status=status, **fields)
        self.store.log("node_transition", node_id=node.id, frm=node.status.value, to=status.value, reason=reason)
        acts.append(f"node {node.id}: {node.status.value} -> {status.value} ({reason})")
        return new

    def tick(self, snapshot: Any) -> list[str]:
        acts: list[str] = []
        if not self.cfg.enabled or not hasattr(self.adapter, "eval_only"):
            for node in self.store.nodes(status=[NodeStatus.EVALUATING]):    # switched off: full runs
                self.book.cancel(node.id)
                self._set(node, NodeStatus.QUEUED, "eval-only disabled: full run", acts, notes=f"{node.notes} evalonly=off")
            return acts
        st = self._state()
        self._guard("requeue", lambda: self._requeue(st, acts), acts)
        self._guard("route", lambda: self._route(st, acts), acts)
        self._guard("null control", lambda: self._null(st, snapshot, acts), acts)
        self._save(st)
        if st["null"].get("status") == "passed":
            for node in self.store.nodes(status=[NodeStatus.EVALUATING]):
                if not self._guard(f"step {node.id}", lambda: self._step(node, st, snapshot, acts), acts):
                    self._failed(node, st, acts)
                self._save(st)
        elif st["null"].get("status") == "failed":
            for node in self.store.nodes(status=[NodeStatus.EVALUATING]):
                if node.id not in st["nodes"]:
                    self._set(node, NodeStatus.QUEUED, "eval-only blocked (null control failed): full run", acts,
                              notes=f"{node.notes} evalonly=blocked")
        self._save(st)
        return acts

    def _failed(self, node: Node, st: dict[str, Any], acts: list[str]) -> None:
        """Retried next tick; after MAX_ERRORS the node leaves the lane for a normal full run."""
        n = st["errors"][node.id] = int(st["errors"].get(node.id, 0)) + 1
        if n < MAX_ERRORS:
            return
        st["nodes"].pop(node.id, None)
        self.book.cancel(node.id)
        self._set(node, NodeStatus.QUEUED, f"eval-only lane raised {n}x: full run instead", acts,
                  notes=f"{node.notes} evalonly=error")

    # -------------------------------------------------------------- intake
    def _requeue(self, st: dict[str, Any], acts: list[str]) -> None:
        """A readout rejected ``init_incompatible`` for an added buffer was never measured: once, it is
        queued again (the eval-only lane skips the init check; it scores on trained weights)."""
        for node in self.store.nodes(status=[NodeStatus.REJECTED]) if st["null"].get("status") != "failed" else ():
            if node.reject_reason != "init_incompatible" or node.id in st["requeued"] or not node.commit:
                continue
            st["requeued"].append(node.id)
            parent = self.store.get_node(node.parent_id or ROOT_NODE_ID)
            if parent is None or not self.adapter.eval_only(parent.commit, node.commit)[0]:
                continue
            self._set(node, NodeStatus.QUEUED, "re-queued as an eval-only readout (init_incompatible was an "
                      "added buffer)", acts, reject_reason="", notes=f"{node.notes} evalonly_requeue=1")

    def _route(self, st: dict[str, Any], acts: list[str]) -> None:
        """Each queued candidate is checked once (``evalonly=yes|no`` in its notes); an eligible one leaves
        the queue for this lane (unless the null control failed: then it stays a normal candidate). A READOUT-ONLY
        diff (intake's ``fams=`` is the lane's family alone) the lane cannot score is rejected
        ``readout_not_evalonly``, never retrained as a full run."""
        if st["null"].get("status") == "failed":
            return
        for node in self.store.nodes(status=[NodeStatus.QUEUED]):
            if re.search(r"\bevalonly=", node.notes or "") or node.id == ROOT_NODE_ID or design_point(node):
                continue
            parent = self.store.get_node(node.parent_id or ROOT_NODE_ID)
            ok, why, _ = self.adapter.eval_only(parent.commit, node.commit) if parent else (False, "no parent", False)
            if not ok and (m := re.search(r"\bfams=(\S+)", node.notes or "")) and m.group(1) == self.cfg.family:
                self._set(node, NodeStatus.REJECTED, f"readout-only diff the eval-only lane cannot score: {why}"[:300],
                          acts, reject_reason="readout_not_evalonly", notes=f"{node.notes} evalonly=no")
                continue
            if not ok:
                self.store.update_node(node.id, notes=f"{node.notes} evalonly=no")
                continue
            self._set(node, NodeStatus.EVALUATING, "eval-only readout: scored on trained checkpoints", acts,
                      notes=f"{node.notes} evalonly=yes")

    # ------------------------------------------------------------ launches
    def _room(self) -> int:
        cap = max(1, int(self.cfg.max_share * self.ctx.cfg.capacity))
        live = sum(is_eval_run(r) for r in self.book.active())
        return max(0, min(cap - live, self.book.free()))

    def _launch(self, node: Node, stage: Stage, seed: int, plan: Mapping[str, Any], acts: list[str]) -> bool:
        """Launch the planned run if it has no row yet (room permitting); retry an infra void. True = the run
        failed for good (its row is void/crash with no retry left)."""
        run = self.store.get_run(Run.make_id(node.id, stage, seed))
        if run is None:
            if self._room() > 0:
                self.adapter.plan_eval(node.id, stage, seed, plan)
                if (run := self.book.launch(node, stage, seed)) is not None:
                    acts.append(f"eval-only submit {run.id} on {plan['source_run']} -> {run.status.value}")
            return False
        if run.status in (RunStatus.VOID, RunStatus.CRASH):
            if run_failed(run, self.book.infra_retry):
                return True
            if run.status is RunStatus.VOID and self._room() > 0 and self.book.pool(node, run.stage, run.seed):
                acts.append(f"eval-only retry {run.id} -> {self.book.retry(run, node).status.value}")
        return False

    # --------------------------------------------------------- null control
    def _null(self, st: dict[str, Any], snapshot: Any, acts: list[str]) -> None:
        null = st["null"]
        if null.get("status") in ("passed", "failed"):
            return
        if not null:
            pool = models(self.ctx, self.store, snapshot, self.exists)
            chosen = [(m, runs[0]) for m, runs in list(pool.items())[:self.cfg.null_models]]
            if not chosen:
                return
            null.update(status="running", ts=self.clock(), runs={
                Run.make_id(m, Stage.AUDIT, NULL_SEED + k): r.id for k, (m, r) in enumerate(chosen)})
            acts.append(f"eval-only null control started on {[r.id for _, r in chosen]}")
            self.store.log("evalonly_null", status="running", runs=null["runs"])
        results, failed = {}, []
        for rid, src_id in sorted(null["runs"].items()):
            src, node = self.store.get_run(src_id), self.store.get_node(rid.split(":")[0])
            if src is None or node is None:
                failed.append(f"{rid}: source gone")
                continue
            seed = int(rid.rsplit(":", 1)[1])
            if self._launch(node, Stage.AUDIT, seed, {"commit": node.commit, "seed": src.seed, "shim": False,
                                                      "resume": str(Path(src.output_dir) / "latest.pt"),
                                                      "source_run": src.id, "model": node.id}, acts):
                failed.append(f"{rid}: run failed")
            run = self.store.get_run(rid)
            if run is not None and run.status is RunStatus.DONE:
                results[rid] = (None if run.aggregate is None or src.aggregate is None
                                else run.aggregate - src.aggregate)
        if not failed and len(results) < len(null["runs"]):
            return
        bad = failed + [f"{rid}: |d final_score|={'?' if d is None else f'{abs(d):.2e}'}"
                        for rid, d in results.items() if d is None or abs(d) >= self.cfg.null_tol]
        null.update(status="failed" if bad else "passed", deltas=results, why="; ".join(bad), done_ts=self.clock())
        self.store.log("evalonly_null", status=null["status"], deltas=results, why=null["why"])
        acts.append(f"eval-only null control {null['status']}: " + (null["why"] or ", ".join(
            f"{rid} d={d:+.2e}" for rid, d in results.items())))

    # ------------------------------------------------------------- lifecycle
    def _uses(self, st: dict[str, Any]) -> Counter:
        return Counter(r["source"] for e in st["nodes"].values() for r in e.get("runs", {}).values())

    def _plan(self, node: Node, parent: Node, entry: dict[str, Any], phase: str, n: int, snapshot: Any,
              st: dict[str, Any]) -> bool:
        """Add ``n`` checkpoints for ``phase`` (grafting the readout onto each model's code); False when too
        few checkpoints / models exist."""
        pool = models(self.ctx, self.store, snapshot, self.exists)
        skip = {r["source"] for r in entry["runs"].values()}
        commits = entry.setdefault("commits", {})
        for m in list(pool):                       # a model the readout does not apply to is no model
            if m not in commits:
                commits[m] = self.adapter.graft(parent.commit, node.commit, self.store.get_node(m).commit,
                                                f"{node.id}-on-{m}")
            if not commits[m]:
                pool.pop(m)
        used = sorted({r["model"] for r in entry["runs"].values()})
        chosen = pick(pool, n, skip=skip, uses=self._uses(st), min_models=self.cfg.min_models, prefer_new=used) \
            or (pick(pool, n, skip=skip, uses=self._uses(st), min_models=1) if phase == "confirm" else [])
        if not chosen:
            return False
        entry.setdefault("shim", self.adapter.eval_only(parent.commit, node.commit)[2])
        k = len(entry["runs"])
        for i, (m, r) in enumerate(chosen):
            rid = Run.make_id(node.id, _PHASE[phase], EO_SEED + k + i)
            entry["runs"][rid] = {"phase": phase, "model": m, "source": r.id, "commit": commits[m]}
            # already launched (the state holding its plan was lost mid-tick): keep what it actually runs
            if (run := self.store.get_run(rid)) is not None and (old := self._launched(run)):
                entry["runs"][rid].update(model=old["model"], source=old["source_run"], commit=old["commit"])
        return True

    def _launched(self, run: Run) -> dict[str, Any] | None:
        plan = getattr(self.adapter, "eval_plan", lambda _r: None)(run)
        return plan if plan and all(plan.get(k) for k in ("model", "source_run", "commit")) else None

    def _step(self, node: Node, st: dict[str, Any], snapshot: Any, acts: list[str]) -> None:
        parent = self.store.get_node(node.parent_id or ROOT_NODE_ID)
        entry = st["nodes"].setdefault(node.id, {"phase": "screen", "runs": {}})
        if not entry["runs"] and not self._plan(node, parent, entry, "screen", self.cfg.screen_ckpts, snapshot, st):
            st["nodes"].pop(node.id)
            self._set(node, NodeStatus.QUEUED, "eval-only: too few checkpoints / models, full run instead", acts,
                      notes=f"{node.notes} evalonly=nockpt")
            return
        obs, conf, open_ = [], [], False
        for rid, r in entry["runs"].items():
            src, seed = self.store.get_run(r["source"]), int(rid.rsplit(":", 1)[1])
            plan = {"commit": r["commit"], "seed": src.seed if src else 0, "shim": bool(entry.get("shim")),
                    "resume": str(Path(src.output_dir) / "latest.pt") if src else "", "source_run": r["source"],
                    "model": r["model"]}
            if src is None or self._launch(node, _PHASE[r["phase"]], seed, plan, acts):
                run = self.store.get_run(rid)
                crash = run is not None and run.status is RunStatus.CRASH
                vt = run.void_type if run is not None else None
                why = f"eval-only run {rid} " + ("crashed" if crash else f"void ({(vt or VoidType.INFRA).value})")
                self._set(node, NodeStatus.CRASHED if crash else NodeStatus.VOID, why, acts)
                return
            run = self.store.get_run(rid)
            if not _complete(self.ctx, run) or not _complete(self.ctx, src):
                open_ = True
                continue
            d = deltas(self.ctx, run, src)
            obs.append((r["model"], d))
            if r["phase"] == "confirm":
                conf.append(d[PANEL])
        entry["obs"] = {m: round(v, 6) for m, v in per_model(obs).items()} if obs else {}
        if open_:
            return
        sig = sigma(self.ctx, {**{nid: e.get("panel", []) for nid, e in st["nodes"].items() if nid != node.id},
                               node.id: [d[PANEL] for _, d in obs]})
        entry["panel"] = [d[PANEL] for _, d in obs]
        if entry["phase"] == "screen":
            status, why = screen_verdict(self.ctx, obs, sig)
            if status is None:
                entry.update(phase="confirm", sigma=sig, alt=confirm_delta(sig, self.ctx.cfg.promotion.alpha_confirm,
                                                                                  self.cfg.confirm_ckpts))
                if not self._plan(node, parent, entry, "confirm", self.cfg.confirm_ckpts, snapshot, st):
                    self._set(node, NodeStatus.SCREENED_OUT, f"{why}; no fresh checkpoints to confirm on", acts)
                    return
                acts.append(f"eval-only {node.id} passed screening ({why}); confirming on fresh checkpoints")
                self.store.log("evalonly_screen", node_id=node.id, why=why)
                return self._step(node, st, snapshot, acts)        # launch them now
            self._set(node, status, why, acts)
            return
        status, why, e = confirm_verdict(self.ctx, obs, conf, entry["alt"], entry["sigma"])
        entry["e"] = e
        if status is NodeStatus.EVAL_PASSED:
            why += "; " + self._graduate(node, parent, entry, snapshot, acts)
        self._set(node, status, why, acts, confirm_e=float(e), confirm_alt=float(entry["alt"]))

    def _graduate(self, node: Node, parent: Node, entry: dict[str, Any], snapshot: Any, acts: list[str]) -> str:
        """The readout on the incumbent's code, queued as a normal candidate (full retrain)."""
        inc = self.store.get_node(snapshot.incumbent_id if snapshot is not None else ROOT_NODE_ID)
        commit = entry.get("commits", {}).get(inc.id) or self.adapter.graft(
            parent.commit, node.commit, inc.commit, f"{node.id}-on-{inc.id}")
        if not commit:
            return f"not graduated: the readout does not apply on the incumbent {inc.id}"
        nid, now = self.new_id(), self.clock()
        text = self.adapter.diff_commits(inc.commit, commit)
        path = Path(self.ctx.state_dir) / "diffs" / f"{nid}.diff"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        self.store.put_node(Node(
            id=nid, task_id=self.ctx.task_id, commit=commit, parent_id=inc.id, strategy=node.strategy,
            operator=Operator.IMPROVE, family=node.family, mechanism_tag=node.mechanism_tag,
            diff_hash=diff_hash(text), status=NodeStatus.QUEUED, island=node.island, diff_path=str(path),
            hypothesis_id=node.hypothesis_id, levers=node.levers, created_ts=now, updated_ts=now,
            notes=f"evalonly=full evalonly_from={node.id}"))
        if node.hypothesis_id:
            self.store.link(node.hypothesis_id, nid, sign=1, auto=False)
        self.store.log("evalonly_graduated", node_id=node.id, new_node=nid, incumbent=inc.id)
        acts.append(f"eval-only {node.id} generalised: queued {nid} (its readout on incumbent {inc.id}) for a full retrain")
        return f"queued {nid} on {inc.id} for a full retrain"


# ================================================================================================================
# v3-c5 I5: THE V3 RUN-LEVEL LANE. The v3 tick (loop.tick) never runs the driver lane above: it ranks runs, not nodes.
# Here a readout-only candidate C (``adapter.eval_only(parent, C)``: its change lies in the readout family's spans,
# no training-path change) keeps the normal ranking, screen / test / fill path; only HOW a run of C at seed s is
# executed changes: when C's parent P has a finished run at s whose latest.pt is on disk, the run resumes it
# (``adapter.plan_eval``: P's weights at s, C's code, the locked probe.py unmodified; the launch-time lock guard checks
# C's commit). The result is an ordinary run row of C at seed s, so core.data / the posterior / the test pair it with
# P at s like any CRN pair -- y[C,s] - y[P,s] is the readout effect at P's checkpoint s with NO training noise.
#
# NOISE of eval-only pairs: the model's Var d = 2(1-rho)sigma2 must be the across-checkpoint spread of those deltas
# (the readout x checkpoint interaction) -- not the seed sd, which training put in and the pair cancels. It is the
# pooled within-node variance of the eval-only nodes' paired deltas, floored at the calibration's replicate_sd (one
# config re-run end to end: the noise a pair of identical training runs at one seed still has; the old lane's
# sigma_floor 0.0005 ~ replicate_sd 0.000513). It is used (rho_eo = 1 - sd^2 / (2 sigma2), never below the change
# class's own rho) only for nodes ALL of whose screening runs are eval-only, and only once the pooled dfs reach
# ``noise_min_df``; before that, and for the TEST decision (status.pair_fit), the normal near / far fits stand
# (conservative: they claim the training noise the pair does not have).
#
# NULL CONTROL per base epoch: P's own commit on P's checkpoint (stage CALIBRATE at the source seed: a same-seed
# duplicate, averaged by seed_means) must reproduce the source's panel within ``null_tol``; until it passes eligible
# runs wait (<= source_wait_s), if it fails they train normally. HARM GUARD: after ``screen_ckpts`` paired deltas, an
# axis in ``harm_tol`` whose loss exceeds its tolerance with P >= ``harm_prob`` holds the node (no more runs / no test).
# ================================================================================================================

WAIT_MAX_S = 3600.0       # v3-c5b S2: no eligible run is withheld longer than this (for a source / the null control)
NULL_WAIT_S = 1800.0      # ... and at most this while only the null control is pending
ELIG_V = 2                # the eligibility rule's version (2: v3-c5b M3 training-path / buffer-init rule)
K_V3 = "evalonly.v3"     # {"elig": {nid: [ok, why, shim]}, "null": {epoch: {...}}, "harm": {nid: why}}


def _axis(name: str) -> str:
    """A harm_tol key as core.data's short axis name (``slide`` / ``final_probe_slide_mean_auc`` -> ``slide``)."""
    from loop.core.data import PROBE_KEYS
    return next((a for a, k in PROBE_KEYS.items() if name in (a, k)), name)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def floor_sd(ctx: LoopContext) -> float:
    """The pair-sd floor: max(evalonly.sigma_floor, the calibration's replicate_sd when measured)."""
    rep = getattr(getattr(ctx, "noise", None), "replicate_sd", None)
    return max(ctx.cfg.evalonly.sigma_floor, rep if isinstance(rep, (int, float)) and rep > 0 else 0.0)


def paired(by: Mapping[str, Sequence[Any]], child: str, parent: str) -> list[tuple[Any, Any]]:
    """(child obs, parent obs) at the seeds both ran (the parent's same-seed runs averaged by taking the first)."""
    at = {}
    for o in by.get(parent, ()):
        at.setdefault(o.seed, o)
    return [(o, at[o.seed]) for o in by.get(child, ()) if o.seed in at]


def posterior_noise(nz: Any, by: Mapping[str, Sequence[Any]], parents: Mapping[str, str], eo: Iterable[str],
                    weights: Mapping[str, float] | None, floor: float, min_df: int) -> Any:
    """``nz`` (a core.model Noise) with each target's rho raised to the eval-only pairs' own: Var d = the pooled
    within-node variance of the eval-only nodes' paired deltas (>= ``floor``^2) -> rho = 1 - Var d / (2 sigma2), never
    below ``nz``'s, capped .99. None when the pooled dfs are < ``min_df`` (the caller keeps ``nz``)."""
    from loop.core import model as M
    rho = dict(nz.rho)
    for t, s2 in nz.sigma2.items():
        ss = df = 0.0
        for nid in eo:
            if nid not in parents:
                continue
            ds = [c.value(t, weights if t == M.COMPOSITE else None) - p.value(t, weights if t == M.COMPOSITE else None)
                  for c, p in paired(by, nid, parents[nid])]
            if len(ds) >= 2:
                m = statistics.fmean(ds)
                ss, df = ss + sum((d - m) ** 2 for d in ds), df + len(ds) - 1
        if df < min_df:
            return None
        vd = max(ss / df, floor * floor)
        if s2 > 0:
            rho[t] = max(nz.rho.get(t, 0.0), min(0.99, 1.0 - vd / (2.0 * s2)))
    return M.Noise(nz.sigma2, rho, nz.df, nz.resume)


class Lane:
    """The v3 tick's eval-only lane (see the block comment above). The tick calls :meth:`route` on the run specs it
    is about to submit, :meth:`prepare` before the posterior (eval-only noise set + harm guard)."""

    def __init__(self, ctx: LoopContext, store: Any, adapter: Any, clock: Callable[[], float],
                 exists: Callable[[Path], bool] = Path.is_file) -> None:
        self.ctx, self.store, self.adapter, self.clock, self.exists = ctx, store, adapter, clock, exists

    @property
    def cfg(self) -> Any:
        return self.ctx.cfg.evalonly

    @property
    def on(self) -> bool:
        return bool(self.cfg.enabled) and all(hasattr(self.adapter, a) for a in ("eval_only", "plan_eval", "eval_plan"))

    def _state(self) -> dict[str, Any]:
        st = self.store.get_state(K_V3) or {}
        # v3-c5b M3: a cached verdict of an older eligibility rule is re-judged (the training-path rule tightened)
        old = dict(st.get("elig") or {})
        elig = old if st.get("elig_v") == ELIG_V else {}
        # nodes whose runs may carry an eval plan (ever eligible under any rule): eval_runs reads only theirs
        planned = sorted(set(st.get("planned") or ()) | {n for n, v in old.items() if v and v[0]})
        return {**{k: v for k, v in st.items() if k not in ("elig", "null", "harm", "elig_v", "planned", "wait")},
                "wait": dict(st.get("wait") or {}),
                "elig": elig, "elig_v": ELIG_V, "planned": planned, "null": dict(st.get("null") or {}),
                "harm": dict(st.get("harm") or {})}

    def _save(self, st: dict[str, Any]) -> None:
        if self.store.get_state(K_V3) != st:
            self.store.set_state(K_V3, st)

    def plan_of(self, node: str, stage: Any, seed: int) -> dict[str, Any] | None:
        from types import SimpleNamespace
        try:
            return self.adapter.eval_plan(SimpleNamespace(node_id=node, stage=stage, seed=int(seed)))
        except Exception:  # noqa: BLE001 - an unreadable plan is no plan
            return None

    def _clear(self, node: str, stage: Any, seed: int) -> None:
        """A stale plan (an earlier tick routed the spec, it never launched) must not make a training run eval-only."""
        if self.plan_of(node, stage, seed) is not None and hasattr(self.adapter, "clear_eval"):
            self.adapter.clear_eval(node, stage, seed)

    # -------------------------------------------------------------------------------- eligibility
    def eligible(self, st: dict[str, Any], data: Any, nid: str) -> tuple[bool, str, bool]:
        """(eligible, why not, needs a shim), cached per node (commits are immutable)."""
        if nid in st["elig"]:
            ok, why, shim = st["elig"][nid]
            return bool(ok), str(why), bool(shim)
        n = data.nodes.get(nid)
        node = self.store.get_node(nid)
        par = self.store.get_node(n.parent) if n is not None and n.parent else None
        if n is None or node is None or par is None or not node.commit or not par.commit:
            got = (False, "no parent / commit", False)
        elif n.port_of:
            got = (False, "a port (a whole chain)", False)
        else:
            try:
                got = tuple(self.adapter.eval_only(par.commit, node.commit))
            except Exception as exc:  # noqa: BLE001 - git trouble: not eligible (a normal run), retried never
                got = (False, f"eval_only raised: {exc!r}"[:200], False)
        st["elig"][nid] = [bool(got[0]), str(got[1])[:300], bool(got[2])]
        return bool(got[0]), str(got[1]), bool(got[2])

    # -------------------------------------------------------------------------------- sources
    def source(self, data: Any, nid: str, seed: int) -> tuple[str, Any]:
        """('ok', the parent's done run at ``seed`` with complete scores and its latest.pt on disk, never itself an
        eval-only run) / ('wait', its run in flight) / ('none', None)."""
        par = data.nodes[nid].parent
        scored = {o.run_id for o in data.obs if o.node == par and o.seed == seed}
        best, live, by_id = None, None, None
        for r in data.runs:
            if r.node != par or r.seed != seed:
                continue
            if r.status == "done" and r.id in scored:
                if self.plan_of(r.node, r.stage, r.seed) is not None:     # v3-c5b S4: the parent ran eval-only
                    by_id = by_id if by_id is not None else {x.id: x for x in data.runs}
                    r = self._origin(r, by_id)
                if r is not None and r.output_dir and self.exists(Path(r.output_dir) / "latest.pt"):
                    best = best or r
            elif r.status in ("submitting", "pending", "running"):
                live = live or r
        return ("ok", best) if best else (("wait", live) if live else ("none", None))

    def _origin(self, run: Any, by_id: Mapping[str, Any], depth: int = 8) -> Any:
        """The TRAINED run an eval-only run's weights came from (v3-c5b S4): its plan's ``source_run``, followed back
        through eval-only runs (an eval-only incumbent's child resumes the original checkpoint, the same weights its
        parent's run scored); None when the chain breaks (unknown / unfinished run) or exceeds ``depth``."""
        for _ in range(depth):
            plan = self.plan_of(run.node, run.stage, run.seed)
            if plan is None:
                return run if run.status == "done" else None
            run = by_id.get(str(plan.get("source_run") or ""))
            if run is None:
                return None
        return None

    # -------------------------------------------------------------------------------- null control
    def _null(self, st: dict[str, Any], data: Any, want: Sequence[Any], acts: list[str]) -> tuple[str, list[Any]]:
        """(status of this epoch's null control, the null runs to launch now). ``want``: source runs the routing
        would use this tick (the null control runs on the first ``null_models`` of them)."""
        from loop.tick import RunSpec
        e = data.epoch or "-"
        null = st["null"].setdefault(e, {})
        if null.get("status") in ("passed", "failed"):
            return null["status"], []
        if not null.get("runs"):
            chosen = list({r.id: r for r in want}.values())[:max(1, self.cfg.null_models)]
            if not chosen:
                return "none", []
            null.update(status="running", ts=self.clock(), runs={})
            for r in chosen:
                rid = Run.make_id(r.node, Stage.CALIBRATE, r.seed)
                if self.store.get_run(rid) is not None:          # a calibrate run of that model already exists
                    continue
                node = self.store.get_node(r.node)
                self.adapter.plan_eval(r.node, Stage.CALIBRATE, r.seed, {
                    "commit": node.commit, "seed": r.seed, "shim": False, "resume": str(Path(r.output_dir) / "latest.pt"),
                    "source_run": r.id, "model": r.node})
                null["runs"][rid] = r.id
            if not null["runs"]:
                null.update(status="failed", why="no free calibrate run id")
                return "failed", []
            self.store.log("evalonly_null", epoch=e, status="running", runs=null["runs"])
            acts.append(f"eval-only null control (epoch {e[:7]}) on {sorted(null['runs'].values())}")
        launch, bad, done = [], [], {}
        panel = {o.run_id: o.panel for o in data.obs}
        for rid, src in null["runs"].items():
            run = self.store.get_run(rid)
            if run is None:
                node_id, _, seed = rid.rsplit(":", 2)
                launch.append(RunSpec(node_id, Stage.CALIBRATE, int(seed)))
            elif run_failed(run, int(self.ctx.spec.caps.infra_retry)):
                bad.append(f"{rid}: {run.status.value}")
            elif rid in panel and src in panel:
                done[rid] = panel[rid] - panel[src]
        if not bad and len(done) < len(null["runs"]):
            if self.clock() - float(null.get("ts") or 0.0) < 3 * max(self.cfg.source_wait_s, 3600.0):
                return "running", launch
            bad.append("timed out")
        bad += [f"{rid}: |d panel|={abs(d):.2e}" for rid, d in done.items() if abs(d) >= self.cfg.null_tol]
        null.update(status="failed" if bad else "passed", deltas=done, why="; ".join(bad), done_ts=self.clock())
        self.store.log("evalonly_null", epoch=e, status=null["status"], deltas=done, why=null["why"])
        acts.append(f"eval-only null control {null['status']} (epoch {e[:7]})" + (f": {null['why']}" if bad else ""))
        return null["status"], []

    def null_sources(self, data: Any) -> list[Any]:
        """The null control's own sources (v3-c6a, ``evalonly.null_at_start``): one done, scored, trained (no eval
        plan) run with its latest.pt on disk per model of the current epoch -- the latest incumbent first (its port),
        then the base root, then the nodes with the most such runs -- at most ``null_models`` distinct models. Cheap:
        one eval-only run each, so the lane is calibrated before an eligible candidate arrives."""
        scored = {o.run_id for o in data.obs}
        by: dict[str, list[Any]] = {}
        for r in data.runs:
            if r.status != "done" or r.id not in scored or (data.epoch and r.base != data.epoch) \
                    or not r.output_dir or self.plan_of(r.node, r.stage, r.seed) is not None:
                continue
            if not self.exists(Path(r.output_dir) / "latest.pt"):
                continue
            by.setdefault(r.node, []).append(r)
        first = [x for x in ((data.port(data.incumbents[-1][1]) if data.incumbents else None), data.base_root) if x]
        order = [n for n in dict.fromkeys(first) if n in by] + sorted(
            (n for n in by if n not in first), key=lambda n: (-len(by[n]), n))
        return [by[n][0] for n in order[:max(1, int(self.cfg.null_models))]]

    # -------------------------------------------------------------------------------- routing
    def route(self, data: Any, specs: Sequence[Any], acts: list[str]) -> list[Any]:
        """The specs to submit: an eligible candidate's run whose parent's same-seed checkpoint exists is planned
        eval-only (``adapter.plan_eval``); one whose source is still running (or whose epoch's null control is
        running) waits up to ``source_wait_s``; the rest train as usual. Plus the null control's runs."""
        if not self.on:
            return list(specs)
        st, now = self._state(), self.clock()
        out, routed, want = [], [], []
        for s in specs:
            if s.node not in data.nodes or not self.eligible(st, data, s.node)[0]:
                out.append(s)
                continue
            how, src = self.source(data, s.node, s.seed)
            routed.append((s, how, src))
            if how == "ok":
                want.append(src)
        cur = st["null"].get(data.epoch or "-") or {}
        live = cur.get("status") == "running"
        if not want and not live and not cur.get("status") and getattr(self.cfg, "null_at_start", True):
            want = self.null_sources(data)            # v3-c6a: its own trigger, once per epoch, before any candidate
        null, extra = self._null(st, data, want, acts) if want or live else (cur.get("status", "none"), [])
        n_eo, waiting = 0, set()                  # v3-c7a A7: the waiting NODES (one per candidate, not per seed run)
        # v3-c5b S2: a withheld run's wait is measured from when it was FIRST withheld (state "wait"); while the null
        # control is pending it is short (<= NULL_WAIT_S), and no run -- a screen above all -- waits > WAIT_MAX_S
        held, waits = {}, dict(st.get("wait") or {})
        for s, how, src in routed:
            shim = self.eligible(st, data, s.node)[2]
            key = f"{s.node}|{getattr(s.stage, 'value', s.stage)}|{s.seed}"
            waited = now - float(waits.get(key, now))
            cap = min(float(self.cfg.source_wait_s), WAIT_MAX_S)
            if how == "ok" and null == "passed":
                self.adapter.plan_eval(s.node, s.stage, s.seed, {
                    "commit": self.store.get_node(s.node).commit, "seed": s.seed, "shim": shim,
                    "resume": str(Path(src.output_dir) / "latest.pt"), "source_run": src.id,
                    "model": src.node})
                if s.node not in st["planned"]:
                    st["planned"] = sorted({*st["planned"], s.node})
                out.append(s)
                n_eo += 1
            elif how == "ok" and null == "running" and waited < min(cap, NULL_WAIT_S):
                waiting.add(s.node)                               # the null control first (cheap: it goes first)
                held[key] = waits.get(key, now)
            elif how == "wait" and null != "failed" and waited < cap:
                waiting.add(s.node)                               # its source checkpoint is being trained
                held[key] = waits.get(key, now)
            else:
                self._clear(s.node, s.stage, s.seed)
                out.append(s)
        if n_eo or waiting:
            acts.append(f"eval-only lane: {n_eo} run(s) on parent checkpoints, {len(waiting)} node(s) "
                        f"({len(held)} run(s)) waiting for a checkpoint / the null control")
        st["wait"] = held
        self._save(st)
        # S2: the null control's runs FIRST -- one cheap eval-only run each; behind every ranked spec they starved
        return [x for x in extra if self.store.get_run(Run.make_id(x.node, x.stage, x.seed)) is None] + out

    # -------------------------------------------------------------------------------- posterior + harm
    def eval_runs(self, data: Any) -> frozenset[str]:
        """v3-c5b M1: the run ids of ``data.obs`` that ran with an eval plan -- every routed eval-only run (of a node
        ever found eligible) and every null-control run -- whatever ``enabled`` says now (a lane switched off keeps
        its past runs out of the noise fits). Plans are written only for those, so only their obs are probed."""
        if not hasattr(self.adapter, "eval_plan"):
            return frozenset()
        st = self._state()
        nodes = set(st["planned"]) | {nid for nid, v in st["elig"].items() if v and v[0]}
        nulls = {rid for e in st["null"].values() for rid in (e.get("runs") or {})}
        return frozenset(o.run_id for o in data.obs
                         if (o.node in nodes or o.run_id in nulls) and self.plan_of(o.node, o.stage, o.seed))

    def pure(self, data: Any) -> frozenset[str]:
        """Eligible nodes ALL of whose finished runs (>= 1) ran eval-only: the ones the eval-only noise applies to."""
        if not self.on:
            return frozenset()
        st, out = self._state(), set()
        by: dict[str, list[Any]] = {}
        for o in data.obs:
            by.setdefault(o.node, []).append(o)
        for nid, (ok, _, _) in st["elig"].items():
            if ok and by.get(nid) and all(self.plan_of(o.node, o.stage, o.seed) for o in by[nid]):
                out.add(nid)
        return frozenset(out)

    def harm(self, data: Any, pure: Iterable[str], acts: list[str], protect: Iterable[str] = ()) -> dict[str, str]:
        """node -> why, for pure eval-only nodes with >= ``screen_ckpts`` paired deltas whose loss on an axis in
        ``harm_tol`` exceeds its tolerance with P >= ``harm_prob`` (each recorded once: logged, the node VETOED).
        v3-c5b S5: the guard acts BEFORE a node enters TEST -- a ``protect`` node (a member of an open TEST block, the
        incumbent) is neither judged nor held: an open test decides it, an incumbent is not vetoed by its screen."""
        tol = {_axis(a): float(t) for a, t in (self.cfg.harm_tol or {}).items()}
        st = self._state()
        protect = set(protect)
        if not tol:
            return {n: w for n, w in st["harm"].items() if n not in protect}
        by: dict[str, list[Any]] = {}
        for o in data.obs:
            by.setdefault(o.node, []).append(o)
        floor = floor_sd(self.ctx)
        for nid in sorted(set(pure) - set(st["harm"]) - protect):
            pairs = paired(by, nid, data.nodes[nid].parent)
            if len(pairs) < self.cfg.screen_ckpts:
                continue
            bad = []
            for a, t in tol.items():
                ds = [c.axes[a] - p.axes[a] for c, p in pairs if a in c.axes and a in p.axes]
                if len(ds) < 2:
                    continue
                se = max(statistics.stdev(ds), floor) / math.sqrt(len(ds))
                p_harm = _Phi((-t - statistics.fmean(ds)) / se)
                if p_harm >= self.cfg.harm_prob:
                    bad.append(f"{a} {statistics.fmean(ds):+.5f} (tol {t:g}, P={p_harm:.2f})")
            if bad:
                why = f"eval-only harm guard on {len(pairs)} paired checkpoints: " + "; ".join(bad)
                st["harm"][nid] = why
                self.store.log("evalonly_harm", node_id=nid, why=why)
                acts.append(f"node {nid}: held ({why})")
                node = self.store.get_node(nid)
                if node is not None and node.status is not NodeStatus.VETOED:
                    self.store.update_node(nid, status=NodeStatus.VETOED)
                    self.store.log("node_transition", node_id=nid, frm=node.status.value, to="vetoed", reason=why)
        self._save(st)
        return {n: w for n, w in st["harm"].items() if n not in protect}

    def prepare(self, data: Any, acts: list[str], protect: Iterable[str] = (), full: Any = None
                ) -> tuple[frozenset[str], dict[str, str]]:
        """(the pure eval-only nodes, the harm-held nodes) of this tick (``protect``: see :meth:`harm`). ``full``: the
        FULL-panel ledger the harm guard reads (v3-c6r: a guard is not ideator-facing, and the visible rob axis is one
        task); default ``data``."""
        if not self.on:
            return frozenset(), {}
        pure = self.pure(data)
        return pure, self.harm(full if full is not None else data, pure, acts, protect)
