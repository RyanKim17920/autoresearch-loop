"""loop.agents.diagnose — the researcher's ``diagnose`` tool and the per-dataset facts behind it (v3-c4i, 09Z MA).

The science meta-analysis of 10-13 found the classification gap in a few fine-grained non-TCGA sets, per dataset and
per head, which no brief showed: the panel axis is a mean of 12 datasets x 3 heads. Everything here is computed from
run summaries, the config and an external per-dataset table; nothing names an idea.

- :func:`breakdown`     per (dataset, head): the incumbent's mean over its done runs vs the best other node of the base
                        epoch (>= ``deficit_min_seeds`` runs) vs the best off-the-shelf public model of
                        ``ideation.external_task_csv`` (rows matching ``external_task_models``; THUNDER's results.csv)
- :func:`dataset_gaps`  per dataset: the largest public lead over the incumbent across the heads (levers section (h))
- :func:`features`      a short SLURM job (primary pool) running ``loop/adapters/feature_diag.py`` on the incumbent's
                        checkpoint: per classification dataset the probe-feature / cls norms and every per-tile sigmoid
                        gate inside ``probe_features``; cached per incumbent, bounded by ``diagnose_wait_s``
- :func:`databank`      the data bank's datasets (manifest lines) or one dataset's files and manifest (read-only)
- :class:`Diagnoser`    the tool: ``diagnose(what=classification | features | databank, name='')``
"""

from __future__ import annotations

import csv
import json
import math
import re
import shlex
import shutil
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

K_CLS = "diagnose.cls"          # {incumbent, rows, ts}: the breakdown of the last view (ideation writes it)
K_INC = "diagnose.incumbent"    # {node, commit, runs: [output_dir, ...]} (ideation writes it each tick)
K_JOBS = "diagnose.jobs"        # {incumbent: {job, ts, out}}: feature-diagnostic jobs submitted
SUBDIR = "diagnose"
ERR_NAME = "features.error.json"      # the job's failure report (feature_diag.py): shown once, never cached
JOB_SCRIPT = Path(__file__).resolve().parents[1] / "adapters" / "feature_diag.py"
POLL_S = 15.0
ALIVE_TRIES, ALIVE_RETRY_S = 3, 2.0     # a failed squeue is retried; still failing: UNKNOWN = alive
_SUBMIT_LOCK = threading.Lock()          # check-and-submit: concurrent sessions never submit the job twice
TOOL_DEF = {"name": "diagnose", "description": (
    "Read-only diagnostics of the INCUMBENT (cached per incumbent). what='classification': per dataset and per probe "
    "head (linear / knn / fewshot) its score vs the best other node and the best public model; what='features': a "
    "short GPU job running its probe_features on a few tiles per classification dataset -- feature norms and every "
    "per-tile gate inside probe_features, per dataset (blocks a few minutes, then answers pending: call again later); "
    "what='databank': the data bank's datasets (name=<dataset> lists its files and manifest)."),
    "input_schema": {"type": "object", "required": ["what"], "properties": {
        "what": {"type": "string", "enum": ["classification", "features", "databank"]}, "name": {"type": "string"}}}}


def opt(ctx: Any, name: str) -> Any:
    from loop.core.levers import opt as o
    return o(ctx, name)


# ---------------------------------------------------------------------------------------------- run summaries
_SUMMARY: dict[str, dict[str, Any]] = {}


def summary(output_dir: str) -> dict[str, Any]:
    """A done run's ``summary.json`` ({} when absent / unreadable); immutable once written, so cached by path."""
    if not output_dir:
        return {}
    if (got := _SUMMARY.get(output_dir)) is not None:
        return got
    try:
        got = json.loads((Path(output_dir) / "summary.json").read_text())
    except (OSError, ValueError):
        return {}
    _SUMMARY[output_dir] = got if isinstance(got, dict) else {}
    return _SUMMARY[output_dir]


def key_re(ctx: Any) -> re.Pattern[str]:
    """``task_metric_key`` as a regex with named groups ``dataset`` / ``head`` (heads: ``external_task_heads``)."""
    heads = "|".join(re.escape(h) for h in sorted(dict(opt(ctx, "external_task_heads") or {}), key=len, reverse=True))
    pat = re.escape(str(opt(ctx, "task_metric_key")))
    pat = pat.replace(re.escape("{dataset}"), r"(?P<dataset>.+?)").replace(re.escape("{head}"), f"(?P<head>{heads})")
    return re.compile(f"^{pat}$")


def task_scores(ctx: Any, summ: Mapping[str, Any], rx: re.Pattern[str] | None = None) -> dict[tuple[str, str], float]:
    rx = rx or key_re(ctx)
    out = {}
    for k, v in summ.items():
        if (m := rx.match(k)) and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[(m["dataset"], m["head"])] = float(v)
    return out


def node_scores(ctx: Any, led: Any, node: str, rx: re.Pattern[str] | None = None) -> tuple[dict, int]:
    """({(dataset, head): [value per done run]}, done runs read)."""
    rx = rx or key_re(ctx)
    out: dict[tuple[str, str], list[float]] = {}
    n = 0
    for r in led.runs:
        if r.node == node and r.status == "done" and (s := summary(r.output_dir)):
            n += 1
            for k, v in task_scores(ctx, s, rx).items():
                out.setdefault(k, []).append(v)
    return out, n


# ---------------------------------------------------------------------------------------------- external table
_EXT: dict[tuple[str, float], dict] = {}


def external(ctx: Any) -> dict[tuple[str, str], tuple[float, str]]:
    """{(OUR dataset name, head): (best value, model)} over the table's off-the-shelf rows; {} without a table."""
    path = str(opt(ctx, "external_task_csv") or "")
    if not path:
        return {}
    p = Path(path)
    if not p.is_absolute():
        p = Path(getattr(getattr(ctx, "spec", None), "repo", "") or ".") / p
    try:
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return {}
    if key in _EXT:
        return _EXT[key]
    models = re.compile(str(opt(ctx, "external_task_models") or "."), re.I)
    excl = [re.compile(x, re.I) for x in opt(ctx, "external_exclude") or ()]
    heads = {v: k for k, v in dict(opt(ctx, "external_task_heads") or {}).items()}
    theirs = {v: k for k, v in dict(opt(ctx, "external_task_names") or {}).items()}
    metric, scale = str(opt(ctx, "external_task_metric")), float(opt(ctx, "external_task_scale"))
    out: dict[tuple[str, str], tuple[float, str]] = {}
    try:
        with p.open(newline="") as f:
            for r in csv.DictReader(f):
                m = str(r.get("model", ""))
                if r.get("metric") != metric or r.get("task") not in heads or not models.search(m) \
                        or any(x.search(m) for x in excl):
                    continue
                try:
                    x = float(r["metric_score"]) * scale
                except (KeyError, TypeError, ValueError):
                    continue
                k = (theirs.get(r.get("dataset", ""), r.get("dataset", "")), heads[r["task"]])
                if k not in out or x > out[k][0]:
                    out[k] = (x, m)
    except OSError:
        return {}
    _EXT.clear()
    _EXT[key] = out
    return out


# ---------------------------------------------------------------------------------------------- the breakdown
def held_names(store: Any) -> frozenset[str]:
    """The blind-holdout task names (v3-c6a, state ``holdout.tasks`` = "axis:task"), any axis: none of them may reach
    an ideator-facing breakdown / feature table."""
    try:
        got = store.get_state("holdout.tasks") or ()
    except Exception:  # noqa: BLE001 - a store without state
        return frozenset()
    return frozenset(str(x).partition(":")[2] or str(x) for x in got)


def unheld(rows: Sequence[Mapping[str, Any]], store: Any) -> list[Mapping[str, Any]]:
    """``rows`` (a cached breakdown) without the held-out datasets (a row cached before the holdout was frozen)."""
    held = held_names(store)
    return [r for r in rows if r.get("dataset") not in held]


def breakdown(ctx: Any, led: Any, incumbent: str, held: Any = ()) -> list[dict[str, Any]]:
    """Per (dataset, head) of the incumbent's done runs: its mean (n), the best other node of the base epoch with >=
    ``deficit_min_seeds`` done runs, the best public model. Datasets in the summary's order, heads as configured."""
    from loop.core.acquire import wave_node
    rx = key_re(ctx)
    held = frozenset(held)
    mine, n_inc = node_scores(ctx, led, incumbent, rx)
    mine = {k: x for k, x in mine.items() if k[0] not in held}      # c6a: the held-out dataset is never shown
    if not mine:
        return []
    mn = int(opt(ctx, "deficit_min_seeds"))
    best: dict[tuple[str, str], tuple[float, str, int]] = {}
    done: dict[str, int] = {}
    for r in led.runs:
        if r.status == "done":
            done[r.node] = done.get(r.node, 0) + 1
    for nid, k in done.items():
        node = led.nodes.get(nid)
        if nid == incumbent or k < mn or node is None or node.base != led.epoch or wave_node(node):
            continue
        sc, _ = node_scores(ctx, led, nid, rx)
        for key, xs in sc.items():
            if key[0] not in held and len(xs) >= mn and (key not in best or statistics.fmean(xs) > best[key][0]):
                best[key] = (statistics.fmean(xs), nid, len(xs))
    ext = external(ctx)
    order = list(dict(opt(ctx, "external_task_heads") or {}))
    rows = []
    for (ds, head), xs in sorted(mine.items(), key=lambda kv: (kv[0][0], order.index(kv[0][1])
                                                                if kv[0][1] in order else 9)):
        b, e = best.get((ds, head)), ext.get((ds, head))
        rows.append({"dataset": ds, "head": head, "ours": statistics.fmean(xs), "n": len(xs),
                     "sd": statistics.stdev(xs) if len(xs) > 1 else None,
                     "best": [b[1], b[0], b[2]] if b else None, "ext": [e[1], e[0]] if e else None})
    return rows


def dataset_gaps(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per dataset: the head with the largest public lead over the incumbent (ext - ours), the best other node's lead
    on that head, sorted by the public lead."""
    per: dict[str, dict[str, Any]] = {}
    for r in rows:
        if not r.get("ext"):
            continue
        gap = r["ext"][1] - r["ours"]
        if r["dataset"] not in per or gap > per[r["dataset"]]["gap"]:
            per[r["dataset"]] = {"dataset": r["dataset"], "head": r["head"], "gap": gap, "ours": r["ours"],
                                 "ext": r["ext"], "node_lead": (r["best"][1] - r["ours"], r["best"][0])
                                 if r.get("best") else None}
    return sorted(per.values(), key=lambda d: -d["gap"])


def breakdown_text(rows: Sequence[Mapping[str, Any]], incumbent: str) -> str:
    if not rows:
        return f"(no per-dataset scores in the run summaries of {incumbent})"
    f = lambda x: "-" if x is None else f"{x:.4f}"     # noqa: E731
    lines = [f"dataset/head | incumbent {incumbent} (runs, sd) | best other node (runs) | best public model"]
    for r in rows:
        b, e = r.get("best"), r.get("ext")
        lines.append(f"{r['dataset']}/{r['head']} | {f(r['ours'])} ({r['n']}, sd {f(r['sd'])}) | "
                     + (f"{f(b[1])} ({b[1] - r['ours']:+.4f}) {b[0]} ({b[2]})" if b else "-") + " | "
                     + (f"{f(e[1])} ({e[1] - r['ours']:+.4f}) {e[0]}" if e else "-"))
    heads: dict[str, list[float]] = {}
    for r in rows:
        if r.get("ext"):
            heads.setdefault(r["head"], []).append(r["ext"][1] - r["ours"])
    if heads:
        lines.append("mean public lead per head: " + ", ".join(f"{h} {statistics.fmean(x):+.4f} over {len(x)} datasets"
                                                              for h, x in heads.items()))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- the data bank
def databank(ctx: Any, store: Any, name: str = "") -> str:
    from loop import databank as DB
    db = getattr(getattr(ctx, "cfg", None), "databank", None)
    if db is None or not getattr(db, "enabled", False) or not getattr(db, "root", ""):
        return "no data bank configured"
    root, cap = Path(db.root), int(opt(ctx, "databank_list_max"))
    if name:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name) or not (root / name).is_dir():
            return f"no dataset {name!r} under {root}"
        d = root / name
        files = sorted(d.iterdir(), key=lambda p: p.name)
        lines = [f"{d}: {len(files)} entries"] + [f"- {p.name}{'/' if p.is_dir() else ''}"
                                                  + ("" if p.is_dir() else f" ({p.stat().st_size:,} B)")
                                                  for p in files[:cap]]
        for meta in ("manifest.json", "spec.json", "status.json"):
            try:
                text = (d / meta).read_text()
            except OSError:
                continue
            lines.append(f"## {meta}\n{text[:4000]}")
        return "\n".join(lines)
    try:
        names = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError as exc:
        return f"cannot list {root}: {exc}"
    reg = DB.book(store)["registered"] if hasattr(store, "get_state") else {}
    lines = [f"{root}: {len(names)} dataset dir(s); registered in the bank: {len(reg)}"]
    for n in names[:cap]:
        try:
            s = reg.get(n) or DB.summary(ctx, n)
            lines.append(f"- {n} [{DB.status(ctx, n)}]: {DB.line(ctx, s)}")
        except Exception as exc:  # noqa: BLE001 - one bad manifest never hides the rest
            lines.append(f"- {n}: (manifest unreadable: {type(exc).__name__})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- the feature job
def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def features_text(rep: Mapping[str, Any], held: Any = ()) -> str:
    if rep.get("error"):
        return f"feature diagnostic FAILED: {rep['error']}"
    f = lambda d, k="mean": "-" if not d or d.get(k) is None else f"{d[k]:.3g}"     # noqa: E731
    lines = [f"feature diagnostic of {rep.get('incumbent', '?')} ({rep.get('n_tiles')} val tiles per dataset): "
             "probe_features norm | its head (cls width) norm | plain-forward cls norm | cos(head, cls) | per-tile "
             "gates inside probe_features (mean / p10 / p50 / p90)"]
    for ds, d in sorted((rep.get("datasets") or {}).items()):
        if ds in held:
            continue
        gates = "; ".join(f"{f(g)} / {f(g, 'p10')} / {f(g, 'p50')} / {f(g, 'p90')}" for g in d.get("gates") or ()) \
            or "no gate"
        lines.append(f"- {ds} ({d.get('tiles')} tiles): {f(d.get('norm_probe_features'))} | "
                     f"{f(d.get('norm_probe_features_head'))} | {f(d.get('norm_cls'))} | "
                     f"{f(d.get('cos_head_vs_cls'))} | {gates}")
    return "\n".join(lines)


def _primary(ctx: Any) -> str:
    pools = list(getattr(ctx.cfg, "pools", ()) or ())
    return next((p.name for p in pools if not p.preemptible), pools[0].name if pools else "")


def features(ctx: Any, store: Any, adapter: Any, *, wait_s: float | None = None, sleep: Any = time.sleep,
             clock: Any = time.time) -> str:
    """The feature diagnostic of the incumbent named in :data:`K_INC`: the cached report, else submit the job (once
    per incumbent while one is alive) and wait up to ``wait_s`` (``ideation.diagnose_wait_s``) for it."""
    inc = store.get_state(K_INC) or {}
    node, commit, runs = inc.get("node"), inc.get("commit"), list(inc.get("runs") or ())
    if not node or not commit:
        return "no incumbent known yet (the loop records it each tick)"
    out_dir = Path(ctx.state_dir) / SUBDIR / _safe(node)
    res = out_dir / "features.json"
    if res.is_file():
        try:
            rep = json.loads(res.read_text())
        except ValueError:
            rep = None
        if isinstance(rep, Mapping) and not rep.get("error"):
            return features_text({"incumbent": node, **rep}, held_names(store))
        res.unlink(missing_ok=True)       # an error / unreadable report is never cached: the next call resubmits
    need = ("_node_worktree", "_sbatch", "runner")
    if any(not hasattr(adapter, a) for a in need):
        return "the feature diagnostic needs the cluster adapter (not available here)"
    with _SUBMIT_LOCK:
        got = _submit(ctx, store, adapter, node, commit, runs, out_dir, res, clock)
    if isinstance(got, str):
        return got
    job = got
    deadline = clock() + (float(opt(ctx, "diagnose_wait_s")) if wait_s is None else wait_s)
    while not res.is_file() and clock() < deadline and _alive(adapter, job.get("job", "")):
        sleep(POLL_S)
    err = out_dir / ERR_NAME
    if res.is_file():
        try:
            return features_text({"incumbent": node, **json.loads(res.read_text())}, held_names(store))
        except ValueError:
            return "feature diagnostic wrote an unreadable report"
    if not _alive(adapter, job.get("job", "")):
        if err.is_file():                 # the job's failure report: shown, never cached (a later call resubmits)
            try:
                return features_text(json.loads(err.read_text()))
            except ValueError:
                pass
        return f"feature diagnostic job {job.get('job')} ended without a report (see {out_dir})"
    return (f"feature diagnostic job {job.get('job')} is still pending / running; call diagnose(what='features') "
            "again later (the result is cached per incumbent)")


def _submit(ctx: Any, store: Any, adapter: Any, node: str, commit: str, runs: Sequence[str], out_dir: Path,
            res: Path, clock: Any) -> dict[str, Any] | str:
    """Under :data:`_SUBMIT_LOCK`: the live job of ``node`` ({job, ts, out}), else submit one; a str = why not."""
    jobs = dict(store.get_state(K_JOBS) or {})
    job = jobs.get(node) or {}
    alive = bool(job.get("job")) and _alive(adapter, job["job"])
    if not alive and job.get("job") and (err := out_dir / ERR_NAME).is_file():
        try:                              # the last job failed: say so ONCE (never cached), the next call resubmits
            rep = json.loads(err.read_text())
        except ValueError:
            rep = {"error": "unreadable failure report"}
        err.unlink(missing_ok=True)
        return features_text({"error": "unknown", **rep} if isinstance(rep, dict) else {"error": str(rep)})
    if not alive and not res.is_file():
        ckpt = next((Path(d) / "latest.pt" for d in runs if (Path(d) / "latest.pt").is_file()), None)
        if ckpt is None:
            return f"no checkpoint of {node} on disk (its runs' latest.pt were pruned)"
        cfgp = next((summary(d).get("config_path") for d in runs if summary(d).get("config_path")
                     and Path(summary(d)["config_path"]).is_file()), "")
        try:
            wt = adapter._node_worktree(node, commit)
        except Exception as exc:  # noqa: BLE001
            return f"cannot prepare the incumbent's worktree: {exc}"[:300]
        cfgp = cfgp or str(Path(wt) / getattr(adapter, "config_rel", "configs/main.yaml"))
        out_dir.mkdir(parents=True, exist_ok=True)
        script = out_dir / JOB_SCRIPT.name
        shutil.copyfile(JOB_SCRIPT, script)
        from loop.adapters.nanopath_v2 import LAUNCHER
        launcher = getattr(adapter, "LAUNCHER", LAUNCHER)       # a project adapter's own job script (nanobrain)
        try:
            heads = [l for l in (Path(wt) / launcher).read_text().splitlines() if l.startswith("#SBATCH")]
        except OSError:
            heads = []
        (Path(wt) / "slurm").mkdir(exist_ok=True)
        q = shlex.quote
        sb = out_dir / "feature_diag.sbatch"
        sb.write_text("\n".join([
            "#!/usr/bin/env bash", *heads, "# loop feature diagnostic (loop.agents.diagnose): read-only, no training",
            "set -uo pipefail", f"cd {q(str(wt))}",
            f'"$PWD/.venv/bin/python" {q(str(script))} "$PWD" {q(str(cfgp))} {q(str(ckpt))} {q(str(res))} '
            f"--n={int(opt(ctx, 'diagnose_tiles'))}", ""]))
        sb.chmod(0o755)
        _, env = adapter._sbatch(cfgp, out_dir, f"diag-{_safe(node)}", pool=_primary(ctx))
        env.pop("SBATCH_REQUEUE", None)
        env.update(WANDB_MODE="disabled", SBATCH_TIMELIMIT=str(int(opt(ctx, "diagnose_job_min"))),
                   SBATCH_JOB_NAME=f"diag-{_safe(node)}")
        rc, out, err = adapter.runner.run(["sbatch", str(sb)], cwd=str(wt), timeout=300, env=env)
        m = re.search(r"Submitted batch job (\d+)", out or "")
        if rc != 0 or not m:
            return f"feature diagnostic not submitted (rc={rc}): {(err or out or '').strip()[-300:]}"
        (out_dir / ERR_NAME).unlink(missing_ok=True)
        job = jobs[node] = {"job": m.group(1), "ts": clock(), "out": str(res)}
        store.set_state(K_JOBS, jobs)
        store.log("diagnose_submitted", node_id=node, job=job["job"])
    return job


def _alive(adapter: Any, jid: str, *, sleep: Any = None) -> bool:
    """squeue says the job is queued / running. A FAILED squeue (rc != 0 other than an unknown job id, a timeout, an
    exception) is retried ``ALIVE_TRIES`` times; still failing it counts as alive (unknown is never dead: a dead
    verdict resubmits the job)."""
    if not jid:
        return False
    for k in range(ALIVE_TRIES):
        try:
            rc, q, err = adapter.runner.run(["squeue", "-h", "-j", jid, "-o", "%T"], timeout=60)
        except Exception:  # noqa: BLE001
            rc, q, err = -1, "", "exception"
        if rc == 0:
            return bool(q.strip())
        if re.search(r"invalid job id", f"{q} {err}", re.I):      # squeue's own verdict: the job is gone
            return False
        if k + 1 < ALIVE_TRIES:
            (sleep or time.sleep)(ALIVE_RETRY_S)
    return True


# ---------------------------------------------------------------------------------------------- the tool
class Diagnoser:
    """``diagnose(what, name)`` for one researcher session: classification from :data:`K_CLS` (the tick's view; the
    incumbent's own runs + the external table when no view wrote it yet), features (the job), databank."""

    def __init__(self, ctx: Any, store: Any, adapter: Any) -> None:
        self.ctx, self.store, self.adapter = ctx, store, adapter

    def __call__(self, what: str, name: str = "") -> str:
        if what == "databank":
            return databank(self.ctx, self.store, name)
        if what == "features":
            return features(self.ctx, self.store, self.adapter)
        if what == "classification":
            got = self.store.get_state(K_CLS) or {}
            inc = (self.store.get_state(K_INC) or {}).get("node") or got.get("incumbent", "")
            if isinstance(got.get("rows"), list) and got.get("incumbent") == inc:    # [] = computed, no scores
                return breakdown_text(unheld(got["rows"], self.store), inc)
            return f"(no per-dataset breakdown of {inc or 'the incumbent'} computed yet: the next tick writes it)"
        return f"ERROR: unknown what={what!r}; use classification, features or databank"


def record(ctx: Any, store: Any, led: Any, incumbent: str, commit: str, rows: Sequence[Mapping[str, Any]] | None
           ) -> None:
    """What the tool reads: the incumbent (node, commit, its done runs' dirs) and the view's breakdown. ``rows``
    None (ideation's every-tick call): the breakdown is PRECOMPUTED here when the cached one is of another incumbent
    or of fewer of its done runs (v3-c5 I2: at startup and on an incumbent switch the first diagnose call of a session
    found nothing until a levers pass recomputed it), so a session never waits on it."""
    runs = [r.output_dir for r in led.runs if r.node == incumbent and r.status == "done" and r.output_dir]
    inc = {"node": incumbent, "commit": commit, "runs": runs}
    if store.get_state(K_INC) != inc:
        store.set_state(K_INC, inc)
    cur = store.get_state(K_CLS) or {}
    if rows is None and (cur.get("incumbent") != incumbent or cur.get("n_runs") != len(runs)
                         or not isinstance(cur.get("rows"), list)):
        from loop.core import levers as LV
        try:
            LV.summaries(ctx, runs)      # time-boxed, disk-cached: warms the breakdown's reads
            rows = breakdown(ctx, led, incumbent, held_names(store))
        except Exception as exc:  # noqa: BLE001 - the tool then says "not computed yet"; the levers pass retries
            if hasattr(store, "log"):
                store.log("diagnose_error", error=f"breakdown precompute: {exc!r}"[:300])
    if rows is not None:
        if cur.get("incumbent") != incumbent or cur.get("rows") != list(rows) or cur.get("n_runs") != len(runs):
            store.set_state(K_CLS, {"incumbent": incumbent, "rows": list(rows), "n_runs": len(runs),
                                    "ts": time.time()})


# ---------------------------------------------------------------------------------------------- precompute (v3-c5)
K_PRE = "diagnose.precomputed"   # {node}: the incumbent whose feature job was last precomputed (submitted / cached)


#: v3-c5b S8: the precompute job in flight per store (one per process / store), and the failure backoff (s)
_PRE_JOBS: dict[int, dict[str, Any]] = {}
PRE_BACKOFF_S = (120.0, 3600.0)          # first retry after this, doubling per failure, capped


def precompute_features(ctx: Any, store: Any, adapter: Any, executor: Any = None, clock: Any = time.time) -> str:
    """Submit the feature diagnostic of the incumbent named in :data:`K_INC` WITHOUT waiting, once per incumbent
    (at startup and on an incumbent switch; ``ideation.diagnose_precompute``), so a session's first
    ``diagnose(what='features')`` finds the report or a running job. '' when nothing was done, else what was.
    v3-c5b S8: marked done (:data:`K_PRE` ok) only on success (a submitted / live job or a cached report); a failure
    (no checkpoint yet, a failed sbatch / worktree, the last job's failure report) is retried with a doubling backoff
    (``PRE_BACKOFF_S``); with ``executor`` the worktree + sbatch run OFF the tick thread (one job in flight, its
    outcome recorded by the tick that finds it done)."""
    inc = store.get_state(K_INC) or {}
    node, commit, runs = inc.get("node"), inc.get("commit"), list(inc.get("runs") or ())
    key = id(store)
    if (job := _PRE_JOBS.get(key)) is not None:              # the off-thread submit: harvested once done
        if not job["fut"].done():
            return ""
        _PRE_JOBS.pop(key, None)
        try:
            got = job["fut"].result()
        except Exception as exc:  # noqa: BLE001 - a raising submit is a failure (retried with backoff)
            got = f"precompute raised: {exc!r}"[:300]
        return _pre_record(store, job["node"], got, clock)
    if not opt(ctx, "diagnose_precompute") or not node or not commit or adapter is None:
        return ""
    try:                                                      # v3-c6a: owed TEST runs hold primary first
        from loop.runner import K_TEST_RESERVE
        if int(store.get_state(K_TEST_RESERVE) or 0) > 0:
            return ""
    except Exception:  # noqa: BLE001 - no reserve state: not deferred
        pass
    st = store.get_state(K_PRE) or {}
    if st.get("node") == node and st.get("ok"):
        return ""
    if any(not hasattr(adapter, a) for a in ("_node_worktree", "_sbatch", "runner")):
        return ""
    out_dir = Path(ctx.state_dir) / SUBDIR / _safe(node)
    res = out_dir / "features.json"
    if res.is_file():                                          # a report on disk (a session's job): done at once
        return _pre_record(store, node, "cached", clock)
    if st.get("node") == node and clock() < float(st.get("next") or 0.0):
        return ""                                              # backing off after a failure

    def submit() -> Any:
        if res.is_file():
            return "cached"
        with _SUBMIT_LOCK:
            return _submit(ctx, store, adapter, node, commit, runs, out_dir, res, time.time)
    if executor is None:
        try:
            got = submit()
        except Exception as exc:  # noqa: BLE001 - recorded as a failure, retried with backoff
            got = f"precompute raised: {exc!r}"[:300]
        return _pre_record(store, node, got, clock)
    _PRE_JOBS[key] = {"node": node, "fut": executor.submit(submit)}
    return f"feature diagnostic of {node}: precompute started off the tick"


def _pre_record(store: Any, node: str, got: Any, clock: Any) -> str:
    """Record a precompute outcome: ok (a job dict / 'cached') -> done for ``node``; else a failure with backoff."""
    ok = got == "cached" or isinstance(got, Mapping)
    msg = got if isinstance(got, str) else f"job {got.get('job')}"
    if ok:
        store.set_state(K_PRE, {"node": node, "ts": clock(), "ok": True})
        return f"feature diagnostic of {node}: {msg}"[:300]
    st = store.get_state(K_PRE) or {}
    fails = int(st.get("fails") or 0) + 1 if st.get("node") == node else 1
    wait_s = min(PRE_BACKOFF_S[1], PRE_BACKOFF_S[0] * 2 ** (fails - 1))
    store.set_state(K_PRE, {"node": node, "ts": clock(), "ok": False, "fails": fails, "next": clock() + wait_s,
                            "why": str(msg)[:300]})
    if hasattr(store, "log"):
        store.log("diagnose_precompute_failed", node_id=node, fails=fails, retry_s=wait_s, why=str(msg)[:300])
    return f"feature diagnostic of {node} not precomputed (retry in {wait_s:.0f} s): {msg}"[:300]


def cached_features(ctx: Any, store: Any, incumbent: str | None = None) -> dict[str, dict[str, Any]] | None:
    """The cached per-dataset feature diagnostic of ``incumbent`` (default the one in :data:`K_INC`): {dataset:
    {tiles, norm_probe_features, norm_probe_features_head, norm_cls, cos_head_vs_cls, gates: [{mean, p10, p50,
    p90}, ...], ...}} as ``feature_diag.py`` wrote it; None when no successful report is on disk."""
    node = incumbent or (store.get_state(K_INC) or {}).get("node")
    if not node:
        return None
    res = Path(ctx.state_dir) / SUBDIR / _safe(node) / "features.json"
    try:
        rep = json.loads(res.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(rep, Mapping) or rep.get("error") or not isinstance(rep.get("datasets"), Mapping):
        return None
    held = held_names(store)
    return {str(k): dict(v) for k, v in rep["datasets"].items() if isinstance(v, Mapping) and str(k) not in held}

