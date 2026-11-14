"""loop.localruns — the operator's MANUAL RUNS as external evidence (``loop.local_runs``; default off).

On 2026-11-08 the operator ran ~40 nanobrain experiments by hand (job tables ``<home>/*_jobs.tsv``: name, jobid,
worktree, branch / sha, overrides, lane), each in its own worktree's ``output/<name>``. Without this module the loop
never sees them, so ideation re-proposes them. Here:

- :func:`import_runs` (idempotent; the tick runs it every ``local_runs.every_hours`` and ``loop import-runs`` on
  demand): every job-table row whose output dir changed since the last import (a fingerprint of its files) is read:
  ``summary.json`` (steps / images / wall / git), ``config.yaml`` (the EFFECTIVE config: its delta against the root's
  config is the run's config change), ``REQUEUED``, and the per-task jsons, RE-SCORED with the adapter's scorer
  (``local_runs.scorer``) and cross-checked against ``eval/aggregate.json``. The code change is ``git diff <root>
  <run commit> -- '*.py'`` in the task repo; the run's lineage is the first ``local_runs.lineages`` commit that is
  its ancestor (a lineage in ``incomparable`` is on another base: absolute scores only). A FINISHED run (training
  done, every task scored) is stored in the ``local_runs`` table -- NEVER as a ledger node / run: these runs used
  the operator's seeds (7338 / 1), not the loop's CRN seeds, so they must not enter any paired test.
- :func:`compute`: per-axis deltas against the same-seed root replicate (``local_runs.reference_runs``), in units of
  the loop's measured paired sd (a run on a PREVIOUS root's lineage, ``local_runs.context_refs``, is diffed against
  THAT lineage's own same-seed replicate instead: context, never a delta against the current root, never banked); runs with the same change (config delta minus seed + code diff) are grouped; a
  group is PROMISING (an axis >= ``promising_axis_sd`` with the panel not below ``-promising_open_sd``, or the panel
  >= ``promising_open_sd``) or HARMFUL (panel <= ``-harmful_open_sd``).
- :func:`block`: the brief section every v3 brief / the root audit carries; :func:`digest_lines` the digest's.
- a promising group is banked ONCE as a priority confirm idea (``idea_bank_local_runs.jsonl`` in the state dir,
  served by ``ideation.bank_v3``): re-implement exactly that change and test it under the loop's own seeds.
- the one exception to "never a ledger run" (:func:`register_root_replicates`, ``local_runs.root_replicates``): a
  full run IDENTICAL to the root (its commit has no ``.py`` diff vs the root commit and its effective config differs
  only in seed / name / output / logging keys) IS a root seed replicate -- the same experiment the loop's own root
  baseline would run -- so it is registered as a done ``root:baseline:<seed>`` run (11-08: a fresh root's noise fit
  needs >= 2 root seeds; a root replicate run / a root replicate run / a root replicate run are exactly that, paid for already).
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

K_DIGEST = "local_runs.digest"       # {ts, n, n_full, n_smoke, groups, promising, harmful, text, summary}
K_BOOK = "local_runs.book"           # {last_ts, last_ok, error, imports}
BANK = "idea_bank_local_runs.jsonl"
HEADING = "## Already measured locally"
AGG_TOL = 1e-6
DIFF_CAP = 20000


def cfg_of(ctx: Any) -> Any:
    return getattr(getattr(ctx, "cfg", None), "local_runs", None)


def enabled(ctx: Any) -> bool:
    return bool(getattr(cfg_of(ctx), "enabled", False))


def banks(ctx: Any) -> bool:
    return enabled(ctx) and bool(getattr(cfg_of(ctx), "bank", True))


def bank_path(ctx: Any) -> str:
    return str(Path(getattr(ctx, "state_dir", "") or ".") / BANK)


# ------------------------------------------------------------------ inputs
#: the columns of a job table WITHOUT a header row (wave3_jobs.tsv)
COLUMNS = ("name", "jobid", "worktree", "branch", "sha", "account", "overrides", "lane")


def read_tables(paths: Iterable[str], columns: Sequence[str] = COLUMNS) -> list[dict[str, str]]:
    """Every row of the TSV job tables, ``_table`` = the file's stem. A first row with a ``name`` cell is the header
    (keys lower-cased); a table without one reads as ``columns``. A missing table is skipped (it may not exist yet)."""
    out: list[dict[str, str]] = []
    for p in paths:
        path = Path(p).expanduser()
        if not path.is_file():
            continue
        with open(path, newline="") as fh:
            lines = list(csv.reader(fh, delimiter="\t"))
        if not lines:
            continue
        head = [c.strip().lower() for c in lines[0]]
        cols, body = (head, lines[1:]) if "name" in head else (list(columns), lines)
        for cells in body:
            r = {k: str(v or "").strip() for k, v in zip(cols, cells)}
            if r.get("name") and not r["name"].startswith("#"):
                out.append({**r, "_table": path.stem})
    return out


def _roots(ctx: Any) -> tuple[Path | None, Path | None]:
    lc = cfg_of(ctx)
    wt = str(getattr(lc, "worktrees_root", "") or "") or str(
        getattr(getattr(ctx.cfg, "project", None), "worktrees_root", "") or "")
    repo = str(getattr(getattr(ctx, "spec", None), "repo", "") or "")
    return (Path(wt) if wt else None), (Path(repo) if repo else None)


def locate(row: Mapping[str, str], wt_root: Path | None, repo: Path | None) -> Path | None:
    """The run's output dir: ``<worktree>/output/<name>`` (an absolute worktree column, or one under
    ``wt_root``), else the unique ``<wt_root>/*/output/<name>`` / ``<repo>/output/<name>``."""
    name = row["name"]
    wt = row.get("worktree") or ""
    if wt:
        p = Path(wt) if Path(wt).is_absolute() else (wt_root / wt if wt_root else None)
        if p is not None and (p / "output" / name).is_dir():
            return p / "output" / name
    hits = [d for d in ([*wt_root.glob(f"*/output/{name}")] if wt_root and wt_root.is_dir() else []) if d.is_dir()]
    if repo is not None and (repo / "output" / name).is_dir():
        hits.append(repo / "output" / name)
    return hits[0] if len(hits) == 1 else None


def fingerprint(out: Path) -> str:
    parts = []
    for rel in ("summary.json", "config.yaml", "eval/aggregate.json", "REQUEUED"):
        try:
            st = (out / rel).stat()
            parts.append(f"{rel}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append(f"{rel}:-")
    ev = out / "eval"
    parts.append(f"eval:{len(list(ev.glob('*.json'))) if ev.is_dir() else 0}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def families(ctx: Any) -> tuple[tuple[str, float, tuple[str, ...]], ...]:
    """(axis key, weight, tasks) per axis from ``loop.project.axes`` (the adapter's families)."""
    def g(a: Any, k: str, d: Any = None) -> Any:
        return a.get(k, d) if isinstance(a, Mapping) else getattr(a, k, d)
    axes = tuple(g(getattr(ctx.cfg, "project", None), "axes", ()) or ())
    return tuple((str(g(a, "key")), float(g(a, "weight", 0.0) or 0.0), tuple(g(a, "tasks", ()) or ()))
                 for a in axes if g(a, "tasks"))


def scorer(ctx: Any) -> Callable[[Mapping[str, Any]], float]:
    """``local_runs.scorer`` ("module:function"); '' = the ``task_score`` function of the execution adapter's module."""
    spec = str(getattr(cfg_of(ctx), "scorer", "") or "")
    if not spec:
        adapter = str(getattr(getattr(ctx, "cfg", None), "execution_adapter", "") or "")
        spec = f"{adapter.partition(':')[0]}:task_score"
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def rescore(out: Path, fams: Sequence[tuple[str, float, tuple[str, ...]]],
            score: Callable[[Mapping[str, Any]], float]) -> tuple[dict[str, Any] | None, list[str], str]:
    """(scores {panel, <axis>..., tasks {t: v}}, integrity notes, '') re-scored from ``eval/<task>.json``; (None,
    [], why) when a task is missing / unscorable (an unfinished eval)."""
    tasks = list(dict.fromkeys(t for _, _, ts in fams for t in ts))
    per: dict[str, float] = {}
    notes: list[str] = []
    name = out.name
    for t in tasks:
        d = _json(out / "eval" / f"{t}.json")
        if not isinstance(d, dict):
            return None, [], f"eval/{t}.json missing"
        if d.get("task") != t:
            return None, [], f"eval/{t}.json is task {d.get('task')!r}"
        try:
            v = float(score(d))
        except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
            return None, [], f"eval/{t}.json unscorable: {exc!r}"[:200]
        if not math.isfinite(v):
            return None, [], f"eval/{t}.json scores {v}"
        ck = str(((d.get("config") or {}).get("model_kwargs") or {}).get("checkpoint") or "")
        if ck and not ck.endswith(f"{name}/checkpoint-last.pth"):
            notes.append(f"{t}: evaluated checkpoint {ck}")
        per[t] = v
    agg = ((_json(out / "eval" / "aggregate.json") or {}).get("summary") or {})
    mean = sum(per.values()) / len(per)
    off = [t for t in tasks if not (isinstance((agg.get(t) or {}).get("score"), (int, float))
                                    and abs(float(agg[t]["score"]) - per[t]) <= AGG_TOL)]
    if off:
        notes.append(f"aggregate.json missing / disagrees on {off[:4]}")
    o = (agg.get("open") or {}).get("score")
    if isinstance(o, (int, float)) and abs(float(o) - mean) > AGG_TOL:
        notes.append(f"aggregate.json open {o:.6f} != re-scored mean {mean:.6f}")
    sc: dict[str, Any] = {k: sum(per[t] for t in ts) / len(ts) for k, _, ts in fams}
    sc["panel"] = mean
    sc["tasks"] = per
    return sc, notes, ""


def _git(repo: Path, *args: str) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        return 1, repr(exc)
    return r.returncode, r.stdout


class _Git:
    """Per-import cache of the git facts of a commit (many runs share one)."""

    def __init__(self, repo: Path | None, root: str, lineages: Mapping[str, str]) -> None:
        self.repo, self.root, self.lineages, self._c = repo, root, dict(lineages), {}

    def facts(self, sha: str) -> dict[str, Any]:
        if sha in self._c:
            return self._c[sha]
        out: dict[str, Any] = {"lineage": "unknown", "numstat": {}, "diff": "", "code_hash": "", "lineage_numstat": {}}
        if self.repo is None or not sha:
            self._c[sha] = out
            return out
        for label, c in self.lineages.items():
            if _git(self.repo, "merge-base", "--is-ancestor", c, sha)[0] == 0:
                out["lineage"] = label
                break

        def numstat(a: str) -> dict[str, list[int]]:
            rc, txt = _git(self.repo, "diff", "--numstat", a, sha, "--", "*.py")
            res: dict[str, list[int]] = {}
            for line in txt.splitlines() if rc == 0 else []:
                p = line.split("\t")
                if len(p) == 3:
                    res[p[2]] = [int(p[0]) if p[0].isdigit() else 0, int(p[1]) if p[1].isdigit() else 0]
            return res
        if self.root:
            out["numstat"] = numstat(self.root)
            rc, diff = _git(self.repo, "diff", self.root, sha, "--", "*.py")
            if rc == 0:
                out["code_hash"] = hashlib.sha1(diff.encode()).hexdigest()[:12] if diff else ""
                out["diff"] = diff if len(diff) <= DIFF_CAP else diff[:DIFF_CAP] + "\n... (diff truncated)"
        if out["lineage"] in self.lineages and out["lineage"] != "unknown":
            out["lineage_numstat"] = numstat(self.lineages[out["lineage"]])
        self._c[sha] = out
        return out


def _flat(x: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(x, Mapping):
        for k, v in x.items():
            out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, Mapping) else {f"{prefix}{k}": v})
    return out


def config_delta(run: Mapping[str, Any], root: Mapping[str, Any], ignore: Iterable[str]) -> dict[str, list[Any]]:
    ign = tuple(ignore)
    keys = sorted(set(run) | set(root))
    return {k: [root.get(k), run.get(k)] for k in keys
            if not any(k == i or k.startswith(i + ".") for i in ign) and run.get(k) != root.get(k)}


def relative_delta(run: Mapping[str, list[Any]], ref: Mapping[str, list[Any]]) -> dict[str, list[Any]]:
    """The config change of a run against ANOTHER run, both given as deltas against the same root ({key: [root, run]}):
    a key's value is the delta's second entry, else the root's (the first entry of whichever delta has it)."""
    out: dict[str, list[Any]] = {}
    for k in sorted(set(run) | set(ref)):
        root = (run.get(k) or ref.get(k))[0]
        a, b = (ref[k][1] if k in ref else root), (run[k][1] if k in run else root)
        if a != b:
            out[k] = [a, b]
    return out


def record(ctx: Any, row: Mapping[str, str], out: Path, root_cfg: Mapping[str, Any], git: _Git,
           fams: Sequence[tuple[str, float, tuple[str, ...]]], score: Callable[[Mapping[str, Any]], float]
           ) -> tuple[dict[str, Any] | None, str]:
    """(the stored row, '') of a FINISHED run, else (None, why not)."""
    lc = cfg_of(ctx)
    summ = _json(out / "summary.json")
    if not isinstance(summ, dict) or summ.get("done") is not True:
        return None, "training not done"
    sc, notes, why = rescore(out, fams, score)
    if sc is None:
        return None, why
    try:
        cfg = _flat(yaml.safe_load((out / "config.yaml").read_text()) or {})
    except (OSError, yaml.YAMLError):
        cfg = {}
    g = summ.get("git") if isinstance(summ.get("git"), dict) else {}
    sha = str(g.get("sha") or row.get("sha") or "")
    facts = git.facts(sha)
    wall = summ.get("wall_seconds")
    req = out / "REQUEUED"
    restarts = len(req.read_text().splitlines()) if req.is_file() else 0
    if g.get("dirty"):
        notes.append("the worktree was DIRTY at training: the code diff may be incomplete")
    lineage = facts["lineage"]
    context = lineage in (lc.context_refs or {})
    return {"name": row["name"], "root": git.root, "context": context, "jobid": row.get("jobid", ""), "table": row.get("_table", ""),
            "lane": row.get("lane", ""), "out": str(out), "branch": str(g.get("branch") or row.get("branch") or ""),
            "sha": sha, "lineage": lineage,
            "comparable": lineage not in tuple(lc.incomparable) and lineage != "unknown" and not context,
            "seed": cfg.get("seed"), "kind": "full" if isinstance(wall, (int, float)) and wall >= lc.full_min_wall_s
            else "smoke", "requeued": req.is_file() or int(summ.get("start_step") or 0) > 0, "restarts": restarts,
            "steps": summ.get("step"), "images": summ.get("images"), "wall_s": wall, "epochs": summ.get("epochs"),
            "scores": sc, "integrity": notes,
            "config_delta": config_delta(cfg, root_cfg, lc.ignore_keys) if root_cfg else {},
            "code_numstat": facts["numstat"], "lineage_numstat": facts["lineage_numstat"],
            "code_hash": facts["code_hash"], "code_diff": facts["diff"],
            "overrides": row.get("overrides", "")[:600], "notes": str(cfg.get("notes") or "")[:300]}, ""


# ------------------------------------------------------------------ the digest
def _sds(ctx: Any) -> dict[str, float]:
    out = {"panel": float(getattr(ctx.cfg, "panel_sd", float("nan")))}
    out.update({k: float(v) for k, v in (getattr(ctx.cfg, "axis_sd", {}) or {}).items()})
    return out


def _change(r: Mapping[str, Any]) -> str:
    cd = {k: v for k, v in (r.get("config_delta") or {}).items()}
    return json.dumps([sorted((k, json.dumps(v, default=str)) for k, v in cd.items()), r.get("code_hash") or ""])


def compute(ctx: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lc = cfg_of(ctx)
    axes = [k for k, _, _ in families(ctx)]
    keys = ["panel", *axes]
    sds = _sds(ctx)
    full = [r for r in rows if r.get("kind") == "full"]
    by_name = {r["name"]: r for r in full}
    refs = {str(s): by_name[n] for s, n in (lc.reference_runs or {}).items() if n in by_name}
    # a previous root's lineage: its runs are diffed against THAT root's same-seed replicate (context only)
    ctx_refs = {lin: {str(s): by_name[n] for s, n in (m or {}).items() if n in by_name}
                for lin, m in (lc.context_refs or {}).items()}
    ref_names = {r["name"] for r in refs.values()} | {r["name"] for m in ctx_refs.values() for r in m.values()}
    groups: dict[str, dict[str, Any]] = {}
    for r in sorted(full, key=lambda r: r["name"]):
        if r["name"] in ref_names:
            continue
        context = r.get("lineage") in ctx_refs
        gk = (r["lineage"], _change(r)) if r.get("comparable") or context else (r["lineage"], "run:" + r["name"])
        g = groups.setdefault(json.dumps(gk), {"names": [], "seeds": [], "deltas": [], "lineage": r["lineage"],
                                               "comparable": bool(r.get("comparable")) and not context,
                                               "context": context, "rows": [], "ref_cfg": None})
        g["names"].append(r["name"])
        g["seeds"].append(r.get("seed"))
        g["rows"].append(r)
        ref = (ctx_refs[r["lineage"]] if context else refs).get(str(r.get("seed"))) \
            if r.get("comparable") or context else None
        if ref is not None:
            g["deltas"].append({k: r["scores"][k] - ref["scores"][k] for k in keys})
            g["ref_cfg"] = g["ref_cfg"] or ref
    out_groups = []
    for g in groups.values():
        n = len(g["deltas"])
        mean = {k: sum(d[k] for d in g["deltas"]) / n for k in keys} if n else {}
        z = {k: (mean[k] / (sds[k] / math.sqrt(n)) if n and sds.get(k, 0) > 0 and math.isfinite(sds[k]) else None)
             for k in keys} if n else {}
        verdict = ("context (previous root)" if g["context"] else "absolute only (other base)") \
            if not g["comparable"] else ("no same-seed reference" if not n else "flat")
        if n and not g["context"]:
            zp = z.get("panel")
            ax = [k for k in axes if (z.get(k) or 0) >= lc.promising_axis_sd]
            if zp is not None and zp <= -lc.harmful_open_sd:
                verdict = "HARMFUL"
            elif (zp is not None and zp >= lc.promising_open_sd) or (ax and (zp is None or zp >= -lc.promising_open_sd)):
                verdict = "PROMISING"
        r0 = g["rows"][0]
        abs_mean = {k: sum(r["scores"][k] for r in g["rows"]) / len(g["rows"]) for k in keys}
        cd = r0.get("config_delta") or {}
        if g["context"]:          # its change is vs its OWN root replicate (config) and lineage commit (code)
            cd = relative_delta(cd, (g["ref_cfg"] or {}).get("config_delta") or {}) if g["ref_cfg"] else {}
        out_groups.append({"names": g["names"], "seeds": g["seeds"], "lineage": g["lineage"],
                           "comparable": g["comparable"], "context": g["context"], "n_paired": n, "delta": mean,
                           "z": z, "abs": abs_mean, "verdict": verdict, "config_delta": cd,
                           "code_numstat": (r0.get("lineage_numstat") if g["context"] else r0.get("code_numstat")) or {},
                           "lineage_numstat": r0.get("lineage_numstat") or {},
                           "branch": r0.get("branch"), "sha": r0.get("sha"),
                           "requeued": [r["name"] for r in g["rows"] if r.get("requeued")],
                           "steps": [r.get("steps") for r in g["rows"]],
                           "integrity": sorted({x for r in g["rows"] for x in r.get("integrity") or []})})
    order = {"PROMISING": 0, "flat": 1, "no same-seed reference": 2, "HARMFUL": 3, "context (previous root)": 4,
             "absolute only (other base)": 5}
    out_groups.sort(key=lambda g: (order.get(g["verdict"], 9), -(g["delta"].get("panel", g["abs"]["panel"])
                                                                  if g["delta"] else g["abs"]["panel"])))
    d = {"n": len(rows), "n_full": len(full), "n_smoke": sum(r.get("kind") != "full" for r in rows),
         "refs": {s: r["name"] for s, r in refs.items()},
         "context_refs": {lin: {s: r["name"] for s, r in m.items()} for lin, m in ctx_refs.items()},
         "ref_scores": {s: {k: r["scores"][k] for k in keys} for s, r in refs.items()},
         "groups": out_groups, "sds": sds, "axes": axes,
         "promising": [g["names"] for g in out_groups if g["verdict"] == "PROMISING"],
         "harmful": [g["names"] for g in out_groups if g["verdict"] == "HARMFUL"]}
    d["text"] = render(d, int(lc.block_chars))
    return d


def _f(x: Any, nd: int = 4, sign: bool = False) -> str:
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return "?"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _short(v: Any) -> str:
    s = json.dumps(v, default=str) if not isinstance(v, str) else v
    return s if len(s) <= 40 else s[:37] + "..."


def change_text(g: Mapping[str, Any], k: int = 8) -> str:
    cd = g.get("config_delta") or {}
    parts = [f"{key}: {_short(a)} -> {_short(b)}" for key, (a, b) in list(cd.items())[:k]]
    if len(cd) > k:
        parts.append(f"+{len(cd) - k} more keys")
    ns = g.get("code_numstat") or {}
    code = ", ".join(f"{Path(p).name} +{a}/-{b}" for p, (a, b) in ns.items()) or "none"
    vs = "its lineage's root" if g.get("context") else "root"
    return f"config {{{'; '.join(parts) or vs}}}; code vs {vs}: {code}"


def _ax(k: str) -> str:
    return k.replace("nb_", "").replace("_mean", "")


def _gline(g: Mapping[str, Any], axes: Sequence[str]) -> str:
    """One group on one line: names, seeds, absolute panel, the deltas (z in paired sd), the change."""
    seeds = "/".join(str(s) for s in g["seeds"])
    a = g["abs"]
    if g["n_paired"]:
        dz = g["z"]
        sc = (f"panel {_f(a['panel'])} (delta {_f(g['delta']['panel'], 4, True)}"
              + (f", {dz['panel']:+.1f} sd)" if dz.get("panel") is not None else ")")
              + "".join(f" {_ax(k)} {_f(g['delta'][k], 4, True)}" + (f" ({dz[k]:+.1f})" if dz.get(k) is not None else "")
                        for k in axes))
    else:
        sc = f"panel {_f(a['panel'])} " + " ".join(f"{_ax(k)} {_f(a[k])}" for k in axes)
    extra = [f"requeued {', '.join(g['requeued'])}"] if g["requeued"] else []
    if g["integrity"]:
        extra.append("integrity: " + "; ".join(g["integrity"])[:120])
    return (f"- {', '.join('local:' + n for n in g['names'])} [seed {seeds}; {g['branch']} {str(g['sha'])[:7]}]: {sc}"
            f" | {change_text(g, 6)}" + (f" | {'; '.join(extra)}" if extra else ""))


def render(d: Mapping[str, Any], chars: int = 6000) -> str:
    sds, axes = d["sds"], d["axes"]
    sdt = ", ".join(f"{k.replace('nb_', '').replace('_mean', '')} {_f(v)}" for k, v in sds.items())
    refs = ", ".join(f"seed {s}: {n}" for s, n in d["refs"].items()) or "none imported"
    ctx_txt = "".join(f"; previous root lineage {lin!r}: " + (", ".join(f"seed {s}: {n}" for s, n in m.items()) or "none")
                      for lin, m in (d.get("context_refs") or {}).items())
    out = [f"{d['n_full']} finished full-budget runs ({d['n_smoke']} smokes not listed). These are the OPERATOR's "
           "single-seed runs (seeds 7338 / 1), NOT CRN-paired with the loop's seeds and NOT ledger nodes: evidence for "
           "a hypothesis, never a measurement of a node. Do NOT re-propose any change listed here as new. A PROMISING "
           "one may be proposed as a CONFIRM node under the loop's own seeds (cite `local:<name>` in "
           "hypothesis.evidence); a HARMFUL one is measured against (build on it only with a stated fix). "
           f"Root replicates: {refs}{ctx_txt}. Delta = run - same-seed root replicate, z in the loop's measured paired sd "
           f"({sdt}; a group of n seeds: sd / sqrt(n)); per-axis z in parentheses."]
    for title, pick in (("PROMISING (confirm under the loop's seeds)", lambda g: g["verdict"] == "PROMISING"),
                        ("HARMFUL (measured loss)", lambda g: g["verdict"] == "HARMFUL"),
                        ("Flat (inside single-seed noise: not a win, not ruled out)", lambda g: g["verdict"] == "flat"),
                        ("No same-seed root replicate (absolute scores only)",
                         lambda g: g["verdict"] == "no same-seed reference"),
                        ("PREVIOUS ROOT lineage (context, NOT the current root: delta / config / code are vs THAT "
                         "lineage's own same-seed root replicate; never a control for the current root, never banked)",
                         lambda g: g.get("context")),
                        ("Other base (NOT the root lineage: absolute scores only, never compare with the root)",
                         lambda g: not g["comparable"] and not g.get("context"))):
        gs = [g for g in d["groups"] if pick(g)]
        if gs:
            out.append(f"### {title}\n" + "\n".join(_gline(g, axes) for g in gs))
    text = "\n".join(out)
    return text if len(text) <= chars else text[:chars] + "\n... (truncated)"


def digest(store: Any) -> dict[str, Any]:
    return dict(store.get_state(K_DIGEST) or {}) if store is not None and hasattr(store, "get_state") else {}


def block(ctx: Any, store: Any) -> str:
    """The brief section ('' when off or nothing imported)."""
    if not enabled(ctx):
        return ""
    d = digest(store)
    if not d.get("text"):
        return ""
    age = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(float(d.get("ts") or 0)))
    return f"{HEADING} (the operator's manual runs, imported {age}; computed, no LLM)\n{d['text']}"


def digest_lines(store: Any) -> list[str]:
    d = digest(store)
    if not d.get("ts"):
        return []
    return [f"- {d.get('n_full', 0)} finished manual runs imported ({d.get('n_smoke', 0)} smokes), "
            f"{len(d.get('groups') or [])} distinct changes",
            "- PROMISING: " + ("; ".join(", ".join(x) for x in d.get("promising") or []) or "none"),
            "- HARMFUL: " + ("; ".join(", ".join(x) for x in d.get("harmful") or []) or "none")]


# ------------------------------------------------------------------ the bank
def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", (text or "").lower())).strip("_")[:n] or "x"


def _ids(g: Mapping[str, Any]) -> set[str]:
    """Every bank id a group's change may already carry: one per run name (the id is its FIRST name when banked, and a
    later seed that sorts first renamed the group: 11-08 ``w4-pdec-s1`` banked the pool-dec change a second time)."""
    return {f"local-confirm-{_slug(n)}" for n in g.get("names") or ()}


def bank_lines(ctx: Any, d: Mapping[str, Any], have: set[str]) -> list[dict[str, Any]]:
    from loop.select.bank import SCORE
    repo = str(getattr(getattr(ctx, "spec", None), "repo", "") or "")
    root = str(getattr(getattr(ctx.cfg, "project", None), "base_sha", "") or "")
    out = []
    for g in d.get("groups") or []:
        if g["verdict"] != "PROMISING" or not g["comparable"]:
            continue
        bid = f"local-confirm-{_slug(g['names'][0])}"
        if _ids(g) & have:          # banked when an earlier seed of the group finished (11-08: w4-pdec-s7338 then -s1)
            continue
        cd = g.get("config_delta") or {}
        knobs = {k: v[1] for k, v in cd.items()}
        files = ", ".join(g.get("code_numstat") or {}) or "none"
        cfg_txt = json.dumps(knobs, default=str)
        idea = (f"CONFIRM the operator's manual run(s) {', '.join(g['names'])} (branch {g['branch']}, commit "
                f"{str(g['sha'])[:12]}) under the loop's own seeds: re-implement EXACTLY its change against the root, "
                f"nothing else. Config: {cfg_txt}. Code ({files}): `git -C {repo} diff {root[:12]}...{str(g['sha'])[:12]} "
                "-- '*.py'` (from their merge base). It was measured locally on the operator's seeds only (not "
                "CRN-paired), so the loop's paired test is the first real measurement.")
        if SCORE.search(idea):
            idea = idea.replace(f"Config: {cfg_txt}.", "Config: the keys in `knobs`.")
        ev = (f"local single-seed runs {', '.join(g['names'])} (seeds {g['seeds']}): delta vs same-seed root "
              + ", ".join(f"{k} {_f(g['delta'][k], 4, True)} ({(g['z'].get(k) or 0):+.1f} sd)" for k in g["delta"]))
        best = max((k for k in g["z"] if k != "panel"), key=lambda k: g["z"].get(k) or -1e9, default="")
        out.append({"id": bid, "title": f"confirm local run {g['names'][0]} under the loop's seeds"[:200],
                    "family": "", "moonshot": False, "priority": True, "idea": idea,
                    "why_it_might_work": f"It moved {best} beyond {cfg_of(ctx).promising_axis_sd:g} paired sd in the "
                                         "operator's manual run without hurting the panel beyond noise.",
                    "v1_fate": f"local import {time.strftime('%Y-%m-%d', time.gmtime(float(d.get('ts') or 0)))}: "
                               "promising, unconfirmed", "source": "loop.localruns", "evidence": ev,
                    "axis": best, "knobs": knobs})
    return out


def _write_bank(ctx: Any, d: Mapping[str, Any]) -> int:
    p = Path(bank_path(ctx))
    have: set[str] = set()
    if p.exists():
        rows = [line for line in p.read_text().splitlines() if line.strip()]
        ids = [json.loads(line).get("id") for line in rows]
        drop: set[str] = set()                    # a change banked twice (one id per seed): keep its first line
        for g in d.get("groups") or []:
            dup = [i for i in ids if i in _ids(g)]
            drop.update(dup[1:])
        if drop:
            p.write_text("".join(line + "\n" for line, i in zip(rows, ids) if i not in drop))
        have = set(ids)
    lines = bank_lines(ctx, d, have)
    if lines:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:
            fh.write("".join(json.dumps(x, default=str) + "\n" for x in lines))
    return len(lines)


# ------------------------------------------------------------------ confirm nodes (local_runs.confirm_nodes)
# 11-09: the banked confirm ideas were never served. Every operator bank line is ``priority`` too, and the localruns
# lines (family '') shared the '-' interleave group with the root-audit arms, so they sat ~100th of 134 pending bank
# ideas -- and a served one is an LLM session asked to RE-IMPLEMENT a change that already exists as a commit. With
# ``confirm_nodes`` the loop builds each promising group's node directly from the run's own commit: the root's commit
# with the run commit's changed writable files (the loop's config file excluded) and the run's effective config delta
# set in the loop's config, QUEUED on the root with ``local_confirm=<bank id>`` (screened first like a program arm,
# never aged out: acquire.program_child), at most ``confirm_max_open`` open at once, best panel delta first.
K_CONFIRMS = "local_runs.confirms"       # {bank id: {ids, names, seeds, branch, sha, knobs, panel, integrity}}
K_BUILT = "local_runs.confirm_nodes"     # {bank id: node id | "-: why not" | {"fails": n}}
K_REFUSED_AT = "local_runs.confirm_refused_at"   # {bank id: the root commit a '-: why' refusal was made against}
LOCAL_TAG = "local_confirm="
#: notes tag: the source manual runs' seeds -- the confirm node never screens on them (``tick.ModuleCore.screen_specs``)
LOCAL_SEEDS = "local_seeds="
BUILD_FAILS = 3


def source_seeds(spec: Mapping[str, Any]) -> set[int]:
    """The int seeds of a confirm spec's source manual runs."""
    return {int(s) for s in spec.get("seeds") or () if isinstance(s, int) and not isinstance(s, bool)
            or isinstance(s, str) and s.isdigit()}


def confirms(ctx: Any) -> bool:
    return enabled(ctx) and bool(getattr(cfg_of(ctx), "confirm_nodes", False))


def confirm_specs(d: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The PROMISING comparable groups as build specs, keyed by bank id."""
    out: dict[str, dict[str, Any]] = {}
    for g in d.get("groups") or []:
        if g.get("verdict") != "PROMISING" or not g.get("comparable") or not g.get("sha"):
            continue
        out[f"local-confirm-{_slug(g['names'][0])}"] = {
            "ids": sorted(_ids(g)), "names": list(g["names"]), "seeds": list(g.get("seeds") or ()),
            "branch": str(g.get("branch") or ""), "sha": str(g["sha"]),
            "knobs": {k: v[1] for k, v in (g.get("config_delta") or {}).items()},
            "panel": float((g.get("delta") or {}).get("panel") or 0.0), "integrity": list(g.get("integrity") or ())}
    return out


def _find_key(lines: Sequence[str], start: int, end: int, indent: int, part: str) -> tuple[int, int | None, int]:
    """(the line of ``part:`` among lines[start:end] at the block's child indent, or -1; that child indent; where the
    block ends)."""
    rx, block = re.compile(rf"^(\s*){re.escape(part)}\s*:(\s|$)"), None
    for i in range(start, end):
        ln = lines[i]
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        ind = len(ln) - len(ln.lstrip())
        if ind <= indent:
            return -1, block, i
        block = ind if block is None else block
        if ind == block and rx.match(ln):
            return i, block, end
    return -1, block, end


def put_yaml_value(text: str, key: str, value: Any) -> str:
    """``text`` with dotted ``key`` set to ``value``: in place when the leaf exists
    (:func:`loop.agents.researcher.set_yaml_value`), else inserted as new lines at the end of its deepest existing
    parent block (missing parents created); every other line kept as it was."""
    from loop.agents.researcher import set_yaml_value
    try:
        return set_yaml_value(text, key, value)
    except ValueError as exc:
        if "not found" not in str(exc):
            raise
    parts = key.split(".")
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    start, end, indent, d, child = 0, len(lines), -1, 0, None
    for d, part in enumerate(parts):
        hit, child, stop = _find_key(lines, start, end, indent, part)
        if hit < 0:
            end = stop
            break
        indent, start = child, hit + 1
        end = next((j for j in range(start, end) if lines[j].strip() and not lines[j].lstrip().startswith("#")
                    and len(lines[j]) - len(lines[j].lstrip()) <= indent), end)
        child = None
    else:
        raise ValueError(f"key {key!r} is a block, not a leaf")
    at = end
    while at > start and not lines[at - 1].strip():          # before the block's trailing blank lines
        at -= 1
    pad = " " * (child if child is not None else indent + 2 if indent >= 0 else 0)
    new = []
    for j, part in enumerate(parts[d:]):
        if j == len(parts) - d - 1:
            v = yaml.safe_dump(value, default_flow_style=True, width=1 << 16).strip()
            v = v[:-4].rstrip() if v.endswith("\n...") else v.removesuffix("...").rstrip()
            new.append(f"{pad}{part}: {v}\n")
        else:
            new.append(f"{pad}{part}:\n")
            pad += "  "
    return "".join(lines[:at] + new + lines[at:])


def build_confirm(ctx: Any, store: Any, adapter: Any, bid: str, spec: Mapping[str, Any], now: float) -> str:
    """One promising group's node id, or '-: why not'. The root's commit + the run commit's changed WRITABLE files
    (the loop's config file excluded: the run commit's copy predates the root's) + the run's config delta, committed
    without a worktree. Refused when the root changed a file the run also changed since their merge base (copying it
    would revert the root), the run deletes a file, the run's tree was dirty, or the lock / config surface breaks."""
    from loop.agents.researcher import config_file
    from loop.contracts import ROOT_NODE_ID, HypothesisKind, HypothesisRecord, Node as StoreNode, NodeStatus, \
        Operator, Strategy
    from loop.core.program import _new_id
    from loop.select.intake import on_surface
    root = store.get_node(ROOT_NODE_ID)
    sha = str(spec["sha"])
    if any("DIRTY" in str(x) for x in spec.get("integrity") or ()):
        return "-: the run's worktree was dirty (its commit is not the code that ran)"
    surf = getattr(getattr(ctx, "spec", None), "surface", None)
    writable = tuple(getattr(surf, "writable", ()) or ())
    locked = tuple(p for p in getattr(surf, "locked", ()) or () if "::" not in p)
    cfg_rel = config_file(ctx, adapter)
    rc, fork = _git(Path(str(getattr(adapter, "repo", "") or ctx.spec.repo)), "merge-base", root.commit, sha)
    fork = fork.strip() if rc == 0 else ""
    if not fork:
        return f"-: no merge base of the root {root.commit[:12]} and {sha[:12]}"
    files: dict[str, str] = {}
    skipped = []
    for rel in adapter.changed_files(fork, sha):      # the RUN's change (root..sha would list the root's own edits)
        if rel == cfg_rel:
            continue
        if not on_surface(rel, writable) or on_surface(rel, locked):
            skipped.append(rel)
            continue
        text = adapter.file_at(sha, rel)
        if text is None:
            return f"-: the run commit deletes {rel}"
        if adapter.file_at(root.commit, rel) != adapter.file_at(fork, rel):
            return f"-: the root changed {rel} since the merge base {fork[:12]} (copying it would revert the root)"
        files[rel] = text
    cfg_text = adapter.file_at(root.commit, cfg_rel)
    if cfg_text is None:
        return f"-: {cfg_rel} not found at the root"
    knobs = dict(spec.get("knobs") or {})
    try:
        for k, v in knobs.items():
            cfg_text = put_yaml_value(cfg_text, str(k), v)
        got = _flat(yaml.safe_load(cfg_text) or {})
    except (ValueError, yaml.YAMLError) as exc:
        return f"-: config: {exc}"[:300]
    if bad := [k for k, v in knobs.items() if got.get(k) != v]:
        return f"-: config keys not set as the run had them: {bad[:4]}"
    files[cfg_rel] = cfg_text
    names = ",".join(spec.get("names") or ())
    code = sorted(set(files) - {cfg_rel})
    commit = adapter.commit_files(root.commit, files, message=(
        f"loop local confirm {bid}: {spec.get('branch') or '?'} {sha[:12]} ({', '.join(code) or 'config only'}) "
        f"+ {knobs}")[:600])
    if not commit or commit == root.commit:
        return "-: no change against the root"
    if hasattr(adapter, "lock_violations") and (bad := list(adapter.lock_violations(commit))):
        return f"-: lock violation: {'; '.join(str(b) for b in bad)[:200]}"
    if hasattr(adapter, "config_violations") and (bad := list(adapter.config_violations(commit))):
        return f"-: config violation: {'; '.join(str(b) for b in bad)[:200]}"
    seeds = source_seeds(spec)
    nid, tag = _new_id(store), _slug(f"local_{spec['names'][0]}", 60)
    fams = getattr(ctx.cfg, "families", {}) or {}
    fam = "architecture" if "architecture" in fams else next(iter(fams), "architecture")
    hid = f"h-{nid}"
    store.put_hypothesis(HypothesisRecord(
        id=hid, task_id=root.task_id, kind=HypothesisKind.TRIAL, mechanism_tag=tag, family=fam, predictions={},
        claim=(f"CONFIRM the operator's manual run(s) {names} (branch {spec.get('branch')}, commit {sha[:12]}) under "
               f"the loop's own seeds: its code ({', '.join(code) or 'none'}) + config {knobs}")[:1000],
        falsification=(f"its own paired increment over the root on FRESH seeds only: never the source runs' seeds "
                       f"{sorted(seeds)} (the root replicate there is the reference the run was selected against; "
                       "same-seed correlation ~.6 keeps that selection noise)")))
    store.put_node(StoreNode(
        id=nid, task_id=root.task_id, commit=commit, parent_id=root.id, strategy=Strategy.EXPLORE,
        operator=Operator.DRAFT, family=fam, mechanism_tag=tag,
        diff_hash=hashlib.sha256(f"{root.commit}\n{commit}".encode()).hexdigest(), status=NodeStatus.QUEUED,
        hypothesis_id=hid, created_ts=now, updated_ts=now,
        notes=f"brief=local_confirm {LOCAL_TAG}{bid} branch={spec.get('branch') or '?'} sha={sha[:12]} runs={names}"
              + (f" {LOCAL_SEEDS}{','.join(map(str, sorted(seeds)))}" if seeds else "")
              + (f" skipped={','.join(skipped)}" if skipped else "")))
    store.set_state(f"config_diff:{nid}", knobs)
    return nid


def build_confirms(ctx: Any, store: Any, adapter: Any, now: float) -> list[str]:
    """The tick's step: each promising group not built yet, best panel delta first, while fewer than
    ``local_runs.confirm_max_open`` local-confirm nodes are open (not closed). A built group's bank ids are marked
    served (no research session re-implements it); a build that raises is retried on later ticks (BUILD_FAILS)."""
    if not confirms(ctx):
        return []
    specs = store.get_state(K_CONFIRMS) or {}
    if not specs or not all(hasattr(adapter, a) for a in ("commit_files", "file_at", "changed_files")):
        return []
    from loop.contracts import CLOSED_STATUSES, ROOT_NODE_ID
    from loop.select import bank as BK
    root = store.get_node(ROOT_NODE_ID)
    base = str(getattr(getattr(ctx.cfg, "project", None), "base_sha", "") or "")
    if root is None or not root.commit or (base and root.commit != base):
        return []                                  # a re-root in progress: the deltas are against another commit
    built, acts = dict(store.get_state(K_BUILT) or {}), []
    refused_at = dict(store.get_state(K_REFUSED_AT) or {})
    _tag_seeds(store, built, specs)
    open_n = sum(1 for v in built.values() if isinstance(v, str) and not v.startswith("-")
                 and (n := store.get_node(v)) is not None and n.status not in CLOSED_STATUSES)
    cap = int(getattr(cfg_of(ctx), "confirm_max_open", 2))
    for bid, spec in sorted(specs.items(), key=lambda kv: -float(kv[1].get("panel") or 0.0)):
        if open_n >= cap:
            break
        recs = [(i, built.get(i)) for i in sorted({bid, *(spec.get("ids") or ())})]
        if any(isinstance(v, str) and not v.startswith("-") for _, v in recs):
            continue                               # built under this id or an earlier seed's name
        if any(isinstance(v, str) and refused_at.get(i) == root.commit for i, v in recs):
            continue                               # refused (or given up) against THIS root
        rec = built.get(bid)                       # a refusal made against an older root is retried (a re-root)
        if isinstance(rec, str):
            rec = None
        try:
            r: Any = build_confirm(ctx, store, adapter, bid, spec, now)
        except Exception as exc:  # noqa: BLE001 - retried on later ticks, bounded
            n = int(rec.get("fails", 0)) + 1 if isinstance(rec, Mapping) else 1
            r = {"fails": n} if n < BUILD_FAILS else f"-: build raised {n}x: {exc!r}"[:300]
        built[bid] = r
        store.set_state(K_BUILT, built)
        if isinstance(r, str) and r.startswith("-"):
            refused_at[bid] = root.commit
            store.set_state(K_REFUSED_AT, refused_at)
        store.log("local_confirm", bank_id=bid, result=r, sha=spec.get("sha"), names=spec.get("names"))
        if isinstance(r, str) and not r.startswith("-"):
            open_n += 1
            for i in sorted({bid, *(spec.get("ids") or ())}):
                BK.mark(store, i, f"local-confirm-node:{r}")
            acts.append(f"local confirm {bid}: node {r} built from {spec.get('branch') or '?'} {str(spec['sha'])[:12]} "
                        f"+ {len(spec.get('knobs') or {})} config key(s); QUEUED on the root, screened first")
        else:
            acts.append(f"local confirm {bid}: {r}"[:300])
    return acts


def _tag_seeds(store: Any, built: Mapping[str, Any], specs: Mapping[str, Any]) -> None:
    """A confirm node built before ``local_seeds=`` existed gets its source seeds tagged (11-09 review of 182a2ab)."""
    for bid, v in built.items():
        if not isinstance(v, str) or v.startswith("-") or (n := store.get_node(v)) is None:
            continue
        seeds = source_seeds(specs.get(bid) or {})
        if seeds and LOCAL_SEEDS not in (n.notes or ""):
            store.update_node(v, notes=f"{n.notes or ''} {LOCAL_SEEDS}{','.join(map(str, sorted(seeds)))}".strip())


# ------------------------------------------------------------------ root seed replicates
#: config keys a root replicate may differ in: identity / logging / loader plumbing only (NOT ``ckpt`` / ``resume``:
#: an initialised or resumed-from-elsewhere run is not the root's experiment)
REPLICATE_IGNORE = ("name", "notes", "output_dir", "seed", "wandb", "wandb_entity", "wandb_project", "num_workers",
                    "prefetch_factor", "auto_resume", "print_freq", "plot_period", "plot_nrow", "plot_ncol",
                    "plot_num_predict", "device", "debug")


def replicate_why_not(row: Mapping[str, Any], root_cfg: Mapping[str, Any], root: str) -> str:
    """'' when the stored row is a root seed replicate, else why not (one line)."""
    if row.get("kind") != "full":
        return "not a full run"
    if not root or row.get("root") != root:
        return "imported against another root"
    if not row.get("sha"):
        return "no git sha (code identity unknown)"
    if not row.get("comparable") or row.get("context"):
        return f"lineage {row.get('lineage')!r} is not the root's"
    if row.get("code_hash") or row.get("code_numstat"):
        return "its code differs from the root (.py diff)"
    if row.get("integrity"):
        return f"integrity notes: {row['integrity'][:2]}"
    if not isinstance(row.get("seed"), int) or isinstance(row.get("seed"), bool):
        return f"seed {row.get('seed')!r} is not an int"
    try:
        cfg = _flat(yaml.safe_load((Path(str(row.get("out") or "")) / "config.yaml").read_text()) or {})
    except (OSError, yaml.YAMLError) as exc:
        return f"config.yaml unreadable: {type(exc).__name__}"
    if not root_cfg:
        return "no root config"
    if delta := config_delta(cfg, root_cfg, REPLICATE_IGNORE):
        return f"its config differs from the root: {sorted(delta)[:4]}"
    return ""


def replicate_run(ctx: Any, row: Mapping[str, Any], task_id: str) -> Any:
    """The done ``root:baseline:<seed>`` :class:`Run` of a replicate row: the adapter's score keys (the axes, the
    task's score_key = the open panel, ``nb_task_<t>``, ``nb_steps`` / ``nb_images`` / ``nb_wall_s``)."""
    from loop.contracts import ROOT_NODE_ID, Run, RunStatus, Stage
    sc = row["scores"]
    scores: dict[str, float] = {k: float(sc[k]) for k, _, _ in families(ctx)}
    if key := str(getattr(getattr(getattr(ctx, "spec", None), "slots", None), "score_key", "") or ""):
        scores[key] = float(sc["panel"])
    scores.update({f"nb_task_{t}": float(v) for t, v in (sc.get("tasks") or {}).items()})
    for key, src in (("nb_steps", "steps"), ("nb_images", "images"), ("nb_wall_s", "wall_s")):
        v = row.get(src)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            scores[key] = float(v)
    seed = int(row["seed"])
    wall = row.get("wall_s")
    return Run(id=Run.make_id(ROOT_NODE_ID, Stage.BASELINE, seed), task_id=task_id, node_id=ROOT_NODE_ID, seed=seed,
               stage=Stage.BASELINE, status=RunStatus.DONE, job_id=str(row.get("jobid") or ""), scores=scores,
               aggregate=float(sc["panel"]), gpu_h=float(wall) / 3600.0 if isinstance(wall, (int, float)) else 0.0,
               output_dir=str(row.get("out") or ""), resumed=int(row.get("restarts") or 0))


def register_root_replicates(ctx: Any, store: Any, rows: Iterable[Mapping[str, Any]], root_cfg: Mapping[str, Any],
                             now: float) -> list[str]:
    """Register every stored row that is a root seed replicate (:func:`replicate_why_not`) as a done
    ``root:baseline:<seed>`` run; only on the root node whose commit is ``loop.project.base_sha``, never at a seed the
    root already has a run on (any stage / status). Idempotent; returns the registered run ids."""
    from loop.contracts import ROOT_NODE_ID
    if not bool(getattr(cfg_of(ctx), "root_replicates", False)) or not all(
            hasattr(store, a) for a in ("get_node", "runs", "put_run")):
        return []
    root = str(getattr(getattr(ctx.cfg, "project", None), "base_sha", "") or "")
    node = store.get_node(ROOT_NODE_ID)
    if node is None or not root or node.commit != root:
        return []
    taken = {int(r.seed) for r in store.runs(node_id=ROOT_NODE_ID)}
    out: list[str] = []
    for row in sorted(rows, key=lambda r: str(r.get("name"))):
        if replicate_why_not(row, root_cfg, root) or int(row["seed"]) in taken:
            continue
        run = replicate_run(ctx, row, store.task_id)
        store.put_run(dataclasses.replace(run, submitted_ts=now))
        store.log("run_finished", run_id=run.id, node_id=ROOT_NODE_ID, status="done", aggregate=run.aggregate)
        store.log("root_replicate", run_id=run.id, name=row.get("name"), out=row.get("out"), sha=row.get("sha"),
                  seed=run.seed, aggregate=run.aggregate)
        taken.add(run.seed)
        out.append(run.id)
    return out


# ------------------------------------------------------------------ the import
def import_runs(ctx: Any, store: Any, now: float, *, write: bool = True) -> dict[str, Any]:
    """Re-read every job-table run whose files changed, store the finished ones, recompute the digest. Returns
    {rows, located, finished, new, updated, unchanged, unfinished, missing, banked, promising, harmful, digest}."""
    from loop import board as BD
    lc = cfg_of(ctx)
    wt_root, repo = _roots(ctx)
    rows = read_tables(lc.job_tables)
    have = {r["name"]: r for r in store.local_runs()}
    root_cfg = BD.root_config(ctx)
    root = str(getattr(getattr(ctx.cfg, "project", None), "base_sha", "") or "")
    git = _Git(repo, root, lc.lineages)
    fams, score = families(ctx), scorer(ctx)
    res = {"rows": len(rows), "located": 0, "finished": 0, "new": 0, "updated": 0, "unchanged": 0, "unfinished": [],
           "missing": []}
    stored = dict(have)
    seen: set[str] = set()
    for row in rows:
        if row["name"] in seen:
            continue
        seen.add(row["name"])
        out = locate(row, wt_root, repo)
        if out is None:
            res["missing"].append(row["name"])
            continue
        res["located"] += 1
        fp = fingerprint(out)
        old = have.get(row["name"])
        if old is not None and old.get("_fp") == fp and old.get("root") == root:   # a re-root re-reads every run
            res["unchanged"] += 1
            res["finished"] += 1
            continue
        rec, why = record(ctx, row, out, root_cfg, git, fams, score)
        if rec is None:
            res["unfinished"].append(f"{row['name']} ({why})")
            continue
        res["finished"] += 1
        res["new" if old is None else "updated"] += 1
        stored[row["name"]] = {**rec, "_fp": fp, "_ts": now}
        if write:
            store.put_local_run(row["name"], fp, rec, now)
    d = compute(ctx, list(stored.values()))
    d["ts"] = now
    res.update(promising=d["promising"], harmful=d["harmful"], digest=d)
    if write:
        store.set_state(K_DIGEST, {k: d[k] for k in ("ts", "n", "n_full", "n_smoke", "refs", "promising", "harmful",
                                                       "text")} | {"groups": [
            {k: g[k] for k in ("names", "seeds", "lineage", "verdict", "delta", "z", "abs", "n_paired")}
            for g in d["groups"]]})
        res["banked"] = _write_bank(ctx, d) if banks(ctx) else 0
        if confirms(ctx):                        # the promising groups the tick builds as confirm nodes
            store.set_state(K_CONFIRMS, confirm_specs(d))
        res["root_replicates"] = register_root_replicates(ctx, store, stored.values(), root_cfg, now)
        store.log("local_runs_import", **{k: (len(v) if isinstance(v, list) and k in ("unfinished", "missing") else v)
                                          for k, v in res.items() if k != "digest"})
    return res


class LocalRuns:
    """The tick's handle: one import at a time in ``pool`` (sync without one), every ``local_runs.every_hours``."""

    def __init__(self, ctx: Any, store: Any, pool: Any = None, clock: Callable[[], float] = time.time) -> None:
        self.ctx, self.store, self.pool, self.clock = ctx, store, pool, clock
        self._fut: Any = None

    def due(self, now: float) -> bool:
        if not enabled(self.ctx) or self._fut is not None:
            return False
        b = self.store.get_state(K_BOOK) or {}
        if b.get("last_ts") is None:
            return True
        return now - float(b.get("last_ts") or 0.0) >= 3600.0 * float(cfg_of(self.ctx).every_hours)

    def _book(self, now: float, err: str = "") -> None:
        b = dict(self.store.get_state(K_BOOK) or {})
        b.update(last_ts=now, last_ok=not err, error=err[:300], imports=int(b.get("imports") or 0) + 1)
        self.store.set_state(K_BOOK, b)

    @staticmethod
    def _line(res: Mapping[str, Any]) -> str:
        return (f"local runs: {res['finished']} finished imported ({res['new']} new, {res['updated']} updated), "
                f"{len(res['unfinished'])} unfinished, {len(res['missing'])} not found; "
                f"{len(res['promising'])} promising, {res.get('banked', 0)} banked"
                + (f"; root seed replicates registered: {', '.join(res['root_replicates'])}"
                   if res.get("root_replicates") else ""))

    def confirm(self, adapter: Any) -> list[str]:
        """The promising groups' confirm nodes (:func:`build_confirms`; ``local_runs.confirm_nodes``)."""
        return build_confirms(self.ctx, self.store, adapter, self.clock())

    def step(self, acts: list[str]) -> None:
        now = self.clock()
        if self._fut is not None and self._fut.done():
            fut, self._fut = self._fut, None
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 - retried at the next cadence
                self._book(now, repr(exc))
                self.store.log("local_runs_error", error=repr(exc)[:500])
                acts.append(f"local runs: import failed ({type(exc).__name__})")
            else:
                self._book(now)
                if res["new"] or res["updated"] or res.get("root_replicates"):
                    acts.append(self._line(res))
        if not self.due(now):
            return
        if self.pool is None:
            try:
                res = import_runs(self.ctx, self.store, now)
            except Exception as exc:  # noqa: BLE001
                self._book(now, repr(exc))
                self.store.log("local_runs_error", error=repr(exc)[:500])
                acts.append(f"local runs: import failed ({type(exc).__name__})")
                return
            self._book(now)
            acts.append(self._line(res))
            return
        self._book(now)                                  # booked at the start: a slow import is not re-submitted
        self._fut = self.pool.submit(import_runs, self.ctx, self.store, now)
