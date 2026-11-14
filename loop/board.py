"""loop.board — COMPETITION BOARD INGEST (``loop.board``; default off).

What the operator did by hand on 2026-11-08 (fasttrack/wave1.md): fetch the project's public Labless experiment log
and its reference points, read every run's pushed config (the ``/source`` JSON: read as TEXT, never executed), find
the seed noise from same-config pairs, compare per-axis profiles, see which root keys nobody ever varied and how far
the frozen reference models sit above us per task. This module does it every ``board.every_hours``:

- :func:`refresh` (in the tick's pool): GET ``project_url`` + ``log_url``, upsert the runs into the loop db
  (``SqliteStore.board_runs``), fetch up to ``source_max`` missing sources (best runs first), keep each source's
  flattened config, file hashes and a unified diff of its ``.py`` files against our root commit (text only), then
  compute the digest (:func:`compute`) and store it (state ``board.digest``).
- :func:`block` renders the stored digest for the v3 briefs / scout / root audit ('' when off or never fetched).
- ``material`` (a hash of: the run count bucket, the board best, the reference points) tells the root audit
  (loop.root_audit) the evidence changed enough to re-audit.

Pure functions over plain dicts; HTTP is injectable (``fetch(url, timeout) -> obj``) for the tests.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import statistics
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

K_DIGEST = "board.digest"      # {ts, n_runs, material, text, stats}
K_BOOK = "board.book"          # {last_ts, last_ok, error, refreshes}
SEED_RX = re.compile(r"[-_](?:s|seed)?\d{1,5}$")
UA = "overfitx-loop-board/1 (read-only)"


# ------------------------------------------------------------------ http + storage
def http_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310 - a configured https GET
        return json.loads(r.read().decode("utf-8", "replace"))


def cfg_of(ctx: Any) -> Any:
    return getattr(getattr(ctx, "cfg", None), "board", None)


def enabled(ctx: Any) -> bool:
    return bool(getattr(cfg_of(ctx), "enabled", False))


def stored_runs(store: Any) -> list[dict[str, Any]]:
    if hasattr(store, "board_runs"):
        return store.board_runs()
    return list((store.get_state("board.runs") or {}).values())


def _put_runs(store: Any, rows: Sequence[Mapping[str, Any]], ts: float) -> None:
    if hasattr(store, "put_board_runs"):
        store.put_board_runs(rows, ts)
        return
    have = dict(store.get_state("board.runs") or {})
    for r in rows:
        old = have.get(r["run_id"]) or {}
        have[r["run_id"]] = {**r, "_source": old.get("_source")}
    store.set_state("board.runs", have)


def _put_source(store: Any, rid: str, src: Mapping[str, Any]) -> None:
    if hasattr(store, "put_board_source"):
        store.put_board_source(rid, src)
        return
    have = dict(store.get_state("board.runs") or {})
    if rid in have:
        have[rid] = {**have[rid], "_source": dict(src)}
        store.set_state("board.runs", have)


# ------------------------------------------------------------------ configs + root
def flatten(x: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(x, Mapping):
        return out
    for k, v in x.items():
        if isinstance(v, Mapping):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _yaml(text: str) -> dict[str, Any]:
    import yaml
    try:
        got = yaml.safe_load(text)
    except Exception:  # noqa: BLE001 - a bad config is no config
        return {}
    return got if isinstance(got, dict) else {}


def _git_show(repo: str, rev: str, path: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", repo, "show", f"{rev}:{path}"], capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    return r.stdout if r.returncode == 0 else None


def root_ref(ctx: Any) -> tuple[str, str, str]:
    """(repo, root commit, root config path) of the task ('' when unknown)."""
    proj = getattr(getattr(ctx, "cfg", None), "project", None)
    repo = str(getattr(getattr(ctx, "spec", None), "repo", "") or "")
    return repo, str(getattr(proj, "base_sha", "") or ""), str(getattr(proj, "config_rel", "") or "")


def root_config(ctx: Any) -> dict[str, Any]:
    repo, sha, rel = root_ref(ctx)
    if not (repo and sha and rel):
        return {}
    text = _git_show(repo, sha, rel)
    return flatten(_yaml(text)) if text else {}


def source_digest(ctx: Any, src: Mapping[str, Any], root_files: Callable[[str], str | None] | None = None,
                  diff_chars: int = 20000) -> dict[str, Any]:
    """A /source JSON -> {config (flattened), files {path: sha1}, diff_stat {path: [+, -]}, diff (text)}. The files are
    only hashed and diffed as text."""
    bc = cfg_of(ctx)
    cpath = str(getattr(bc, "config_path", "config.yaml") or "config.yaml")
    files, cfg, stat, chunks = {}, {}, {}, []
    for f in src.get("files") or []:
        if not isinstance(f, Mapping):
            continue
        path, text = str(f.get("path") or ""), f.get("content")
        if not path or ".." in path.split("/") or path.startswith("/") or not f.get("present") or not isinstance(text, str):
            continue
        files[path] = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]
        if path == cpath:
            cfg = flatten(_yaml(text))
        elif path.endswith(".py") and root_files is not None:
            old = root_files(path)
            if old is None or old == text:
                if old is None:
                    stat[path] = [len(text.splitlines()), 0]
                continue
            d = list(difflib.unified_diff(old.splitlines(), text.splitlines(), f"root/{path}", f"run/{path}",
                                          lineterm="", n=1))
            stat[path] = [sum(l.startswith("+") and not l.startswith("+++") for l in d),
                          sum(l.startswith("-") and not l.startswith("---") for l in d)]
            chunks.append("\n".join(d))
    diff = "\n".join(chunks)
    return {"config": cfg, "files": files, "diff_stat": stat,
            "diff": diff[:diff_chars] + ("\n... (truncated)" if len(diff) > diff_chars else "")}


# ------------------------------------------------------------------ the digest
def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def axes_of(ctx: Any) -> dict[str, tuple[str, ...]]:
    """axis name -> its benchmark tasks (``loop.project.axes[*].tasks``); {} without a project block."""
    proj = getattr(getattr(ctx, "cfg", None), "project", None)
    out = {}
    for a in getattr(proj, "axes", ()) or ():
        g = (lambda k, d=None: a.get(k, d)) if isinstance(a, Mapping) else (lambda k, d=None: getattr(a, k, d))
        if g("tasks"):
            out[str(g("name"))] = tuple(g("tasks"))
    return out


def _score(r: Mapping[str, Any]) -> float | None:
    return _num(r.get("metric_value"))


def _axis_vals(r: Mapping[str, Any], axes: Mapping[str, Sequence[str]]) -> dict[str, float]:
    m = r.get("metrics") or {}
    out = {}
    for a, ts in axes.items():
        v = [_num(m.get(t)) for t in ts]
        if v and all(x is not None for x in v):
            out[a] = statistics.fmean(v)  # type: ignore[arg-type]
    return out


def group_key(r: Mapping[str, Any], ignore: Iterable[str]) -> str:
    cfg = ((r.get("_source") or {}).get("config")) or {}
    if cfg:
        ig = set(ignore)
        return "cfg:" + json.dumps({k: v for k, v in cfg.items() if k.split(".")[-1] not in ig and k not in ig},
                                   sort_keys=True, default=str)
    return "title:" + SEED_RX.sub("", str(r.get("title") or r.get("run_id")))


def _pooled(groups: Iterable[Sequence[float]]) -> tuple[float | None, int]:
    ss, df = 0.0, 0
    for g in groups:
        if len(g) >= 2:
            ss += statistics.variance(g) * (len(g) - 1)
            df += len(g) - 1
    return (math.sqrt(ss / df) if df else None), df


def compute(ctx: Any, runs: Sequence[Mapping[str, Any]], project: Mapping[str, Any] | None,
            root_cfg: Mapping[str, Any] | None = None, incumbent: Mapping[str, float] | None = None) -> dict[str, Any]:
    """The digest: {n_runs, noise, groups, root, axes, refs, untested, varied, contributors, best, material, text}."""
    bc = cfg_of(ctx)
    ignore = tuple(getattr(bc, "ignore_keys", ()) or ())
    axes = axes_of(ctx)
    top_k = int(getattr(bc, "top_k", 10) or 10)
    # dedupe re-submissions of one result (same title + same score)
    seen, rs = set(), []
    for r in runs:
        s = _score(r)
        if s is None:
            continue
        k = (str(r.get("title")), round(s, 6))
        if k in seen:
            continue
        seen.add(k)
        rs.append(r)
    by: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rs:
        by[group_key(r, ignore)].append(r)
    groups = []
    for k, g in by.items():
        sc = [_score(r) for r in g]
        av = [_axis_vals(r, axes) for r in g]
        groups.append({"key": k, "titles": [str(r.get("title")) for r in g], "ids": [str(r.get("run_id")) for r in g],
                       "contributors": sorted({str(r.get("contributor") or "?") for r in g}), "n": len(g),
                       "scores": sc, "mean": statistics.fmean(sc),  # type: ignore[arg-type]
                       "axes": {a: statistics.fmean([x[a] for x in av if a in x]) for a in axes
                                if any(a in x for x in av)},
                       "config": ((g[0].get("_source") or {}).get("config")) or {},
                       "main": any(r.get("is_main") for r in g),
                       "diff_stat": ((max(g, key=lambda r: _score(r) or 0).get("_source") or {}).get("diff_stat")) or {}})
    groups.sort(key=lambda g: -g["mean"])
    sd, df = _pooled([g["scores"] for g in groups])
    axis_sd = {a: _pooled([[x[a] for x in (_axis_vals(r, axes) for r in by[g["key"]]) if a in x] for g in groups])
               for a in axes}
    # the root group: a board run named in board.root_runs, else the group whose config equals our root config
    roots = set(getattr(bc, "root_runs", ()) or ())
    root = next((g for g in groups if roots & set(g["titles"])), None)
    if root is None and root_cfg:
        ig = set(ignore)
        rc = {k: v for k, v in root_cfg.items() if k.split(".")[-1] not in ig}
        root = next((g for g in groups if g["config"] and all(g["config"].get(k) == v for k, v in rc.items())), None)
    base = next((g for g in groups if g["main"]), None)
    # per-axis profile of the top groups vs the root (real = beyond 2 standard errors of a difference of means)
    prof = []
    for g in groups[:top_k]:
        row = {"titles": g["titles"][:3], "n": g["n"], "mean": g["mean"], "contributors": g["contributors"], "axes": {}}
        ref = root or base
        if ref is not None and ref is not g:
            for a in axes:
                if a in g["axes"] and a in ref["axes"] and (s := axis_sd[a][0]):
                    d = g["axes"][a] - ref["axes"][a]
                    se = s * math.sqrt(1 / g["n"] + 1 / ref["n"])
                    row["axes"][a] = (d, "REAL" if abs(d) > 2 * se else "noise")
        prof.append(row)
    # reference models (frozen FMs): per task and per axis vs the root group (else the board best)
    refs = []
    anchor = root or (groups[0] if groups else None)
    for p in (project or {}).get("reference_points") or []:
        m = p.get("metrics") or {}
        ra = {a: statistics.fmean([float(m[t]) for t in ts]) for a, ts in axes.items() if all(_num(m.get(t)) is not None
                                                                                              for t in ts)}
        task_gap = {}
        if anchor is not None:
            am = defaultdict(list)
            for r in by[anchor["key"]]:
                for t, v in (r.get("metrics") or {}).items():
                    if _num(v) is not None and t in m:
                        am[t].append(float(v))
            task_gap = {t: float(m[t]) - statistics.fmean(v) for t, v in am.items() if _num(m.get(t)) is not None}
        refs.append({"label": str(p.get("label") or p.get("id")), "kind": str(p.get("kind") or ""),
                     "score": _num(p.get("final_score")), "axes": ra, "task_gap": task_gap,
                     "axis_gap": {a: ra[a] - anchor["axes"][a] for a in ra if anchor and a in anchor["axes"]}})
    # root keys: never varied on the board vs varied (by whom, the best group that differs on it)
    untested, varied = [], []
    cfg_groups = [g for g in groups if g["config"]]
    own: list[str] = []                 # keys our root sets differently from its own board run (the fork's changes)
    if root_cfg and cfg_groups:
        ig = set(ignore)
        for k, v in sorted(root_cfg.items()):
            if k.split(".")[-1] in ig or k in ig:
                continue
            if root is not None and root["config"] and root["config"].get(k, "<absent>") != v:
                own.append(k)
                continue
            diff = [g for g in cfg_groups if g["config"].get(k, "<absent>") != v]
            if not diff:
                untested.append(k)
            else:
                best = max(diff, key=lambda g: g["mean"])
                varied.append({"key": k, "root": v, "groups": len(diff),
                               "contributors": sorted({c for g in diff for c in g["contributors"]}),
                               "best": best["mean"], "best_title": best["titles"][0], "best_value": best["config"].get(k)})
    # ONE-KEY CONTRASTS: two board configs that differ in exactly one (non-ignored) key -- the board's own ablations
    ig = set(ignore)
    keyed = [(g, {k: v for k, v in g["config"].items() if k.split(".")[-1] not in ig and k not in ig})
             for g in cfg_groups]
    contrasts = []
    for i, (a, ca) in enumerate(keyed):
        for b, cb in keyed[i + 1:]:
            ks = [k for k in set(ca) | set(cb) if ca.get(k, "<absent>") != cb.get(k, "<absent>")]
            if len(ks) != 1 or not sd:
                continue
            lo, hi = (a, b) if a["mean"] <= b["mean"] else (b, a)
            k = ks[0]
            se = sd * math.sqrt(1 / lo["n"] + 1 / hi["n"])
            dax = {x: (hi["axes"][x] - lo["axes"][x],
                       "REAL" if (s := axis_sd[x][0]) and abs(hi["axes"][x] - lo["axes"][x])
                       > 2 * s * math.sqrt(1 / lo["n"] + 1 / hi["n"]) else "noise")
                   for x in axes if x in lo["axes"] and x in hi["axes"]}
            contrasts.append({"key": k, "lo": lo["config"].get(k, "<absent>"), "hi": hi["config"].get(k, "<absent>"),
                              "d": hi["mean"] - lo["mean"], "z": (hi["mean"] - lo["mean"]) / se,
                              "n": (lo["n"], hi["n"]), "titles": (lo["titles"][0], hi["titles"][0]), "axes": dax})
    contrasts.sort(key=lambda c: -c["z"])
    contrib: dict[str, list[float]] = defaultdict(list)
    keys_by: dict[str, Counter] = defaultdict(Counter)
    for g in groups:
        for c in g["contributors"]:
            contrib[c].append(g["mean"])
            if root_cfg and g["config"]:
                keys_by[c].update(k for k, v in g["config"].items()
                                  if k.split(".")[-1] not in set(ignore) and root_cfg.get(k, "<absent>") != v)
            else:
                keys_by[c].update(w for t in g["titles"] for w in re.split(r"[-_]", SEED_RX.sub("", t)) if len(w) > 1)
    contributors = [{"name": c, "configs": len(v), "best": max(v), "top3": statistics.fmean(sorted(v)[-3:]),
                     "themes": [k for k, _ in keys_by[c].most_common(8)]}
                    for c, v in sorted(contrib.items(), key=lambda x: -max(x[1]))]
    best = groups[0] if groups else None
    refs_h = hashlib.sha1(json.dumps([(r["label"], r["score"]) for r in refs], default=str).encode()).hexdigest()[:8]
    bucket = len(rs) // max(1, int(getattr(bc, "material_new_runs", 10) or 10))
    material = hashlib.sha1(f"{bucket}|{best['ids'][0] if best else ''}|{refs_h}".encode()).hexdigest()[:12]
    d = {"n_runs": len(rs), "n_groups": len(groups), "noise": {"sd": sd, "df": df,
                                                                "axis_sd": {a: s for a, (s, _) in axis_sd.items()}},
         "root": {"titles": root["titles"], "n": root["n"], "mean": root["mean"], "axes": root["axes"]} if root else None,
         "base": {"titles": [str(r.get("title")) for r in by[base["key"]] if r.get("is_main")] or base["titles"],
                  "mean": base["mean"], "n": base["n"]} if base else None,
         "profile": prof, "refs": refs, "untested": untested, "varied": varied, "contributors": contributors,
         "own": own, "contrasts": contrasts,
         "best": {"titles": best["titles"], "n": best["n"], "mean": best["mean"]} if best else None,
         "facts": [str(x) for x in getattr(bc, "reference_facts", ()) or ()], "material": material,
         "incumbent": dict(incumbent or {})}
    d["text"] = render(d, int(getattr(bc, "digest_chars", 7000) or 7000))
    return d


def _f(x: Any, nd: int = 4) -> str:
    return "?" if x is None else f"{x:.{nd}f}"


def _per_key(cs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The strongest contrast of each key (|z|), with how many one-key contrasts the board has on it."""
    best: dict[str, dict[str, Any]] = {}
    count: Counter = Counter(c["key"] for c in cs)
    for c in cs:
        if c["key"] not in best or abs(c["z"]) > abs(best[c["key"]]["z"]):
            best[c["key"]] = dict(c)
    out = sorted(best.values(), key=lambda c: -abs(c["z"]))
    for c in out:
        c["titles"] = (c["titles"][0], f"{c['titles'][1]}; {count[c['key']]} contrast(s) on this key")
    return out


def render(d: Mapping[str, Any], chars: int = 7000) -> str:
    nz = d["noise"]
    out = [f"{d['n_runs']} scored runs in {d['n_groups']} distinct configs. SEED NOISE (pooled same-config runs): one "
           f"run's sd {_f(nz['sd'])} on the score ({nz['df']} df)"
           + ("; per axis " + ", ".join(f"{a} {_f(s)}" for a, s in nz["axis_sd"].items() if s is not None)
              if nz["axis_sd"] else "")
           + (f". A single-run difference under ~{2 * nz['sd']:.3f} is noise." if nz["sd"] else ".")]
    if d.get("facts"):
        out.append("Operator-stated facts:\n" + "\n".join(f"- {x}" for x in d["facts"]))
    if d.get("root"):
        r = d["root"]
        out.append(f"OUR ROOT on the board: {', '.join(r['titles'][:3])} mean {_f(r['mean'])} over {r['n']} run(s); "
                   + ", ".join(f"{a} {_f(v)}" for a, v in r["axes"].items()))
    if d.get("base"):
        out.append(f"Unmodified baseline: {d['base']['titles'][0]} {_f(d['base']['mean'])} over {d['base'].get('n', 1)} "
                   "run(s) of its config")
    rows = []
    for p in d["profile"]:
        ax = "; ".join(f"{a} {dv:+.4f} {lab}" for a, (dv, lab) in p["axes"].items())
        rows.append(f"- {_f(p['mean'])} n={p['n']} {', '.join(p['titles'][:2])} ({', '.join(p['contributors'])})"
                    + (f" | vs root: {ax}" if ax else ""))
    out.append("TOP CONFIGS (mean over seeds; per-axis vs our root, REAL = beyond 2 SE, else noise):\n" + "\n".join(rows))
    if d["refs"]:
        lines = []
        for r in d["refs"]:
            gaps = sorted(r["task_gap"].items(), key=lambda x: -x[1])
            lines.append(f"- {r['label']} ({r['kind']}) {_f(r['score'], 3)}; axis gap vs root: "
                         + ", ".join(f"{a} {g:+.3f}" for a, g in r["axis_gap"].items())
                         + ("; largest task gaps: " + ", ".join(f"{t} {g:+.3f}" for t, g in gaps[:4]) if gaps else ""))
        out.append("REFERENCE MODELS (frozen foundation models: the per-task ceiling our features have not reached):\n"
                   + "\n".join(lines))
    if d["untested"]:
        out.append(f"ROOT KEYS NOBODY ON THE BOARD EVER VARIED ({len(d['untested'])}): " + ", ".join(d["untested"][:60]))
    if d.get("own"):
        out.append(f"OUR FORK'S OWN CHANGES over its board run (never scored on the board as such): "
                   + ", ".join(d["own"]))
    if d.get("contrasts"):
        out.append("ONE-KEY CONTRASTS ON THE BOARD (two configs differing in exactly this key; z of the score difference "
                   "over seed noise; per axis REAL = beyond 2 SE):\n" + "\n".join(
                       f"- {c['key']}: {json.dumps(c['lo'], default=str)} -> {json.dumps(c['hi'], default=str)} "
                       f"{c['d']:+.4f} (z {c['z']:.1f}, n {c['n'][0]}/{c['n'][1]}, {c['titles'][0]} -> {c['titles'][1]})"
                       + ("; " + ", ".join(f"{a} {dv:+.4f} {lab}" for a, (dv, lab) in c["axes"].items())
                          if c["axes"] else "")
                       for c in _per_key(d["contrasts"])[:16]))
    if d["varied"]:
        v = sorted(d["varied"], key=lambda x: -x["best"])[:25]
        out.append("ROOT KEYS VARIED ON THE BOARD (key: root value -> best differing config's value, its mean, who):\n"
                   + "\n".join(f"- {x['key']}: {json.dumps(x['root'], default=str)} -> "
                               f"{json.dumps(x['best_value'], default=str)} ({x['best_title']} {_f(x['best'])}; "
                               f"{x['groups']} configs by {', '.join(x['contributors'])})" for x in v))
    if d["contributors"]:
        out.append("CONTRIBUTOR CLUSTERS (configs, best, top-3 mean, recurring changes vs our root):\n" + "\n".join(
            f"- {c['name']}: {c['configs']} configs, best {_f(c['best'])}, top3 {_f(c['top3'])}; "
            f"{', '.join(c['themes'][:8])}" for c in d["contributors"]))
    text = "\n".join(out)
    return text if len(text) <= chars else text[:chars] + "\n... (digest truncated)"


# ------------------------------------------------------------------ the tick step
def refresh(ctx: Any, store: Any, now: float, fetch: Callable[[str, float], Any] | None = None) -> dict[str, Any]:
    """Fetch + store + digest (no LLM). Returns {ok, n_runs, new, sources, material, changed, error}."""
    bc = cfg_of(ctx)
    fetch = fetch or http_json
    to = float(bc.timeout_s)
    project = fetch(bc.project_url, to) if bc.project_url else {}
    log = fetch(bc.log_url, to) if bc.log_url else {}
    rows = [r for r in (log.get("runs") if isinstance(log, Mapping) else log) or [] if isinstance(r, Mapping)]
    before = {r["run_id"]: r for r in stored_runs(store)}
    _put_runs(store, rows, now)
    nsrc = 0
    if bc.source_url and bc.source_max > 0:
        repo, sha, _ = root_ref(ctx)
        cache: dict[str, str | None] = {}

        def root_file(p: str) -> str | None:
            if p not in cache:
                cache[p] = _git_show(repo, sha, p) if repo and sha else None
            return cache[p]
        have = {r["run_id"] for r in stored_runs(store) if r.get("_source")}
        todo = sorted((r for r in rows if r["run_id"] not in have and _score(r) is not None), key=lambda r: -_score(r))
        for r in todo[:bc.source_max]:
            try:
                src = fetch(bc.source_url.format(run_id=r["run_id"]), to)
            except Exception as exc:  # noqa: BLE001 - one missing source is no source
                store.log("board_source_error", run_id=r["run_id"], error=repr(exc)[:300])
                continue
            _put_source(store, r["run_id"], source_digest(ctx, src, root_file))
            nsrc += 1
    inc = store.get_state("board.incumbent_axes") or {}
    d = compute(ctx, stored_runs(store), project if isinstance(project, Mapping) else {}, root_config(ctx), inc)
    old = store.get_state(K_DIGEST) or {}
    d["ts"] = now
    store.set_state(K_DIGEST, {k: d[k] for k in ("ts", "n_runs", "n_groups", "material", "text", "noise", "best",
                                                   "root", "untested", "refs", "facts")})
    changed = old.get("material") != d["material"]
    out = {"ok": True, "n_runs": d["n_runs"], "new": len(set(r["run_id"] for r in rows) - set(before)),
           "sources": nsrc, "material": d["material"], "changed": changed}
    store.log("board_refresh", **out)
    return out


class Board:
    """The tick's handle: one refresh at a time in ``pool`` (sync without one), every ``board.every_hours``."""

    def __init__(self, ctx: Any, store: Any, pool: Any = None, clock: Callable[[], float] = time.time,
                 fetch: Callable[[str, float], Any] | None = None) -> None:
        self.ctx, self.store, self.pool, self.clock, self.fetch = ctx, store, pool, clock, fetch
        self._fut: Any = None

    def due(self, now: float) -> bool:
        if not enabled(self.ctx) or self._fut is not None:
            return False
        b = self.store.get_state(K_BOOK) or {}
        if b.get("last_ts") is None:
            return True
        gap = 3600.0 * float(cfg_of(self.ctx).every_hours) if b.get("last_ok", True) else 1800.0
        return now - float(b.get("last_ts") or 0.0) >= gap

    def _book(self, now: float, res: Mapping[str, Any] | None, err: str = "") -> None:
        b = dict(self.store.get_state(K_BOOK) or {})
        b.update(last_ts=now, last_ok=not err, error=err[:300], refreshes=int(b.get("refreshes") or 0) + 1)
        self.store.set_state(K_BOOK, b)

    def step(self, acts: list[str]) -> None:
        now = self.clock()
        if self._fut is not None and self._fut.done():
            fut, self._fut = self._fut, None
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 - a failed fetch is retried in 30 min
                self._book(now, None, repr(exc))
                self.store.log("board_error", error=repr(exc)[:500])
                acts.append(f"board: refresh failed ({type(exc).__name__}); retry in 30 min")
            else:
                self._book(now, res)
                acts.append(f"board: {res['n_runs']} runs ({res['new']} new, {res['sources']} sources); digest "
                            + ("CHANGED" if res["changed"] else "unchanged"))
        if not self.due(now):
            return
        if self.pool is None:
            try:
                res = refresh(self.ctx, self.store, now, self.fetch)
            except Exception as exc:  # noqa: BLE001
                self._book(now, None, repr(exc))
                self.store.log("board_error", error=repr(exc)[:500])
                acts.append(f"board: refresh failed ({type(exc).__name__})")
                return
            self._book(now, res)
            acts.append(f"board: {res['n_runs']} runs ({res['new']} new, {res['sources']} sources)")
            return
        self._fut = self.pool.submit(refresh, self.ctx, self.store, now, self.fetch)
        acts.append("board: refresh started")


def digest(store: Any) -> dict[str, Any]:
    return dict(store.get_state(K_DIGEST) or {}) if store is not None and hasattr(store, "get_state") else {}


def block(ctx: Any, store: Any, chars: int | None = None) -> str:
    """The digest as a brief section ('' when the board is off or was never fetched)."""
    if not enabled(ctx):
        return ""
    d = digest(store)
    if not d.get("text"):
        return ""
    age = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(float(d.get("ts") or 0)))
    text = d["text"] if chars is None else d["text"][:chars]
    return (f"## Competition board digest (COMPUTED from the public Labless log, {age}; other contributors' runs, "
            "not ours: each is ONE run unless n > 1)\n" + text)


def is_board_id(store: Any, rid: str) -> bool:
    return any(r.get("run_id") == rid for r in stored_runs(store)) if store is not None else False
