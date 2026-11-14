"""loop.databank — THE DATA BANK (operator 2026-10-07: "the SYSTEM chooses and consumes datasets"). The re-tiling
pipeline (scripts/retile/build.py, outside the loop) writes TCGA tile datasets in the dataloader's Arrow format to
``databank.root/<name>/`` with a one-word ``STATUS`` (queued | extracting | packing | ready | failed). The loop:

* REGISTERS every ready non-pilot dataset each tick (one small file read per dataset) with its manifest summary
  (tile px, mpp mix, tiles, slides): the researchers' data bank line (:func:`bank_lines`);
* BUILDS researchers' ``dataset_spec`` requests (a finish or an idea plan; validated here, the name derived from the
  spec's content hash when absent, deduped by name / content): build.py is launched detached, at most
  ``max_building`` builds at once, and refused when the bank would pass ``max_total_tb`` (projected) -- every
  refusal is logged against the requesting node;
* HOLDS a node whose ``data.dataset_dir`` points at a dataset not ready yet (no run is launched) and voids it when
  that build failed or the dataset was never requested;
* ADOPTS each ready dataset named in ``databank.adopt`` (operator 2026-10-07: per dataset, none by default -- a
  researcher's requested dataset is consumed by the requesting node itself) with no LLM: a config-only node on the incumbent (``data.dataset_dir`` + the
  ``num_workers`` bump) whose researcher-less plan declares a CAMPAIGN over ``adopt_space`` (crop scales / sizes,
  LR): larger tiles change what a crop sees, so it is tuned (loop.campaign) before it is judged;
* lists, in the daily digest, datasets no open node references older than ``retain_days`` (never deleted).

A smoke of a node on a not-ready dataset reads ``smoke_dataset`` (same schema) instead (:func:`stand_in`)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from loop import project as P
from loop.contracts import (CLOSED_STATUSES, PANEL, ROOT_NODE_ID, AxisPrediction, HypothesisKind, HypothesisRecord,
                            NodeStatus, Operator)

K_BANK = "databank"     # {"registered": {name: summary}, "requests": {name: req}, "adopted": {name: node id}}
BUILDING = ("queued", "extracting", "packing")
BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "retile" / "build.py"
BYTES_224 = 29_900      # build.py's projection: mean JPEG q95 bytes of a 224 px tile, ~ area
SOURCE_ROWS = {"original_arrow": 4_000_000, "sample_list": 25_000_000}
_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_TXT = {"type": "string"}
SPEC_SCHEMA = {
    "type": "object", "required": ["tile_px", "target_mpp", "n_tiles", "seed"],
    "description": "optional: a NEW dataset built offline from the raw TCGA WSIs (scripts/retile; same Arrow format, "
                   "one tile = one sample, caps unchanged). The loop builds it (one at a time, disk-capped), registers "
                   "it in the data bank when ready and tunes it on the incumbent itself. name: optional "
                   "[A-Za-z0-9_-] (default: from the spec's hash); point data.dataset_dir at "
                   "<databank root>/<name> to train on it (the node waits until it is ready).",
    "properties": {
        "name": _TXT, "tile_px": {"type": "integer", "minimum": 64, "maximum": 2048},
        "target_mpp": {"description": "'native_mix' (each source tile's own level: today's scale mix with more "
                                      "context) | output microns/px (e.g. 0.5) | [{mpp, frac}] mixture (fracs sum 1)"},
        "n_tiles": {"description": "int or 'all' (every eligible source row)"},
        "seed": {"type": "integer"}, "jpeg_quality": {"type": "integer", "minimum": 50, "maximum": 100},
        "sampling": {"type": "object", "properties": {
            "source": {"type": "string", "enum": ["original_arrow", "sample_list"],
                       "description": "the 4M (slide, x, y, level) rows of the current data | 25M tissue-filtered rows"},
            "slide_balance": {"type": "string", "enum": ["match_source", "equal_per_slide"]},
            "levels": {"type": ["array", "null"], "items": {"type": "integer"}},
            "exclude_upsample": {"type": "boolean"}, "min_tissue": {"type": "number", "minimum": 0, "maximum": 1},
            "tissue_retries": {"type": "integer", "minimum": 0, "maximum": 10},
            "score_filter": {"type": ["object", "null"], "description": "offline curation scores: {path: parquet("
                             "stem, x, y, level, score), min?, top_frac?}"}}}}}
SAMPLING = {"source": "original_arrow", "slide_balance": "match_source", "exclude_upsample": True, "min_tissue": 0.0,
            "tissue_retries": 0, "levels": None, "score_filter": None}
OPERATIONAL = ("name", "slurm", "reader", "est_bytes_per_tile", "num_shards", "keep_parts")


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def content(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The spec's CONTENT (what the tiles are) with build.py's defaults: two specs build the same dataset iff equal."""
    out = {"jpeg_quality": 95, **{k: v for k, v in spec.items() if k not in OPERATIONAL}}
    out["sampling"] = {**SAMPLING, **(spec.get("sampling") or {})}
    return out


def spec_hash(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(content(spec), sort_keys=True, default=str).encode()).hexdigest()


def validate(spec: Any) -> tuple[dict[str, Any], str]:
    """(the spec to build, '') or ({}, why it is refused): scripts/retile/README.md's schema, TCGA only, no GenBio,
    no pilot name, no operational keys (slurm is the pipeline's default)."""
    from loop.select.intake import DISALLOWED_DATA
    if not isinstance(spec, Mapping):
        return {}, "dataset_spec must be an object"
    if DISALLOWED_DATA.search(json.dumps(spec, default=str)):
        return {}, P.rule("databank_forbid_msg", "GenBio data / models may not be used")
    known = {"name", "tile_px", "target_mpp", "n_tiles", "seed", "jpeg_quality", "sampling"}
    if bad := sorted(set(spec) - known):
        return {}, f"unknown key(s) {bad} (known: {sorted(known)})"
    if miss := [k for k in ("tile_px", "target_mpp", "n_tiles", "seed") if k not in spec]:
        return {}, f"missing {miss}"
    s = dict(spec)
    if not (isinstance(s["tile_px"], int) and not isinstance(s["tile_px"], bool) and 64 <= s["tile_px"] <= 2048):
        return {}, "tile_px: an int in [64, 2048]"
    t = s["target_mpp"]
    groups = t if isinstance(t, list) else [{"mpp": t, "frac": 1.0}]
    if not groups or not all(isinstance(g, Mapping) and (g.get("mpp") == "native_mix" or _num(g.get("mpp"))
                                                         and 0.05 <= g["mpp"] <= 64) and _num(g.get("frac"))
                             and g["frac"] > 0 for g in groups) or abs(sum(g["frac"] for g in groups) - 1) > 1e-6:
        return {}, "target_mpp: 'native_mix', a microns/px in [0.05, 64], or [{mpp, frac}] with fracs summing to 1"
    n = s["n_tiles"]
    if not (n == "all" and not isinstance(t, list) or isinstance(n, int) and not isinstance(n, bool)
            and 1000 <= n <= 25_000_000):
        return {}, "n_tiles: an int in [1000, 25M], or 'all' for a single-mpp spec"
    if not isinstance(s["seed"], int) or isinstance(s["seed"], bool):
        return {}, "seed: an int"
    if "jpeg_quality" in s and not (isinstance(s["jpeg_quality"], int) and 50 <= s["jpeg_quality"] <= 100):
        return {}, "jpeg_quality: an int in [50, 100]"
    sm = s.get("sampling") or {}
    if not isinstance(sm, Mapping) or (bad := sorted(set(sm) - set(SAMPLING))):
        return {}, f"sampling: unknown key(s) {bad if isinstance(sm, Mapping) else sm} (known: {sorted(SAMPLING)})"
    checks = {"source": lambda v: v in SOURCE_ROWS, "slide_balance": lambda v: v in ("match_source", "equal_per_slide"),
              "levels": lambda v: v is None or isinstance(v, list) and all(isinstance(x, int) for x in v),
              "exclude_upsample": lambda v: isinstance(v, bool), "min_tissue": lambda v: _num(v) and 0 <= v <= 1,
              "tissue_retries": lambda v: isinstance(v, int) and 0 <= v <= 10,
              "score_filter": lambda v: v is None or isinstance(v, Mapping) and isinstance(v.get("path"), str)
              and set(v) <= {"path", "min", "top_frac"}}
    if bad := [k for k, v in sm.items() if not checks[k](v)]:
        return {}, f"sampling: bad value for {bad}"
    if (sf := sm.get("score_filter")) and not Path(sf["path"]).is_file():
        return {}, f"sampling.score_filter.path {sf['path']} does not exist (compute the scores offline first)"
    name = str(s.get("name") or f"ds{s['tile_px']}_{spec_hash(s)[:8]}")
    if not _NAME.fullmatch(name) or "pilot" in name.lower() or name.startswith("_"):
        return {}, f"name {name!r}: [A-Za-z0-9_-], at most 64 chars, never a pilot name"
    return {**s, "name": name}, ""


# ------------------------------------------------------------------------------------------------ the bank on disk
def _root(ctx: Any) -> Path:
    return Path(ctx.cfg.databank.root)


def pilot(name: str) -> bool:
    return "pilot" in name.lower() or name.startswith("_")


def status(ctx: Any, name: str) -> str:
    """STATUS of ``root/<name>`` ('' when there is none)."""
    try:
        return (_root(ctx) / name / "STATUS").read_text().strip()
    except OSError:
        return ""


def on_disk(ctx: Any) -> dict[str, str]:
    """name -> STATUS of every dataset directory under the root (pilots and ``_meta`` included)."""
    try:
        dirs = [p for p in _root(ctx).iterdir() if p.is_dir() and not p.name.startswith("_")]
    except OSError:
        return {}
    return {p.name: status(ctx, p.name) for p in sorted(dirs)}


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def dataset_of(ctx: Any, path: Any) -> str:
    """The data-bank dataset a ``data.dataset_dir`` value points at ('' when it is not under the root)."""
    if not isinstance(path, str) or not path:
        return ""
    root = os.path.normpath(str(_root(ctx)))
    p = os.path.normpath(path)
    return p[len(root) + 1:].split("/")[0] if p.startswith(root + "/") else ""


def stand_in(ctx: Any, path: Any) -> str:
    """The dataset a SMOKE reads for ``path`` when that is a data-bank dataset not ready yet, else ''."""
    db = getattr(ctx.cfg, "databank", None)
    name = dataset_of(ctx, path) if db is not None and db.enabled else ""
    return db.smoke_dataset if name and status(ctx, name) != "ready" else ""


def summary(ctx: Any, name: str) -> dict[str, Any]:
    """A ready dataset's manifest in a few fields."""
    d = _root(ctx) / name
    m = _read(d / "manifest.json")
    spec, st = m.get("spec") or _read(d / "spec.json"), _read(d / "status.json")
    pct = m.get("output_mpp_percentiles") or {}
    return {"path": str(d), "tile_px": spec.get("tile_px"), "target_mpp": spec.get("target_mpp"),
            "n_tiles": m.get("n_tiles"), "n_slides": m.get("n_slides"), "bytes": m.get("bytes"),
            "mpp_p25_50_75": [pct.get(k) for k in ("25", "50", "75")] if pct else None,
            "sampling": spec.get("sampling") or m.get("filter"), "ready": st.get("ready") or m.get("created") or "",
            "spec_sha": spec_hash(spec) if spec else ""}


def line(ctx: Any, s: Mapping[str, Any]) -> str:
    sm, p = s.get("sampling") or {}, s.get("mpp_p25_50_75")
    mpp = s.get("target_mpp")
    mpp = mpp if isinstance(mpp, str) else json.dumps(mpp)
    out = (f"{s['path']}: {s.get('n_tiles') or 0:,} tiles of {s.get('tile_px')} px, target mpp {mpp}"
           + (f" (output mpp p25/p50/p75 {'/'.join(f'{x:.2f}' for x in p if x is not None)})" if p else "")
           + f", {s.get('n_slides') or 0:,} slides, sampling {sm.get('source', '?')}/{sm.get('slide_balance', '?')}"
           + (f" min_tissue {sm['min_tissue']}" if sm.get("min_tissue") else "")
           + (f" score_filter {sm['score_filter'].get('path')}" if isinstance(sm.get("score_filter"), Mapping) else "")
           + (f", {s['bytes'] / 1e9:.0f} GB" if s.get("bytes") else ""))
    return out + (f". {ctx.cfg.databank.note}" if ctx.cfg.databank.note and (s.get("tile_px") or 0) > 224 else "")


def book(store: Any) -> dict[str, Any]:
    return {"registered": {}, "requests": {}, "adopted": {}, **(store.get_state(K_BANK) or {})}


def bank_lines(ctx: Any, store: Any) -> list[str]:
    """The researchers' data bank: the spec's curated entries, every registered dataset, the builds in flight."""
    out = list(getattr(ctx.cfg, "data_bank", ()) or ())
    db = getattr(ctx.cfg, "databank", None)
    if db is None or not db.enabled or not hasattr(store, "get_state"):
        return out
    bk = book(store)
    out += [line(ctx, s) for _, s in sorted(bk["registered"].items())]
    out += [f"{_root(ctx) / n}: BUILDING ({r['state']}; requested by {', '.join(r['by'])}) -- {json.dumps(content(r['spec']))}"
            for n, r in sorted(bk["requests"].items()) if r["state"] in ("waiting", "submitted")]
    return out


def _projected(spec: Mapping[str, Any]) -> float:
    """build.py's disk projection of a whole dataset, bytes."""
    n = spec.get("n_tiles")
    n = SOURCE_ROWS.get(content(spec)["sampling"]["source"], 4_000_000) if n == "all" else int(n or 0)
    return n * float(spec.get("est_bytes_per_tile") or BYTES_224 * (int(spec["tile_px"]) / 224) ** 2) * 1.02


def usage_tb(ctx: Any, bk: Mapping[str, Any]) -> float:
    """TB the bank holds or will hold: ready datasets' manifest bytes, the others' projections."""
    total, seen = 0.0, set()
    for name, st in on_disk(ctx).items():
        seen.add(name)
        d = _root(ctx) / name
        b = _read(d / "manifest.json").get("bytes") if st == "ready" else None
        if b is None and st in BUILDING and (spec := _read(d / "spec.json")).get("tile_px"):
            b = _projected(spec)
        total += float(b or 0.0)
    total += sum(_projected(r["spec"]) for n, r in bk["requests"].items() if n not in seen and r["state"] == "waiting")
    return total / 1e12


# ------------------------------------------------------------------------------------------------ requests
def request(ctx: Any, store: Any, by: str, spec: Any) -> str:
    """Record a researcher's ``dataset_spec`` (``by``: the node, else the research request): queued for a build, an
    existing / requested dataset with the same content (deduped), or refused -- logged against ``by``."""
    if not ctx.cfg.databank.enabled:
        return ""
    bk = book(store)
    got, why = validate(spec)
    name = got.get("name", "")
    if not why:
        h = spec_hash(got)
        same = next((n for n, s in bk["registered"].items() if s.get("spec_sha") == h), "") or next(
            (n for n, r in bk["requests"].items() if spec_hash(r["spec"]) == h and r["state"] != "refused"), "")
        disk = _read(_root(ctx) / name / "spec.json")
        if not same and disk and spec_hash(disk) == h:
            same = name
        if same:
            if same in bk["requests"] and by not in bk["requests"][same]["by"]:
                bk["requests"][same]["by"] = bk["requests"][same]["by"] + [by]
                store.set_state(K_BANK, bk)
            store.log("databank_request", node_id=by, dataset=same, outcome="deduped")
            return f"databank: {by} asked for {name}: same content as {same} (deduped)"
        if disk or name in bk["requests"] or name in bk["registered"]:
            why = f"name {name} already holds a different spec (datasets are immutable: pick a new name)"
        elif (tb := usage_tb(ctx, bk) + _projected(got) / 1e12) > ctx.cfg.databank.max_total_tb:
            why = f"projected bank {tb:.2f} TB > databank.max_total_tb {ctx.cfg.databank.max_total_tb:g}"
    if why:
        store.log("databank_request", node_id=by, dataset=name, outcome="refused", reason=why)
        return f"databank: REFUSED {by}'s dataset_spec{f' {name}' if name else ''}: {why}"
    bk["requests"][name] = {"spec": got, "by": [by], "state": "waiting", "ts": time.time()}
    store.set_state(K_BANK, bk)
    store.log("databank_request", node_id=by, dataset=name, outcome="queued", spec=got)
    return f"databank: {by} requested {name} ({json.dumps(content(got))}); queued for a build"


def _alive(pid: Any) -> bool:
    return bool(pid) and Path(f"/proc/{int(pid)}").exists()


def launch(ctx: Any, spec_path: Path, log: Path) -> int:
    """build.py --spec, detached (it plans inline, then submits the Slurm jobs and exits): its pid."""
    db = ctx.cfg.databank
    with log.open("a") as f:
        py = db.python or str(getattr(ctx.cfg.project, "python", "") or "") or sys.executable
        p = subprocess.Popen([py, str(BUILD_SCRIPT), "--spec", str(spec_path), "--min-free-tb",
                              str(db.min_free_tb)], stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, cwd=str(BUILD_SCRIPT.parent))
    return p.pid


def _builds(ctx: Any, store: Any, bk: dict, acts: list[str]) -> None:
    """Submitted builds: STATUS ready / failed (or build.py exited leaving none) settles them; then the next
    waiting request is launched while fewer than ``max_building`` builds are in flight."""
    disk = on_disk(ctx)
    for name, r in bk["requests"].items():
        if r["state"] != "submitted":
            continue
        st = disk.get(name, "")
        if st in ("ready", "failed") or (st not in BUILDING and not _alive(r.get("pid"))):
            r["state"] = st if st in ("ready", "failed") else "failed"
            if r["state"] == "failed":
                r["reason"] = _read(_root(ctx) / name / "status.json").get("reason") or \
                    f"build.py exited without a build (log {r.get('log')})"
                store.log("databank_build_failed", dataset=name, reason=str(r["reason"])[:500], by=r["by"])
                for by in r["by"]:
                    store.log("databank_request", node_id=by, dataset=name, outcome="failed", reason=str(r["reason"])[:300])
                acts.append(f"databank: build {name} FAILED: {str(r['reason'])[:200]}")
    busy = sum(st in BUILDING for st in disk.values()) + sum(
        r["state"] == "submitted" and disk.get(n, "") not in BUILDING for n, r in bk["requests"].items())
    for name, r in sorted(bk["requests"].items(), key=lambda x: x[1]["ts"]):
        if r["state"] != "waiting" or busy >= ctx.cfg.databank.max_building:
            continue
        d = Path(ctx.state_dir) / "databank"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps(r["spec"], indent=1))
        try:
            r.update(state="submitted", pid=launch(ctx, d / f"{name}.json", d / f"{name}.log"), log=str(d / f"{name}.log"),
                     submitted=time.time())
        except OSError as exc:
            r.update(state="failed", reason=f"launch: {exc!r}"[:300])
            store.log("databank_build_failed", dataset=name, reason=r["reason"], by=r["by"])
            continue
        busy += 1
        store.log("databank_build", dataset=name, pid=r["pid"], by=r["by"])
        acts.append(f"databank: build {name} launched (pid {r['pid']}, for {', '.join(r['by'])})")


# ------------------------------------------------------------------------------------------------ holding
def node_dataset(store: Any, node_id: str, cache: dict[str, str] | None = None) -> str:
    """The node's effective ``data.dataset_dir`` (its config never changes: cached)."""
    from loop.select.intake import parent_config
    if cache is not None and node_id in cache:
        return cache[node_id]
    d = parent_config(store, node_id).get("data.dataset_dir")
    d = d if isinstance(d, str) else ""
    if cache is not None:
        cache[node_id] = d
    return d


def gate(ctx: Any, store: Any, nodes: Sequence[Any], void: Callable[[Any, str], Any],
         cache: dict[str, str] | None = None) -> set[str]:
    """The nodes of ``nodes`` on a data-bank dataset not ready yet (held: nothing launches); one whose build failed,
    or a dataset nobody requested, is voided through ``void(node, reason)`` (and returned too)."""
    if not ctx.cfg.databank.enabled:
        return set()
    bk, held = book(store), set()
    for n in nodes:
        name = dataset_of(ctx, node_dataset(store, n.id, cache))
        if not name or (st := status(ctx, name)) == "ready":
            continue
        r = bk["requests"].get(name) or {}
        held.add(n.id)
        if st == "failed" or r.get("state") in ("failed", "refused"):
            void(n, f"dataset {name} failed: " + str(r.get("reason") or _read(_root(ctx) / name / "status.json")
                                                     .get("reason") or "build failed")[:200])
        elif not st and not r:
            void(n, f"dataset {name} does not exist under {_root(ctx)} and was never requested (dataset_spec)")
    return held


# ------------------------------------------------------------------------------------------------ adoption
def space(ctx: Any, base: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``adopt_space`` resolved against the incumbent's config: {key, low, high, scale} (a ``factor`` -> value / f ..
    value x f); keys it lacks are dropped (loop.campaign keeps only the settable ones)."""
    from loop.campaign import resolve
    return resolve(ctx.cfg.databank.adopt_space, base)


def adopt(ctx: Any, store: Any, adapter: Any, snapshot: Any, name: str, new_id: Any, now: float) -> tuple[str, str]:
    """(node id, '') or ('', why not): ``name`` as a config-only node on the incumbent, its campaign queued."""
    from loop import campaign as CP
    from loop import design as D
    inc = getattr(snapshot, "incumbent_id", None) or ROOT_NODE_ID
    s = book(store)["registered"][name]
    with D._Tree(ctx, store, adapter, f"databank-{name}", base_id=inc) as t:
        if "data.dataset_dir" not in t.base:
            return "", f"the incumbent {inc}'s config has no data.dataset_dir"
        if dataset_of(ctx, t.base["data.dataset_dir"]) == name:
            return "", f"the incumbent {inc} already trains on it"
        levels: dict[str, Any] = {"data.dataset_dir": s["path"]}
        nw = ctx.cfg.databank.num_workers
        if nw and _num(t.base.get("train.num_workers")) and t.base["train.num_workers"] < nw:
            levels["train.num_workers"] = nw
        fam = "data_curation" if "data_curation" in ctx.cfg.families else next(
            (f for f, r in ctx.cfg.families.items() if "data.dataset_dir" in r.keys), "data_curation")
        hyp = HypothesisRecord(
            id="", task_id=ctx.task_id, kind=HypothesisKind.TRIAL, family=fam, mechanism_tag=f"databank_{name}",
            claim=f"train the incumbent {inc} on the data-bank dataset {line(ctx, s)} instead of "
                  f"{t.base['data.dataset_dir']}; a campaign over the crop / LR knobs tunes it before it is judged",
            predictions={PANEL: AxisPrediction(1, 1e-3)}, created_ts=now,
            falsification=f"no tuned config on {name} beats {inc} on fresh confirm seeds")
        nid = t.node(levels, knobs=[{"key": k, "low": t.base[k], "high": v, "levers": {"data_curation": 1}}
                                    for k, v in levels.items()], tag=f"databank_{name}", operator=Operator.DRAFT,
                     new_id=new_id, hyp=hyp, notes=f"dataset={name} brief=databank")
        sp = space(ctx, t.base)
    if not nid or nid == inc:
        return "", "rejected at intake (design_point_rejected in the log)"
    store.update_node(nid, family=fam)
    store.set_state(f"ideation:{nid}", {"campaign": True, "search_space": sp, "dataset": name})
    CP.enqueue(store, nid, f"declared campaign: data-bank adoption of {name}")
    return nid, ""


# ------------------------------------------------------------------------------------------------ the tick
def _register(ctx: Any, store: Any, bk: dict, acts: list[str]) -> bool:
    """Every ready non-pilot dataset under the root not registered yet: its manifest summary (one STATUS read each)."""
    new = False
    for name, st in on_disk(ctx).items():
        if st == "ready" and not pilot(name) and name not in bk["registered"]:
            bk["registered"][name] = summary(ctx, name)
            store.log("databank_registered", dataset=name, **bk["registered"][name])
            acts.append(f"databank: registered {name} ({line(ctx, bk['registered'][name])})")
            new = True
    return new


def register(ctx: Any, store: Any) -> list[str]:
    """Registration only (no build, no adoption): what a brief needs to list the bank, also in shadow."""
    db = getattr(ctx.cfg, "databank", None)
    if db is None or not db.enabled:
        return []
    bk, acts = book(store), []
    if _register(ctx, store, bk, acts):
        store.set_state(K_BANK, bk)
    return acts


def tick(ctx: Any, store: Any, adapter: Any, snapshot: Any, *, new_id: Any, now: float) -> list[str]:
    """Register ready datasets, settle / launch builds, adopt what became ready (cheap: one STATUS read each)."""
    db = getattr(ctx.cfg, "databank", None)
    if db is None or not db.enabled:
        return []
    bk, acts = book(store), []
    _register(ctx, store, bk, acts)
    _builds(ctx, store, bk, acts)
    store.set_state(K_BANK, bk)
    if db.adopt and snapshot is not None:
        for name in sorted(set(bk["registered"]) & set(db.adopt) - set(bk["adopted"])):
            try:
                nid, why = adopt(ctx, store, adapter, snapshot, name, new_id, now)
            except Exception as exc:  # noqa: BLE001 -- a git / adapter failure is retried next tick, never the loop
                store.log("databank_adopt_error", dataset=name, error=repr(exc)[:500])
                acts.append(f"databank: adoption of {name} failed: {exc!r}"[:300])
                continue
            bk = book(store)
            bk["adopted"][name] = nid or f"-: {why}"
            store.set_state(K_BANK, bk)
            store.log("databank_adopted", dataset=name, node_id=nid, why=why)
            acts.append(f"databank: adoption node {nid} campaign ({name} on the incumbent)" if nid
                        else f"databank: {name} not adopted: {why}")
    return acts


def plan(ctx: Any, store: Any, snapshot: Any) -> list[str]:
    """``--dry-run``: what the next tick would do (nothing written, no build, no node)."""
    db = getattr(ctx.cfg, "databank", None)
    if db is None or not db.enabled:
        return []
    bk, out = book(store), []
    disk = on_disk(ctx)
    new = [n for n, st in disk.items() if st == "ready" and not pilot(n) and n not in bk["registered"]]
    out += [f"would register {line(ctx, summary(ctx, n))}" for n in new]
    if db.adopt and snapshot is not None:
        inc = getattr(snapshot, "incumbent_id", None) or ROOT_NODE_ID
        out += [f"would adopt {n} on the incumbent {inc}: config-only node (data.dataset_dir"
                + (f", train.num_workers {db.num_workers}" if db.num_workers else "") + ") + a campaign over "
                + ", ".join(str(r["key"]) for r in db.adopt_space) for n in sorted({*new, *bk["registered"]} & set(db.adopt) - set(bk["adopted"]))]
    out += [f"databank: {n} {st} (not registered: building / failed / pilot)" for n, st in disk.items()
            if n not in bk["registered"] and n not in new]
    out += [f"databank request {n}: {r['state']} (by {', '.join(r['by'])})" for n, r in bk["requests"].items()
            if r["state"] in ("waiting", "submitted")]
    return out + [f"databank usage {usage_tb(ctx, bk):.2f} TB of {db.max_total_tb:g} TB cap"]


# ------------------------------------------------------------------------------------------------ retention
def prune_candidates(ctx: Any, store: Any, now: float) -> list[str]:
    """Ready datasets no open (or confirmed) node trains on, ready more than ``retain_days`` ago: listed, never
    deleted."""
    db = getattr(ctx.cfg, "databank", None)
    if db is None or not db.enabled:
        return []
    used = {dataset_of(ctx, node_dataset(store, n.id)) for n in store.nodes()
            if n.status not in CLOSED_STATUSES or n.status is NodeStatus.CONFIRMED}
    out = []
    for name, st in on_disk(ctx).items():
        if st != "ready" or name in used:
            continue
        ready = _read(_root(ctx) / name / "status.json").get("ready") or ""
        try:
            ts = time.mktime(time.strptime(ready, "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            ts = (_root(ctx) / name / "STATUS").stat().st_mtime
        if (age := (now - ts) / 86400.0) > db.retain_days:
            gb = (_read(_root(ctx) / name / "manifest.json").get("bytes") or 0) / 1e9
            out.append(f"{_root(ctx) / name} ({age:.0f} days, {gb:.0f} GB" + (", pilot" if pilot(name) else "") + ")")
    return out


# ------------------------------------------------------------------------------------------------ v3: run what is banked
K_RUNS = "databank.runs"     # {base epoch: {dataset: node id | "-: why not"}}: once per dataset per base epoch


def _new_id(store: Any) -> str:
    seq = len(store.nodes())
    while store.get_node(f"n{seq:04d}") is not None:
        seq += 1
    return f"n{seq:04d}"


def _config_only(node: Any) -> bool:
    """The node's stored diff touches config files only (a re-rootable change: its config_diff is the whole of it)."""
    from loop.select.intake import _CONFIG_SUFFIXES
    try:
        text = Path(node.diff_path).read_text() if node.diff_path else ""
    except OSError:
        return False
    paths = re.findall(r"^diff --git a/(\S+)", text, re.M)
    return bool(paths) and all(p.endswith(_CONFIG_SUFFIXES) for p in paths)


def config_node(ctx: Any, store: Any, adapter: Any, parent: str, levels: Mapping[str, Any], name: str, now: float,
                notes: str, claim: str) -> tuple[str, str]:
    """(node id, '') of a QUEUED config-only explore node ``parent`` + ``levels`` (family: the one owning
    data.dataset_dir), or ('', why not). Built by the factorial point builder (git only)."""
    from loop.adapters import flatten
    from loop.contracts import Node as StoreNode, Strategy
    from loop.core import factorial as F
    from loop.select.intake import family_of
    p = store.get_node(parent)
    if p is None:
        return "", f"unknown parent {parent}"
    try:
        cur = flatten(adapter.config_at(p.commit)) if hasattr(adapter, "config_at") else {}
    except Exception:  # noqa: BLE001 - an unreadable config: the levels as given
        cur = {}
    levels = dict(levels)
    nw = getattr(ctx.cfg.databank, "num_workers", 0)
    if nw and _num(cur.get("train.num_workers")) and cur["train.num_workers"] < nw:
        levels["train.num_workers"] = nw
    if cur and all(json.dumps(cur.get(k), default=str) == json.dumps(v, default=str) for k, v in levels.items()):
        return "", f"{parent} already has {levels}"
    knobs = [F.Knob(k, cur.get(k), v, "lin") for k, v in levels.items()]
    sha, why = F.build_point(adapter, p.commit, levels, knobs, {}, f"databank-{name}-{int(now)}",
                             getattr(adapter, "config_rel", "config.yaml"))
    if not sha or sha == p.commit:
        return "", why or "the build changed nothing"
    nid, fam = _new_id(store), family_of(ctx, "data.dataset_dir") or "data_curation"
    hyp = HypothesisRecord(id=f"h-{nid}", task_id=ctx.task_id, kind=HypothesisKind.TRIAL, family=fam,
                           mechanism_tag=f"databank_{name}", claim=claim, predictions={PANEL: AxisPrediction(1, 1e-3)},
                           falsification=f"no gain over {parent} on fresh seeds", created_ts=now)
    store.put_hypothesis(hyp)
    store.put_node(StoreNode(id=nid, task_id=ctx.task_id, commit=sha, parent_id=parent, strategy=Strategy.EXPLORE,
                             operator=Operator.DRAFT, family=fam, mechanism_tag=f"databank_{name}",
                             diff_hash=hashlib.sha256(f"{p.commit}\n{sha}".encode()).hexdigest(),
                             status=NodeStatus.QUEUED, hypothesis_id=hyp.id, created_ts=now, updated_ts=now,
                             notes=notes))
    store.set_state(f"config_diff:{nid}", levels)
    return nid, ""


def run_unrun(ctx: Any, store: Any, adapter: Any, led: Any, incumbent: str, now: float) -> list[str]:
    """``databank.run_unrun`` (v3, 10-13: five banked datasets never ran on the current base; a node waited QUEUED on
    an old base since 10-08): (1) every QUEUED config-only node of an OLDER base that trains on a data-bank dataset is
    re-rooted onto the incumbent (a new node with its config diff; the old one VOID, ``re-rooted``); (2) every ready
    registered dataset no node of the current base trains on gets ONE config-only node on the incumbent
    (``data.dataset_dir`` + the ``num_workers`` bump). Once per dataset per base epoch (``K_RUNS``); the nodes are
    screened by the acquisition like any QUEUED idea node. At most ``databank.max_concurrent`` of them screen at once
    (fewer than ``screen_runs`` done runs), the smallest dataset first: two ~500 GB sets never hit the I/O together."""
    from loop.adapters import flatten
    from loop.core.levers import used_datasets
    db = getattr(ctx.cfg, "databank", None)
    if db is None or not db.enabled or not getattr(db, "run_unrun", False) or not led.epoch:
        return []
    inc = led.port(incumbent) or incumbent
    if inc not in led.nodes or led.nodes[inc].base != led.epoch:
        return []
    runs = dict(store.get_state(K_RUNS) or {})
    rkey = f"{led.epoch}#retries"                        # {dataset: reruns spent} beside the per-epoch {dataset: node}
    done, acts, retries = dict(runs.get(led.epoch) or {}), [], dict(runs.get(rkey) or {})
    used, reg = used_datasets(led, store, ctx), book(store)["registered"]
    finished: dict[str, int] = {}
    for r in getattr(led, "runs", ()):
        if r.status == "done":
            finished[r.node] = finished.get(r.node, 0) + 1
    retry: set[str] = set()                              # a node that ended VOID / CRASHED with 0 done runs reruns,
    for name, nid in list(done.items()):                 # at most databank.max_retries times per dataset
        n = None if str(nid).startswith("-") else store.get_node(nid)
        if n is not None and n.status in (NodeStatus.VOID, NodeStatus.CRASHED) and not finished.get(nid, 0) \
                and int(retries.get(name, 0)) < int(getattr(db, "max_retries", 2)):
            retries[name] = int(retries.get(name, 0)) + 1
            retry.add(name)
            del done[name]
            acts.append(f"databank: {name} node {nid} ended {n.status.value} with 0 done runs: retry {retries[name]}")
    retry |= {name for name in retries if name not in done}   # a retry not launched yet (no free slot) stays owed

    def save() -> None:                                  # after EACH node: a partial failure never duplicates nodes
        store.set_state(K_RUNS, {**runs, led.epoch: done, **({rkey: retries} if retries else {})})

    busy = [nid for nid in done.values() if not str(nid).startswith("-") and (n := store.get_node(nid)) is not None
            and n.status not in CLOSED_STATUSES and finished.get(nid, 0) < int(getattr(db, "screen_runs", 3))]
    free = int(getattr(db, "max_concurrent", 1)) - len(busy)
    stranded: dict[str, list[tuple[Any, dict]]] = {}     # dataset -> every stranded node (none with a done run)
    for n in store.nodes(status=[NodeStatus.QUEUED]):
        ln = led.nodes.get(n.id)
        if ln is None or ln.base == led.epoch or finished.get(n.id, 0):
            continue
        cd = flatten(store.get_state(f"config_diff:{n.id}") or {})
        name = dataset_of(ctx, cd.get("data.dataset_dir"))
        if name and _config_only(n):
            stranded.setdefault(name, []).append((n, cd))
    cands = {name: stranded.get(name, []) for name in (*stranded, *reg)}
    order = sorted((float((reg.get(name) or {}).get("bytes") or math.inf), name) for name in cands
                   if name not in done and (name not in used or name in retry) and status(ctx, name) == "ready")
    for _, name in order[:max(0, free)]:              # smallest first; at most max_concurrent screening at once
        if cands[name]:
            for old, cd in cands[name]:                  # ALL stranded nodes of the dataset move to the incumbent
                nid, why = config_node(ctx, store, adapter, inc, cd, name, now,
                                       f"dataset={name} brief=databank reroot_of={old.id}",
                                       f"{old.mechanism_tag} re-rooted from {old.id} (QUEUED on an older base) onto "
                                       f"the incumbent {inc}: {json.dumps(cd, default=str)}")
                done.setdefault(name, nid or f"-: {why}")
                if nid:
                    store.update_node(old.id, status=NodeStatus.VOID, reject_reason=f"re-rooted onto {inc} as {nid}")
                    store.log("databank_reroot", node_id=nid, reroot_of=old.id, dataset=name, parent=inc)
                    acts.append(f"databank: {old.id} ({name}) re-rooted onto {inc} as {nid}")
                    if str(done[name]).startswith("-"):
                        done[name] = nid
                save()
            continue
        nid, why = config_node(ctx, store, adapter, inc, {"data.dataset_dir": reg[name]["path"]}, name, now,
                               f"dataset={name} brief=databank",
                               f"train the incumbent {inc} on the data-bank dataset {line(ctx, reg[name])}")
        done[name] = nid or f"-: {why}"
        store.log("databank_run", node_id=nid, dataset=name, parent=inc, why=why)
        acts.append(f"databank: {name} runs on the incumbent {inc} as {nid}" if nid else
                    f"databank: {name} not run on {inc}: {why}")
        save()
    if done != dict(runs.get(led.epoch) or {}) or retries != dict(runs.get(rkey) or {}):
        save()
    return acts
