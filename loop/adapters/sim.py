"""loop.adapters.sim — an in-process ExecutionAdapter for tests and offline runs.

Everything is a file under ``<ctx.state_dir>/sim``: worktrees are temp dirs holding
a tiny fake repo, 'commits' are content hashes in an object store, jobs are JSON
records advanced one step per ``poll``. Nothing here touches git, SLURM or a network.

THE HIDDEN TRUTH. A run's per-axis score is

    base[axis] + true_effect(config vs root)[axis] + sd[axis] * noise(commit, seed, axis)

``true_effect`` is a planted lever table (``LEVER_EFFECTS``): lowering ``train.lr``,
``train.weight_decay`` or ``aug.strength`` helps ("train weaker"); changing
``model.depth`` / ``model.width`` without an ``INIT_FIX`` marker in ``model.py`` is
catastrophic. ``sd`` is ``ctx.cfg.axis_sd``. Noise is a pure function of
``(commit, seed)`` so identical runs reproduce, and ``shared_frac`` of
its variance can be a per-seed component common to every commit (paired deltas
then get quieter, as on a real benchmark). Tests read the truth through
:meth:`SimAdapter.true_effect` / :meth:`true_panel`.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import math
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from loop.adapters import MISSING, flatten, merge_config, set_dotted
from loop.contracts import (InitCompat, JobHandle, JobStatus, Run, RunStatus, SmokeResult,
                            VoidType, Worktree)
from loop.spec import LoopContext

__all__ = ["SimAdapter", "LEVER_EFFECTS", "BASE_CONFIG", "INIT_MARKER", "SYNTAX_MARKER"]

BASE_CONFIG: dict[str, Any] = {
    "train": {"lr": 1.0e-3, "weight_decay": 0.05, "warmup": 5, "epochs": 10},
    "aug": {"strength": 0.5},
    "model": {"depth": 12, "width": 384},
    "loss": {"temperature": 0.1},
    "eval": {"seed": 0},
}
INIT_MARKER = "INIT_FIX"          # in model.py: the architecture change carries a proper init
SYNTAX_MARKER = "SYNTAX_ERROR"    # anywhere in a .py file: smoke() fails
ARCH_KEYS = ("model.depth", "model.width")
ARCH_BROKEN_EFFECT = -0.08        # panel-scale hit of an architecture change with broken init
ARCH_FIXED_COEF = 0.001

#: key -> effect per unit of log(new/old), before the per-axis scale. Negative = lowering helps.
LEVER_EFFECTS: dict[str, float] = {
    "train.lr": -0.012, "train.weight_decay": -0.008, "train.warmup": -0.002,
    "train.epochs": 0.004, "aug.strength": -0.006, "loss.temperature": 0.001,
}
LOG_CLIP = 1.5
AXIS_SCALE = (1.0, 0.7, 1.3, 0.9, 1.1)      # cycled over ctx.cfg.axes
BASE_SCORE = 0.60
ROOT_FILES = {
    "model.py": "def build(cfg):\n    return cfg\n",
    "train.py": "def train(cfg):\n    return cfg\n",
    "aug.py": "def augment(x):\n    return x\n",
    "eval.py": "def evaluate(model):\n    return 0.0\n",
}
CONFIG_FILE = "config.yaml"
OOM_LADDER = ("expandable_segments", "activation_checkpointing")
OOM_TAIL = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB (sim)"


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def _gauss(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).gauss(0.0, 1.0)


def _dump_config(cfg: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(cfg), sort_keys=True)


class SimAdapter:
    config_rel = "config.yaml"            # the sim's one config file (contracts.ADAPTER_REQUIRED)

    def __init__(self, ctx: LoopContext, *, duration_ticks: int = 3, shared_frac: float = 0.0,
                 gpu_h_per_tick: float = 0.05, clock: Callable[[], float] = time.time) -> None:
        self.ctx, self.duration_ticks, self.shared_frac = ctx, duration_ticks, shared_frac
        self.gpu_h_per_tick, self._clock = gpu_h_per_tick, clock
        self.lever_effects = dict(LEVER_EFFECTS)          # tests may plant their own
        self.root = Path(ctx.state_dir) / "sim"
        for sub in ("objects", "jobs", "worktrees", "runs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._faults: list[dict[str, Any]] = []
        # planted memory truth: a config whose ``oom_key`` exceeds ``oom_above`` OOMs (runs and memory
        # smokes) unless it runs with ``oom_fixed_by`` (or a stronger ladder step); None = never
        self.oom_key, self.oom_above = "train.epochs", None
        self.oom_fixed_by = OOM_LADDER[-1]
        self.smokes_seen: list[dict[str, Any]] = []
        self._root_sha = self._put_object(
            {**ROOT_FILES, CONFIG_FILE: _dump_config(BASE_CONFIG)}, parent=None, message="root")

    # ------------------------------------------------------------------ objects
    def _put_object(self, files: Mapping[str, str], *, parent: str | None, message: str) -> str:
        sha = hashlib.sha1(json.dumps(sorted(files.items())).encode()).hexdigest()[:12]
        path = self.root / "objects" / f"{sha}.json"
        if not path.exists():                              # first writer owns the ancestry
            path.write_text(json.dumps({"files": dict(files), "parent": parent, "message": message}))
        return sha

    def _object(self, sha: str) -> dict[str, Any]:
        try:
            return json.loads((self.root / "objects" / f"{sha}.json").read_text())
        except FileNotFoundError:
            raise KeyError(f"sim: unknown commit {sha!r}") from None

    def _config(self, sha: str) -> dict[str, Any]:
        return yaml.safe_load(self._object(sha)["files"][CONFIG_FILE])

    def config_at(self, commit: str) -> dict[str, Any]:
        return flatten(self._config(commit))

    def _ancestors(self, sha: str | None) -> list[str]:
        chain: list[str] = []
        while sha:
            chain.append(sha)
            sha = self._object(sha)["parent"]
        return chain

    def _merge_base(self, a: str, b: str) -> str:
        theirs = set(self._ancestors(b))
        return next((s for s in self._ancestors(a) if s in theirs), self._root_sha)

    def _read_tree(self, path: str | Path) -> dict[str, str]:
        base = Path(path)
        return {p.relative_to(base).as_posix(): p.read_text() for p in sorted(base.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts
                and not any(part.startswith(".") for part in p.relative_to(base).parts)}

    # ---------------------------------------------------------------- truth
    def axis_scale(self, axis: str) -> float:
        return AXIS_SCALE[self.ctx.cfg.axes.index(axis) % len(AXIS_SCALE)]

    def base_score(self, axis: str) -> float:
        return BASE_SCORE + 0.02 * self.ctx.cfg.axes.index(axis)

    def true_effect(self, commit: str) -> dict[str, float]:
        """Hidden per-axis effect of ``commit`` vs root."""
        new, old = flatten(self._config(commit)), flatten(self._config(self._root_sha))
        fixed = INIT_MARKER in self._object(commit)["files"].get("model.py", "")
        panel = 0.0
        for key, coef in self.lever_effects.items():
            if key in new and key in old and new[key] != old[key] and new[key] > 0 < old[key]:
                panel += coef * max(-LOG_CLIP, min(LOG_CLIP, math.log(new[key] / old[key])))
        for key in ARCH_KEYS:
            if new.get(key) != old.get(key):
                ratio = math.log(new[key] / old[key]) if new.get(key) and old.get(key) else 0.0
                panel += ARCH_FIXED_COEF * ratio if fixed else ARCH_BROKEN_EFFECT
        return {a: panel * self.axis_scale(a) for a in self.ctx.cfg.axes}

    def true_panel(self, commit: str) -> float:
        eff = self.true_effect(commit)
        return sum(self.ctx.cfg.weights[a] * eff[a] for a in self.ctx.cfg.axes)

    def _score(self, commit: str, seed: int) -> dict[str, float]:
        eff, f = self.true_effect(commit), self.shared_frac
        return {a: self.base_score(a) + eff[a] + self.ctx.cfg.axis_sd[a] * (
                    math.sqrt(f) * _gauss("seed", seed, a) + math.sqrt(1 - f) * _gauss(commit, seed, a))
                for a in self.ctx.cfg.axes}

    # ------------------------------------------------------------ worktrees
    def root_commit(self) -> str:
        return self._root_sha

    def prepare_worktree(self, *, parent_commit: str, name: str) -> Worktree:
        path = self.root / "worktrees" / _safe(name)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        for rel, text in self._object(parent_commit)["files"].items():
            (path / rel).parent.mkdir(parents=True, exist_ok=True)
            (path / rel).write_text(text)
        return Worktree(path=str(path), branch=f"sim/{_safe(name)}", base_commit=parent_commit)

    def reap_worktree(self, worktree: Worktree) -> None:
        shutil.rmtree(worktree.path, ignore_errors=True)

    def reap_orphans(self) -> list[str]:
        paths = sorted(str(p) for p in (self.root / "worktrees").iterdir())
        for p in paths:
            shutil.rmtree(p, ignore_errors=True)
        return paths

    def commit(self, worktree: Worktree, *, message: str) -> str:
        return self._put_object(self._read_tree(worktree.path), parent=worktree.base_commit,
                                message=message)

    def diff(self, worktree: Worktree) -> tuple[str, tuple[str, ...], Mapping[str, Any]]:
        old, new = self._object(worktree.base_commit)["files"], self._read_tree(worktree.path)
        changed = tuple(sorted(p for p in set(old) | set(new) if old.get(p) != new.get(p)))
        text = "".join(line for p in changed for line in difflib.unified_diff(
            old.get(p, "").splitlines(True), new.get(p, "").splitlines(True),
            fromfile=f"a/{p}", tofile=f"b/{p}"))
        cfg_old = flatten(yaml.safe_load(old[CONFIG_FILE]))
        cfg_new = flatten(yaml.safe_load(new[CONFIG_FILE])) if CONFIG_FILE in new else {}
        cfg_diff = {k: cfg_new.get(k) for k in sorted(set(cfg_old) | set(cfg_new))
                    if cfg_old.get(k) != cfg_new.get(k)}
        return text, changed, cfg_diff

    def graft(self, parent_commit: str, commit: str, onto: str, name: str) -> str:
        """``commit``'s change over ``parent_commit`` re-applied on ``onto`` (:meth:`graft_port`); ``onto`` itself when
        it already holds the change, '' when it does not apply."""
        sha, how = self.graft_port(parent_commit, commit, onto, name)
        return sha or (onto if how == "contained" else "")

    def graft_port(self, parent_commit: str, commit: str, onto: str, name: str) -> tuple[str, str]:
        """(commit, 'applied') / ('', 'contained') / ('', 'conflict: ...'): 3-way per flattened config key
        (loop.adapters.merge_config: a key ``onto`` holds equal or subsumed is kept), other files whole."""
        if onto == parent_commit:
            return commit, "applied"
        base, theirs, ours = (self._object(s)["files"] for s in (parent_commit, commit, onto))
        out, clash = dict(ours), []
        for path in sorted(set(base) | set(theirs)):
            if path == CONFIG_FILE:
                fb, ft, fo = (flatten(yaml.safe_load(f[CONFIG_FILE])) for f in (base, theirs, ours))
                put, _, bad = merge_config(fb, ft, fo)
                clash += [f"{CONFIG_FILE}:{k}" for k in bad]
                merged = {k: v for k, v in {**fo, **put}.items() if v is not MISSING}
                nested: dict[str, Any] = {}
                for key, val in merged.items():
                    set_dotted(nested, key, val)
                out[path] = _dump_config(nested)
            elif base.get(path) != theirs.get(path):
                if ours.get(path) not in (base.get(path), theirs.get(path)):
                    clash.append(path)
                out.pop(path, None) if theirs.get(path) is None else out.__setitem__(path, theirs[path])
        if clash:
            return "", "conflict: " + ", ".join(clash)
        cfg = lambda f: flatten(yaml.safe_load(f[CONFIG_FILE])) if CONFIG_FILE in f else None   # noqa: E731
        if {**out, CONFIG_FILE: None} == {**ours, CONFIG_FILE: None} and cfg(out) == cfg(ours):
            return "", "contained"
        return self._put_object(out, parent=onto, message=f"loop graft {name}"), "applied"

    def changed_files(self, a: str, b: str) -> list[str]:
        """The paths whose content differs between commits ``a`` and ``b`` (``git diff --name-only``)."""
        fa, fb = self._object(a)["files"], self._object(b)["files"]
        return sorted(p for p in set(fa) | set(fb) if fa.get(p) != fb.get(p))

    def graft_keys(self, commit: str, keys: Sequence[str], onto: str, name: str) -> tuple[str, str]:
        """ONLY config ``keys`` as ``commit`` sets them, onto ``onto`` (a prerequisite port's key patch)."""
        fs, fo = flatten(self._config(commit)), flatten(self._config(onto))
        if any(k not in fs for k in keys):
            return "", f"conflict: {CONFIG_FILE}: {commit} lacks {sorted(k for k in keys if k not in fs)}"
        put = {k: fs[k] for k in keys if fo.get(k, MISSING) != fs[k]}
        if not put:
            return "", "contained"
        nested: dict[str, Any] = {}
        for key, val in {**fo, **put}.items():
            set_dotted(nested, key, val)
        out = {**self._object(onto)["files"], CONFIG_FILE: _dump_config(nested)}
        return self._put_object(out, parent=onto, message=f"loop graft keys {name}"), "applied"

    def merge(self, worktree: Worktree, *, other_commit: str) -> tuple[bool, str]:
        """3-way merge (ours = worktree files, theirs = other_commit, base = nearest common
        ancestor). Config merges per flattened key, other files per whole file. Conflicting
        paths/keys keep OURS and are named in the returned text."""
        base = self._object(self._merge_base(worktree.base_commit, other_commit))["files"]
        theirs, ours = self._object(other_commit)["files"], self._read_tree(worktree.path)
        conflicts: list[str] = []

        def pick(b: Any, o: Any, t: Any, name: str) -> Any:
            if o == t or t == b:
                return o
            if o == b:
                return t
            conflicts.append(name)
            return o

        for path in sorted(set(base) | set(ours) | set(theirs)):
            if path == CONFIG_FILE:
                fb, fo, ft = (flatten(yaml.safe_load(f[CONFIG_FILE])) for f in (base, ours, theirs))
                merged: dict[str, Any] = {}
                for key in sorted(set(fb) | set(fo) | set(ft)):
                    val = pick(fb.get(key), fo.get(key), ft.get(key), f"{path}:{key}")
                    if val is not None:
                        set_dotted(merged, key, val)
                ours[path] = _dump_config(merged)
            else:
                val = pick(base.get(path), ours.get(path), theirs.get(path), path)
                if val is not None:
                    ours[path] = val
                else:
                    ours.pop(path, None)
        wt = Path(worktree.path)
        for path, text in ours.items():
            (wt / path).parent.mkdir(parents=True, exist_ok=True)
            (wt / path).write_text(text)
        for stale in set(self._read_tree(wt)) - set(ours):
            (wt / stale).unlink()
        return (not conflicts), ("" if not conflicts else "conflicts: " + ", ".join(conflicts))

    # --------------------------------------------------------- smoke / init
    def oom_ladder(self, commit: str) -> tuple[str, ...]:
        return OOM_LADDER

    def _fixed(self, mitigation: str, fixed_by: str) -> bool:
        return bool(fixed_by) and mitigation in OOM_LADDER and OOM_LADDER.index(mitigation) >= OOM_LADDER.index(fixed_by)

    def _ooms(self, cfg: Mapping[str, Any], mitigation: str) -> bool:
        v = flatten(cfg).get(self.oom_key)
        return (self.oom_above is not None and isinstance(v, (int, float)) and v > self.oom_above
                and not self._fixed(mitigation, self.oom_fixed_by))

    def smoke(self, worktree: Worktree, *, memory: bool = False, mitigation: str = "") -> SmokeResult:
        tree = self._read_tree(worktree.path)
        self.smokes_seen.append({"path": worktree.path, "memory": memory, "mitigation": mitigation})
        if memory and self._ooms(yaml.safe_load(tree.get(CONFIG_FILE, "")) or {}, mitigation):
            return SmokeResult(False, "smoke OOM", OOM_TAIL)
        for rel, text in tree.items():
            if not rel.endswith(".py"):
                continue
            if SYNTAX_MARKER in text:
                return SmokeResult(False, f"{rel}: planted {SYNTAX_MARKER}", log_tail=text[-200:])
            try:
                compile(text, rel, "exec")
            except SyntaxError as exc:
                return SmokeResult(False, f"{rel}: {exc.msg} (line {exc.lineno})", log_tail=str(exc))
        return SmokeResult(True, "", "sim smoke ok")

    def init_compat(self, worktree: Worktree, *, parent_commit: str) -> InitCompat:
        child = flatten(yaml.safe_load(self._read_tree(worktree.path)[CONFIG_FILE]))
        parent = flatten(self._config(parent_commit))
        changed = [k for k in ARCH_KEYS if child.get(k) != parent.get(k)]
        fixed = INIT_MARKER in self._read_tree(worktree.path).get("model.py", "")
        if changed and not fixed:
            return InitCompat(False, f"architecture change {changed} lacks an {INIT_MARKER} marker",
                              {"architecture_changed": changed, "init_marker": False})
        return InitCompat(True, "", {"architecture_changed": changed, "init_marker": fixed})

    # ----------------------------------------------------------------- jobs
    def inject_fault(self, pattern: str, kind: str, *, at_poll: int = 1, times: int | None = 1,
                     fixed_by: str = "") -> None:
        """Runs whose id matches the fnmatch ``pattern`` fail. ``kind``: ``crash``, ``oom`` (a crash
        with a CUDA OOM tail, skipped by a run launched with the ``fixed_by`` ladder step or a
        stronger one), ``hang`` (no progress: an infra void once ``heartbeat.freeze_min`` has passed)
        or a ``VoidType`` value (``infra``, ``eval``).
        The first ``times`` matching submissions are affected (None = all)."""
        if kind not in ("crash", "hang", "oom") and kind not in {v.value for v in VoidType}:
            raise ValueError(f"unknown fault kind {kind!r}")
        self._faults.append({"pattern": pattern, "kind": kind, "at_poll": at_poll,
                             "left": times, "fixed_by": fixed_by})

    def _job_path(self, job_id: str) -> Path:
        return self.root / "jobs" / f"{job_id}.json"

    def _job_id(self, run: Run) -> str:
        return "sim-" + hashlib.sha1(f"{run.id}|{run.attempt}".encode()).hexdigest()[:10]

    def _handle(self, job: Mapping[str, Any]) -> JobHandle:
        return JobHandle(job["run_id"], job["job_id"], job["output_dir"], job["submitted_ts"])

    def submit(self, *, run: Run, commit: str) -> JobHandle:
        path = self._job_path(self._job_id(run))
        if path.exists():                                   # idempotent per (run.id, attempt)
            return self._handle(json.loads(path.read_text()))
        self._object(commit)                                # unknown commit -> KeyError
        out = self.root / "runs" / _safe(run.id) / f"a{run.attempt}"
        out.mkdir(parents=True, exist_ok=True)
        fault = next((f for f in self._faults if fnmatch.fnmatch(run.id, f["pattern"]) and f["left"] != 0
                      and not (f["kind"] == "oom" and self._fixed(run.mitigation, f.get("fixed_by", "")))), None)
        if fault is not None and fault["left"] is not None:
            fault["left"] -= 1
        if fault is None and self._ooms(self._config(commit), run.mitigation):
            fault = {"kind": "oom", "at_poll": 1}
        job = {"job_id": self._job_id(run), "run_id": run.id, "commit": commit, "seed": run.seed,
               "output_dir": str(out), "submitted_ts": self._clock(), "polls": 0,
               "fault": fault and {"kind": fault["kind"], "at_poll": fault["at_poll"]}, "final": None,
               "mitigation": run.mitigation}
        path.write_text(json.dumps(job))
        return self._handle(job)

    def poll(self, handle: JobHandle) -> JobStatus:
        path = self._job_path(handle.job_id)
        job = json.loads(path.read_text())
        if job["final"] is None:
            job["polls"] += 1
            fault, gpu_h = job["fault"], job["polls"] * self.gpu_h_per_tick
            if fault and fault["kind"] != "hang" and job["polls"] >= fault["at_poll"]:
                crash = fault["kind"] in ("crash", "oom")
                job["final"] = {"state": "crash" if crash else "void", "gpu_h": gpu_h,
                                "void_type": None if crash else fault["kind"],
                                "log_tail": OOM_TAIL if fault["kind"] == "oom" else
                                f"sim: injected {fault['kind']} at poll {job['polls']}"}
            elif fault and fault["kind"] == "hang":         # frozen after freeze_min: cancelled
                if self._clock() - job["submitted_ts"] > 60 * self.ctx.cfg.heartbeat.freeze_min:
                    job["final"] = {"state": "void", "void_type": "infra", "gpu_h": gpu_h,
                                    "log_tail": "sim: frozen, no progress; cancelled"}
            elif job["polls"] >= self.duration_ticks:
                scores = self._score(job["commit"], job["seed"])
                job["final"] = {"state": "done", "gpu_h": gpu_h, "scores": scores, "log_tail": "sim: done",
                                "aggregate": sum(self.ctx.cfg.weights[a] * v for a, v in scores.items())}
                Path(job["output_dir"], "summary.json").write_text(json.dumps(scores))
            path.write_text(json.dumps(job))
        f = job["final"] or {"state": "running", "gpu_h": job["polls"] * self.gpu_h_per_tick}
        return JobStatus(RunStatus(f["state"]), f.get("scores", {}), f.get("aggregate"),
                         VoidType(f["void_type"]) if f.get("void_type") else None, f["gpu_h"],
                         f.get("log_tail", ""))

    def find(self, run: Run) -> JobHandle | None:
        path = self._job_path(self._job_id(run))
        return self._handle(json.loads(path.read_text())) if path.exists() else None

    def cancel(self, handle: JobHandle) -> None:
        path = self._job_path(handle.job_id)
        job = json.loads(path.read_text())
        if job["final"] is None:
            job["final"] = {"state": "void", "void_type": "infra", "log_tail": "sim: cancelled",
                            "gpu_h": job["polls"] * self.gpu_h_per_tick}
            path.write_text(json.dumps(job))
