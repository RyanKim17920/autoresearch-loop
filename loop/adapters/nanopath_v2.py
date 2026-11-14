"""loop.adapters.nanopath_v2 — ExecutionAdapter for nanopath v2: plain git worktrees + the repo's own
sbatch launcher, receipts for idempotent resume, freeze detection, AST/span eval locks (DESIGN §10).
No v1 tooling, trial runner, gate or ledger is used."""

from __future__ import annotations

import ast
import difflib
import getpass
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence

import yaml

from loop import databank as DB
from loop import diagnostics as DG
from loop import infra as INFRA
from loop.adapters import MISSING, flatten, merge_config, set_dotted
from loop.adapters.initcompat import Limits, run_init_check
from loop.contracts import (InitCompat, JobHandle, JobStatus, Run, RunStatus, SmokeResult,
                            SubmissionRejected, VoidType, Worktree, is_oom)
from loop.spec import LoopContext

BASE_SHA = "065b87f57b9d1ee1fafe8df9fcdc2b0e4b2b5d8f"   # crn-knobs, branch loop-v3-base (v3 epoch, 10-12)
RUNS_ROOT = os.environ.get("NANOPATH_RUNS_ROOT") or "/path/to/runs"
WORKTREES_ROOT = os.environ.get("NANOPATH_WORKTREES") or "/path/to/worktrees"
LAUNCHER = "./submit/train_1gpu.sbatch"
CONFIG_REL = "configs/main.yaml"
NO_VENV = ":(exclude).venv"                  # the .venv symlink is not matched by `.venv/`
#: env symlinks a worktree carries (never content): excluded via the repo's info/exclude (shared by every worktree)
#: and unstaged before a diff -- a researcher's own `git add -A` staged the .venv symlink -> surface_violation (10-13)
ENV_LINKS = (".venv", ".venv-*", "venv")
CRASH_STATES = {"FAILED", "OUT_OF_MEMORY"}
#: a FAILED job whose log shows one of these died of the machine, not the mechanism
HARDWARE = ("cuda error", "nccl", "ecc error", "no space left", "disk quota",
            "stale file handle", "input/output error", "bus error")
#: sbatch refused the submission itself: a configuration error, identical on every retry
REJECTED = re.compile(r"Batch job submission failed|invalid account|invalid qos|invalid partition"
                      r"|Invalid account|Invalid qos|Invalid partition", re.I)
TRANSIENT = re.compile(r"socket timed out|slurm_load_jobs|connection (timed out|refused)|"
                       r"unable to contact slurm|temporarily unavailable", re.I)
WAIT_STATES = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "REQUEUED", "SUSPENDED"}
#: the machine was taken away: resubmitted as the same run (a free retry, not an infra void)
FREE_RETRY_STATES = {"PREEMPTED", "NODE_FAIL"}
REQUEUE_STATES = {"PREEMPTED", "REQUEUED", "NODE_FAIL"}   # sacct -D instances that ended in a requeue
SQUEUE_FIELDS = "State:20|,StartTime:25|,Reason:200|,RestartCnt:8"   # restarts: SLURM requeues under the same id
MAX_FREE_RETRIES = 10                        # then an ordinary infra void
SMOKE_MIN = 20                               # smoke job time limit (minutes): SLURM ends a RUNNING smoke there
#: QUEUE time only (PENDING): a started smoke is never killed by it (live req-00424: queued 59m38s, started,
#: and was killed 21 s later by the old queue + run timer)
SMOKE_WAIT_S = 3600
SMOKE_RUN_S = 60 * SMOKE_MIN + 300           # a RUNNING smoke's own wall: SLURM's time limit + slack
SMOKE_POLL_S = 15.0
SMOKE_RUN_STATES = {"RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}
WT_RECENT_S = 3600.0          # a node worktree launched from within this long is never tidied away
#: stray node worktrees tidy() may remove: eval-only ``<node>~<commit10>`` (safe'd to ``_``) and wave baselines
#: ``w###.x``; group 1 / 2 = the node id. Never ``_research``, ``_graft``, ``root``, ``crn-*``
STRAY_WT = re.compile(r"^(?:(n\d+|root)_[0-9a-f]{10}|(w\d+\.x))$")
REAP_TRIES, REAP_WAIT_S = 5, 3.0             # a worktree dir a dying job still writes into: reap again
#: OOM ladder of the old driver (numerics-neutral on every base): the allocator's expandable segments
#: (fragmentation). Activation checkpointing is not in it: before the crn-knobs base (3144ffa / 065b87f, the v3
#: epoch) FlopCounterMode counted the recompute (~+1/3 FLOPs/step) and train.py keys LR/WD/teacher temp/EMA/KDE/
#: freeze on frac = train_flops / max_train_flops, so it changed the schedule; crn-knobs excludes the recompute
EXPANDABLE = "expandable_segments"
ALLOC_ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
#: the v3 runner's last OOM rung (loop.runner.ACT_CKPT; never in :meth:`oom_ladder`, so the old driver never asks for
#: it): a config copy with ``train.activation_checkpointing: true`` -- schedule-neutral on the crn-knobs base, see above
ACT_CKPT, ACT_CKPT_KEY = "activation_checkpointing", "train.activation_checkpointing"
#: a memory smoke keeps the point's own batch shape (the plain smoke's batch 16 never OOMs)
MEMORY_KEEP = ("train.batch_size",)
SMOKE_OVERRIDES: dict[str, Any] = {
    # calibration reserves 6144 + <=12800 tiles of the budget before optimizing
    "train.max_train_samples": 24_000, "train.max_train_flops": 50_000_000_000_000,
    "train.batch_size": 16, "train.num_workers": 0, "train.prefetch_factor": None,
    "train.persistent_workers": False, "train.log_every": 1, "train.save_every": None,
    "train.eval_every": 1_000_000, "train.val_batches": 1, "probe.enabled": False,
}
#: train.py measures one step's FLOPs (FlopCounterMode, first step) and prints them; / the smoke's batch = per sample
FLOPS_RE = re.compile(r"measured_flops_per_step:\s*([\d,]+)")
FLOP_FILES = ("model.py", "train.py")       # with the config: what FLOPs per sample depend on

#: PROBE SMOKE (``slurm.smoke_probe``: off | readout | always; loop.adapters.probe_smoke): after the training smoke, the
#: same SLURM job drives the smoke checkpoint through the LOCKED probe.py's own readout entry points on a few real tiles
#: per probe dataset (~1-2 min), so a readout that crashes in probe.py fails the smoke instead of a full run (live
#: a node: linalg.eigh in probe_features, after 2.4 GPU-h of training). "readout": only when the candidate's diff (vs
#: its worktree base) touches model.py, a ``file::`` span of the evalonly family outside model.py, or a PROBE_KEYS key
SMOKE_PROBE_DEFAULT = "readout"
SMOKE_PROBE_MODES = ("off", "readout", "always")
PROBE_MODEL_FILES = ("model.py",)            # what probe.py imports from the worktree (ViT, probe_transforms)
PROBE_KEYS = ("probe.", "data.mean", "data.std")   # config keys the probe reads: prefix match, so "data.mean" is exact-ish;
#: generic ``model.*`` keys and model.py edits OUTSIDE these functions no longer trigger (+10 min per architecture diff)
#: model.py functions the locked probe.py calls (model.encode_image / model.probe_features / probe_transforms) or
#: readout spans (default of slurm.smoke_probe_funcs). NOT the generic ``forward``: every Block / ViT edit touches one
PROBE_FUNCS = ("encode_image", "probe_features", "probe_transforms", "forward_features", "get_intermediate_layers",
               "eval_readout")
PROBE_SMOKE_MIN = 10                          # extra SLURM minutes for the probe leg (measured ~80 s on CPU)
PROBE_SAVE_EVERY = 10 ** 9                    # train.save_every: no periodic save; latest.pt once, after the readout fit
PROBE_SMOKE_OK = "probe_smoke: ok"
PROBE_SMOKE_FAILED = "probe_smoke: FAILED"
PROBE_SMOKE_SCRIPT = Path(__file__).with_name("probe_smoke.py")


#: eval-only (loop.evalonly): outside the readout spans a candidate may only add / drop fitted buffers whose
#: prefixes load_pretrained tolerates as missing (so the full retrain it graduates to still loads), their
#: robust_norm-off flag defaults (``self.rn_fitted = self.pf_fitted = False``), or comments
EO_BUFFER = re.compile(r"""^\s*self\.register_buffer\(\s*["'](rn|pf|sb|ct)_\w*["']""")
EO_TRIVIAL = re.compile(r"^\s*((#.*)?|(self\.(rn|pf|sb|ct)_\w+\s*=\s*)+(False|True|None)\s*(#.*)?)$")
#: v3-c5 I5 (``evalonly.readout_keys``): a literal constant (``self.readout_k = 4`` / ``SHRINK = 0.1``) outside the
#: spans -- eligible only when its name is read nowhere but inside the readout spans (``_readout_only``)
_EO_LIT = (r"""[-+]?[\d.eE_]+|True|False|None|'[^'\\]*'|"[^"\\]*"|\(\s*(?:[-+]?[\d.eE_]+\s*,?\s*)*\)"""
           r"""|\[\s*(?:[-+]?[\d.eE_]+\s*,?\s*)*\]""")
EO_CONST = re.compile(rf"^\s*(?:self\.)?([A-Za-z_]\w*)\s*(?::\s*[\w\[\], .]+)?=\s*(?:{_EO_LIT})\s*(#.*)?$")


#: an EO_BUFFER / EO_TRIVIAL line's declared names (``self.rn_fitted = self.pf_fitted = False`` -> both)
EO_NAMES = re.compile(r"""(?:self\.|["'])((?:rn|pf|sb|ct)_\w*)""")


def _eo_decls(lines: Sequence[str | None]) -> dict[str, set[str]]:
    """name -> its declarations outside the spans (an EO_BUFFER line's call without spaces / comment, an EO_TRIVIAL
    assignment's literal): an EXISTING buffer / attribute whose declaration changed changes its initial value."""
    out: dict[str, set[str]] = {}
    for l in lines:
        if l is None:
            continue
        code = l.split("#", 1)[0]
        if EO_BUFFER.match(l):
            for n in EO_NAMES.findall(code)[:1]:
                out.setdefault(n, set()).add("buffer:" + re.sub(r"\s+", "", code))
        elif EO_TRIVIAL.match(l) and code.strip():
            val = code.rsplit("=", 1)[-1].strip()
            for n in EO_NAMES.findall(code):
                out.setdefault(n, set()).add("attr:" + val)
    return out


def train_reachable(sources: Sequence[str | None], spans: Sequence[str], train: Sequence[str]) -> set[str]:
    """The ``Qual.meth`` spans TRAINING can execute (v3-c5b M3): ``train`` (``evalonly.train_spans``) plus every span
    method they reach through ``self.<m>`` / ``self(...)`` (-> forward) in any of ``sources`` (parent and child).
    ``not self.training`` is NOT not-training: train.py's EMA teacher runs ``forward`` in eval mode during training,
    so every branch of a reachable span runs on the teacher. Only spans outside this set are readout-only."""
    train = [t for t in train if t in spans]
    if not train:
        return set()
    by_cls: dict[tuple[str, str], dict[str, str]] = {}      # (file, class) -> method -> span
    for l in spans:
        rel, _, tgt = l.partition("::")
        if "..." not in tgt and "." in tgt:
            cls, meth = tgt.rsplit(".", 1)
            by_cls.setdefault((rel, cls), {})[meth] = l
    reach = set(train)
    for src in sources:
        for (rel, cls), meths in by_cls.items():
            s = _span(src, f"{rel}::{cls}")
            if s is None or s[2] is None:
                continue
            calls: dict[str, set[str]] = {}
            for fn in s[2].body:
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    got = calls.setdefault(fn.name, set())
                    for n in ast.walk(fn):
                        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
                            got.add(n.attr)
                        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "self":
                            got.add("forward")
            todo = [t.rsplit(".", 1)[1] for t in reach if t in meths.values()]
            seen = set(todo)
            while todo:
                for m in calls.get(todo.pop(), ()):
                    if m not in seen:
                        seen.add(m)
                        todo.append(m)
            reach |= {meths[m] for m in seen if m in meths}
    return reach


def _not_training(test: ast.AST) -> bool:
    """``not self.training`` (alone, or one operand of an ``and``): a body that never runs while training."""
    def nt(t: ast.AST) -> bool:
        return (isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not) and isinstance(t.operand, ast.Attribute)
                and t.operand.attr == "training" and isinstance(t.operand.value, ast.Name)
                and t.operand.value.id == "self")
    return nt(test) or (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) and any(map(nt, test.values)))


class _TrainView(ast.NodeTransformer):
    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if _not_training(node.test):
            node.body = [ast.Pass()]
        return node


def train_view(source: str | None, lock: str) -> str | None:
    """A def span as TRAINING executes it: its AST with every ``if not self.training ...`` body blanked (v3-c5 I5,
    ``evalonly.train_spans``: a readout edit whose train view differs is a training-path change)."""
    import copy
    s = _span(source, lock)
    return None if s is None or s[2] is None else ast.dump(_TrainView().visit(copy.deepcopy(s[2])))


SHIM_NAME = "resume_shim.pt"
#: the resume checkpoint with the child's default init for added buffers, minus dropped ones (strict load)
SHIM = r'''
import importlib.util, json, os, sys
from pathlib import Path
import torch, yaml
wt, src, dst = sys.argv[1:4]
cfg = yaml.safe_load((Path(wt) / "configs/main.yaml").read_text())
spec = importlib.util.spec_from_file_location("loop_shim_model", Path(wt) / "model.py")
mod = importlib.util.module_from_spec(spec)
sys.path.insert(0, wt)
spec.loader.exec_module(mod)
fresh = mod.ViT(variant=cfg["model"]["type"], drop_path_rate=float(cfg["dino"]["drop_path_rate"])).state_dict()
ck = torch.load(src, map_location="cpu", weights_only=False)
out = {}
for key in ("model", "model_ema"):
    old = ck[key]
    ck[key] = {k: old[k] if k in old and old[k].shape == v.shape else v for k, v in fresh.items()}
    out[key] = {"added": sorted(set(fresh) - set(old)), "dropped": sorted(set(old) - set(fresh))}
torch.save(ck, dst + ".tmp")
os.replace(dst + ".tmp", dst)
print(json.dumps(out))
'''


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _read(path: Path) -> str | None:
    return path.read_text() if path.is_file() else None


def _yaml_flat(text: str | None) -> dict[str, Any]:
    try:
        return flatten(yaml.safe_load(text or "") or {})
    except yaml.YAMLError:
        return {}


_YKEY = re.compile(r"^( *)([A-Za-z0-9_.-]+):(?:\s|$)")


def _yaml_keys(lines: Sequence[str]) -> dict[str, tuple[int, int]]:
    """dotted key -> its [start, end) line range in a block-style YAML file (a nested mapping / block list
    included; trailing comments and blanks belong to the next key). Keys inside list items are not addressed."""
    out: dict[str, tuple[int, int]] = {}
    stack: list[tuple[int, str]] = []
    code = [i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#")]
    for k, i in enumerate(code):
        m = _YKEY.match(lines[i])
        if m is None:
            continue
        ind = len(m.group(1))
        while stack and stack[-1][0] >= ind:
            stack.pop()
        stack.append((ind, m.group(2)))
        end = i + 1
        for j in code[k + 1:]:
            dent = len(lines[j]) - len(lines[j].lstrip(" "))
            if dent < ind or dent == ind and not lines[j].lstrip().startswith("- "):   # (a block list may sit at ind)
                break
            end = j + 1
        out[".".join(s for _, s in stack)] = (i, end)
    return out


def yaml_apply(text: str, src: str, put: Mapping[str, Any]) -> str | None:
    """``text`` with each key of ``put`` set as ``src`` writes it -- its lines verbatim (comments, layout), a new
    key after the key it follows in ``src`` (with the comment lines just above it), a :data:`MISSING` one
    deleted; None when a key cannot be placed or the result does not parse to ``text``'s keys plus ``put``."""
    lines, sl = text.splitlines(keepends=True), src.splitlines(keepends=True)
    if sl and not sl[-1].endswith("\n"):
        sl[-1] += "\n"
    skeys = _yaml_keys(sl)
    for key, val in put.items():
        tk = _yaml_keys(lines)
        if val is MISSING:
            if key in tk:
                a, b = tk[key]
                del lines[a:b]
            continue
        if key not in skeys:
            return None
        a, b = skeys[key]
        if key in tk:
            ta, tb = tk[key]
            lines[ta:tb] = sl[a:b]
            continue
        while a > 0 and sl[a - 1].lstrip().startswith("#") and sl[a - 1] not in lines:
            a -= 1
        parent, prev = key.rpartition(".")[0], None
        for k2, (a2, _) in skeys.items():
            if k2.rpartition(".")[0] == parent and a2 < skeys[key][0] and k2 in tk:
                prev = k2 if prev is None or a2 > skeys[prev][0] else prev
        at = tk[prev][1] if prev else (tk[parent][0] + 1 if parent in tk else (len(lines) if not parent else None))
        if at is None:
            return None
        lines[at:at] = sl[a:b]
    out = "".join(lines)
    want = {k: v for k, v in {**_yaml_flat(text), **put}.items() if v is not MISSING}
    return out if _yaml_flat(out) == want else None


def _span(source: str | None, lock: str) -> tuple[int, int, ast.AST | None] | None:
    """The 0-based line range a ``file::`` entry pins (and its def for a ``Qual.name`` entry). None (the lock does
    not resolve: a violation unless the canonical copy is unresolvable the same way) also when the pin is AMBIGUOUS:
    a span whose start string is on more than one line (a decoy copy above the real one would take the pin), or a
    ``Qual.name`` whose name is defined / rebound more than once in its scope (python runs the LAST binding: a
    second ``def load_model`` below the locked one would replace it at import)."""
    if source is None:
        return None
    target = lock.split("::", 1)[1]
    if "..." in target:
        start, end = target.split("...", 1)
        lines = source.splitlines()
        hits = [k for k, ln in enumerate(lines) if start in ln]
        if len(hits) != 1:
            return None
        i = hits[0]
        j = next((k for k in range(i, len(lines)) if end in lines[k]), None)
        return None if j is None else (i, j, None)
    try:
        node: ast.AST = ast.parse(source)
    except SyntaxError:
        return None
    for part in target.split("."):
        defs = [n for n in ast.walk(node) if n is not node and getattr(n, "name", None) == part
                and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        if len(defs) != 1 or _rebound(node, defs[0], part):
            return None
        node = defs[0]
    return node.lineno - 1, node.end_lineno - 1, node


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _rebound(scope: ast.AST, keep: ast.AST, name: str) -> bool:
    """``name`` is bound again in the scope that holds the def ``keep`` (its own statements, not nested defs): an
    assignment, an annotated / augmented assignment, an import alias, a for / with / walrus target or a ``del``."""
    up = {id(c): n for n in ast.walk(scope) for c in ast.iter_child_nodes(n)}
    parent = up.get(id(keep), scope)
    while parent is not scope and not isinstance(parent, (ast.Module, *_SCOPES)):   # an if / try / with block
        parent = up.get(id(parent), scope)
    todo = list(ast.iter_child_nodes(parent))
    while todo:
        n = todo.pop()
        if n is keep:
            continue
        if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, (ast.Store, ast.Del)):
            return True
        if isinstance(n, ast.alias) and (n.asname or n.name.split(".")[0]) == name:
            return True
        if isinstance(n, _SCOPES):
            continue                    # a nested def's own body is another scope (its name was counted above)
        todo.extend(ast.iter_child_nodes(n))
    return False


def locked_source(source: str | None, lock: str) -> str | None:
    """What a lock pins in one file's source (a def's AST, a span's stripped lines); None when absent."""
    if (s := _span(source, lock)) is None:
        return None
    i, j, node = s
    return ast.dump(node) if node is not None else "\n".join(l.strip() for l in source.splitlines()[i:j + 1])


def outside_spans(source: str | None, locks: Sequence[str]) -> list[str | None]:
    """The file's lines with every pinned span blanked: what changed OUTSIDE them."""
    lines: list[str | None] = list((source or "").splitlines())
    for lock in locks:
        if (s := _span(source, lock)) is not None:
            lines[s[0]:s[1] + 1] = [None] * (s[1] + 1 - s[0])
    return [l for l in lines if l is not None]


SLURM_BINARIES = frozenset({"sbatch", "srun", "scancel", "squeue", "sacct", "salloc", "scontrol", "sacctmgr", "sinfo"})


def _is_slurm(argv) -> bool:
    argv = list(argv)
    prog = os.path.basename(str(argv[0])) if argv else ""
    return prog in SLURM_BINARIES or prog.endswith(".sbatch")


PROTECT_WAITING = "PENDING,REQUEUED,REQUEUE_HOLD,REQUEUE_FED,PREEMPTED,SUSPENDED"


class WorktreeLock:
    """The adapter's worktree lock (v3-c8a B5): ``with lock:`` is EXCLUSIVE (worktree remove / prune / tidy:
    reentrant for its owner thread), ``with lock.shared():`` is SHARED -- ``git worktree add`` of DIFFERENT
    destinations run at once (live 10-14: ~25 s per add on virtiofs, serial under one lock: 193 s of launch.submit
    for ~8 new nodes). A shared holder may take it exclusively when it is the only reader (the add's prune
    fallback); the exclusive owner may take it shared (nested)."""

    def __init__(self) -> None:
        self._c = threading.Condition(threading.Lock())
        self._owner: int | None = None
        self._depth = 0
        self._readers: dict[int, int] = {}
        self._writers_waiting = 0       # v3 review S8: writer preference -- a stream of shared holders cannot starve an exclusive one

    def acquire(self) -> None:
        me = threading.get_ident()
        with self._c:
            if self._owner == me:
                self._depth += 1
                return
            self._writers_waiting += 1
            try:
                while self._owner is not None or any(n for t, n in self._readers.items() if t != me):
                    self._c.wait()
            finally:
                self._writers_waiting -= 1
                self._c.notify_all()
            self._owner, self._depth = me, 1

    def release(self) -> None:
        with self._c:
            if self._owner != threading.get_ident():
                raise RuntimeError("WorktreeLock released by a thread that does not own it")
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
                self._c.notify_all()

    def __enter__(self) -> "WorktreeLock":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def shared(self) -> Any:
        from contextlib import contextmanager

        @contextmanager
        def cm() -> Any:
            me = threading.get_ident()
            with self._c:
                # a NEW shared holder waits while a writer waits; one that already holds it (nested) never does (deadlock)
                while (self._owner is not None and self._owner != me) or (
                        self._writers_waiting > 0 and self._owner != me and not self._readers.get(me)):
                    self._c.wait()
                self._readers[me] = self._readers.get(me, 0) + 1
            try:
                yield self
            finally:
                with self._c:
                    self._readers[me] -= 1
                    if not self._readers[me]:
                        del self._readers[me]
                    self._c.notify_all()
        return cm()

#: v3 review M1: the init-compat probe and the smoke run with the hub OFFLINE: a diff that downloads weights fails to construct
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


#: pool_capacity: each sacctmgr / squeue call's timeout, and how long a failed probe is not repeated (a hung slurmdbd
#: would otherwise cost this many seconds on EVERY tick; the declared slots stand meanwhile)
CAP_PROBE_TIMEOUT_S = 20
CAP_PROBE_BACKOFF_S = 300.0


class SubprocessRunner:
    """stdin is always /dev/null: the launcher must never see a terminal."""

    def run(self, argv, *, cwd=None, timeout=None, env=None):
        if os.environ.get("LOOP_FORBID_SLURM") and _is_slurm(argv):
            raise RuntimeError(f"REAL SLURM CALLED FROM TESTS: {list(argv)!r} (LOOP_FORBID_SLURM=1)")
        try:
            p = subprocess.run(list(argv), cwd=cwd, env={**os.environ, **(env or {})},
                               stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, "", f"timeout after {timeout}s"
        except OSError as exc:
            return 127, "", str(exc)
        return p.returncode, p.stdout, p.stderr


def _project_value(ctx: Any, attr: str, default: str) -> str:
    """``loop.project.<attr>`` of the task (loop.project.ProjectCfg) when set, else ``default`` (no project block, or
    the field left empty: nanopath's literal -- parity)."""
    return str(getattr(getattr(getattr(ctx, "cfg", None), "project", None), attr, "") or "") or default


class NanopathV2Adapter:
    # multi-project (Stage 1): the project constants as overridable class attributes (a subclass -- e.g.
    # loop.adapters.nanobrain -- replaces them); base_sha / runs_root / worktrees_root / config_rel are read from the
    # task's loop.project block first (``_project_value``), these are the fallbacks. Values == the module literals.
    BASE_SHA, RUNS_ROOT, WORKTREES_ROOT, CONFIG_REL = BASE_SHA, RUNS_ROOT, WORKTREES_ROOT, CONFIG_REL
    LAUNCHER, SMOKE_OVERRIDES, FLOPS_RE, MEMORY_KEEP = LAUNCHER, SMOKE_OVERRIDES, FLOPS_RE, MEMORY_KEEP
    venv_root: Path | None = None       # a worktree's .venv link target (None = <repo>/.venv); nanobrain: loop.project.python

    def __init__(self, ctx: LoopContext, *, runner: Any = None,
                 clock: Callable[[], float] = time.time, repo: str | None = None,
                 runs_root: str | None = None, worktrees_root: str | None = None) -> None:
        self.ctx, self.runner, self._clock = ctx, runner or SubprocessRunner(), clock
        self.repo = repo or ctx.spec.repo
        self.config_rel = _project_value(ctx, "config_rel", self.CONFIG_REL)
        self.base_sha = _project_value(ctx, "base_sha", self.BASE_SHA)
        runs_root = runs_root or _project_value(ctx, "runs_root", self.RUNS_ROOT)
        worktrees_root = worktrees_root or _project_value(ctx, "worktrees_root", self.WORKTREES_ROOT)
        self.state = Path(ctx.state_dir)
        self._tag = hashlib.sha1(str(self.state.resolve()).encode()).hexdigest()[:6]
        # namespaced by state dir: two loops (or a test) never share a run dir, worktree or branch
        self.runs, self.worktrees = Path(runs_root) / f"lp{self._tag}", Path(worktrees_root) / f"lp{self._tag}"
        self.freeze_s = 60.0 * ctx.cfg.heartbeat.freeze_min
        # smoke jobs hold SLURM slots too: in flight per pool, placed by the driver's RunBook
        self.smokes: Counter[str] = Counter()
        self.pool_picker: Callable[[], str] | None = None
        # resume-on-requeue (loop.resume sets these): every run, or only these run ids (its calibration)
        self.resume_ok, self.resume_runs = False, set()
        self._cap_retry_at = 0.0                    # pool_capacity: no probe before this time after a failed one
        # runs never resumed whatever resume_ok says (v3: TEST-stage runs restart from step 0 when preempted)
        self.no_resume: Callable[[Run], bool] = lambda _run: False
        self._smoke_lock = threading.Lock()
        self._cap_cache: tuple[float, tuple[str, str]] | None = None     # smoke_capacity: (when, verdict), 30 s
        #: the smoke hold after an infra failure (loop.infra) and the preflight run before every smoke is submitted
        self.smoke_gate = INFRA.SmokeGate(self.state / "smoke_infra.json", clock=clock)
        self.preflight: Callable[[str, str], str] | None = INFRA.preflight
        # submit() is thread-safe (worktree add under _wt_lock, per-run receipts, no store access): RunBook may
        # run a tick's submits on slurm.submit_workers threads
        self.concurrent_submit = True
        self._smoke_names: set[str] = set()     # job names of smokes a live smoke() call waits on
        # worktree remove / prune: one at a time and never beside an add (the background prune pass, loop.prune,
        # runs tidy() while researchers create worktrees); adds of different destinations share the lock (v3-c8a B5:
        # one at a time per destination, ``_dest_lock``); a node worktree used within WT_RECENT_S is never tidied
        self._wt_lock = WorktreeLock()
        self._dest_locks: dict[str, threading.RLock] = {}
        self._dest_meta = threading.Lock()
        self._exclude_lock = threading.Lock()
        self._wt_used: dict[str, float] = {}
        self._wt_alerted: set[str] = set()     # _heal_worktree: one ALERT per (worktree, have, want)
        self._shim_lock = threading.Lock()      # state/evalonly/shim.py: one writer (tmp + os.replace)
        # `file::` entries: surface locks and family spans (e.g. eval_readout's eval-time code)
        # LAUNCH-TIME LOCK GUARD (review M1): every launched commit's surface.locked files / spans must equal this
        # canonical commit's (the base epoch's root; a test repo points it at its own base)
        self.lock_ref = self.base_sha
        self._lock_ok: dict[str, list[str]] = {}
        self.locks = tuple(dict.fromkeys(l for l in (*ctx.spec.surface.locked,
                                                     *(p for r in ctx.cfg.families.values() for p in r.paths))
                                         if "::" in l))

    # ------------------------------------------------------------- git
    def _git(self, path: str | Path, *args: str, timeout: float = 120) -> tuple[int, str, str]:
        ident = ("-c", "user.name=loop", "-c", "user.email=loop@localhost")
        return self.runner.run(["git", "-C", str(path), *ident, *args], timeout=timeout)

    def _show(self, path: str, commit: str, rel: str) -> str | None:
        rc, out, _ = self._git(path, "show", f"{commit}:{rel}")
        return out if rc == 0 else None

    def root_commit(self) -> str:
        return self.base_sha

    def base_config(self) -> dict[str, Any]:
        """The root commit's flattened config (intake compares EFFECTIVE configs against it)."""
        return _yaml_flat(self._show(self.repo, self.base_sha, self.config_rel))

    def file_at(self, commit: str, rel: str) -> str | None:
        """``rel``'s text at ``commit`` (None when absent / unreadable)."""
        return self._show(self.repo, commit, rel)

    def config_at(self, commit: str) -> dict[str, Any]:
        """``commit``'s flattened config ({} when unreadable): loop.shift compares incumbents with it."""
        return _yaml_flat(self._show(self.repo, commit, self.config_rel))

    def _dest_lock(self, dest: Path) -> Any:
        """One lock per worktree destination: two threads never add (or check-and-add) the same one at once."""
        with self._dest_meta:
            return self._dest_locks.setdefault(str(dest), threading.RLock())

    def _add_worktree(self, dest: Path, commit: str, *how: str) -> None:
        # `worktree prune` only when the add fails (a stale registration of dest): with ~400 registered worktrees
        # each prune costs ~3.5 s, and a wave point / node worktree paid several (live 10-13 tick timing). The add
        # holds the lock SHARED (adds of other destinations run beside it), the prune + retry EXCLUSIVELY
        with self._dest_lock(dest):
            with self._wt_lock.shared():
                rc, out, err = self._git(self.repo, "worktree", "add", *how, str(dest), commit, timeout=600)
            if rc != 0:
                with self._wt_lock:
                    self._git(self.repo, "worktree", "prune")
                    rc, out, err = self._git(self.repo, "worktree", "add", *how, str(dest), commit, timeout=600)
            if rc != 0:
                raise RuntimeError(f"git worktree add {dest} failed (rc={rc}): {(err or out).strip()[-300:]}")
            dest.mkdir(parents=True, exist_ok=True)
            self._exclude_env_links(dest)
            if not (dest / ".venv").exists():
                (dest / ".venv").symlink_to(self.venv_root or Path(self.repo) / ".venv")

    def _exclude_env_links(self, wt: Path) -> None:
        """``ENV_LINKS`` in the info/exclude every worktree of the repo reads (``git rev-parse --git-path
        info/exclude``: the common dir's), so `git add -A` / `git status` never see the .venv symlink."""
        rc, out, _ = self._git(wt, "rev-parse", "--git-path", "info/exclude")
        if rc != 0 or not out.strip():
            return
        path = Path(out.strip())
        path = path if path.is_absolute() else Path(wt) / path
        try:
            with self._exclude_lock:                      # concurrent adds: one appender
                have = set(path.read_text().splitlines()) if path.is_file() else set()
                miss = [p for p in ENV_LINKS if p not in have]
                if miss:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("a") as f:
                        f.write(("" if not have or path.read_text().endswith("\n") else "\n") + "\n".join(miss) + "\n")
        except OSError:
            pass

    def prepare_worktree(self, *, parent_commit: str, name: str) -> Worktree:
        dest, branch = self.worktrees / "_research" / _safe(name), f"loop/{self._tag}-{_safe(name)}"
        for _ in range(REAP_TRIES):                       # idempotent: a restart re-issues the same name
            self.reap_worktree(Worktree(str(dest), "", ""))
            if not dest.exists():
                break
            time.sleep(REAP_WAIT_S)                       # a dying job of the old process still writes there
        else:
            raise RuntimeError(f"{dest} still exists after {REAP_TRIES} reaps (a live smoke job writes into it?)")
        self._add_worktree(dest, parent_commit, "-B", branch)
        return Worktree(path=str(dest), branch=branch, base_commit=parent_commit)

    def reap_worktree(self, worktree: Worktree) -> None:
        # double force: a worktree left LOCKED by a killed `worktree add` (SIGTERM on deploy) goes too
        with self._wt_lock:
            rc, _, _ = self._git(self.repo, "worktree", "remove", "--force", "--force", worktree.path)
            shutil.rmtree(worktree.path, ignore_errors=True)
            if rc != 0:                                   # not removed as a worktree: clear a stale registration
                self._git(self.repo, "worktree", "prune")

    def tidy(self, commits: Mapping[str, str], requests: Sequence[str], nodes: Sequence[str],
             dry_run: bool = False, halt: Callable[[], bool] = lambda: False,
             keep: Collection[str] | None = None) -> dict[str, int]:
        """Git housekeeping (loop.prune): ``worktree prune``; every node commit kept reachable by a lightweight
        tag ``loop/<tag>/<node id>``; then the research branches ``loop/<tag>-loop-<request id>`` of
        ``requests`` (and a research worktree still registered for one) are deleted -- only when the branch
        tip is not a node commit or that node's tag now holds it -- and the node worktrees of ``nodes`` are
        removed (their commits stay tagged; a later launch re-adds one). ``dry_run`` only counts. Runs in the
        background prune thread: each removal holds the worktree lock on its own (a researcher's worktree add
        waits seconds, not the whole pass), a node worktree used in the last WT_RECENT_S is kept, and ``halt()``
        (a stop) ends the pass between two git commands, never inside one. ``keep`` (node ids; None: skipped) also
        removes the STRAY worktrees no node list names (:meth:`_stray_worktrees`: eval-only ``<node>_<c10>`` and
        ``w###.x``; 10-13: 108 + 1 + 1 piled up)."""
        with self._wt_lock:
            self._git(self.repo, "worktree", "prune")
        rc, out, _ = self._git(self.repo, "for-each-ref", "--format=%(refname) %(objectname)", f"refs/tags/loop/{self._tag}")
        tags = dict(line.split() for line in out.splitlines() if line.count(" ") == 1) if rc == 0 else {}
        for nid, c in commits.items():
            ref = f"refs/tags/loop/{self._tag}/{_safe(nid)}"
            if tags.get(ref) != c and c.strip("0") and self._git(self.repo, "update-ref", ref, c)[0] == 0:
                tags[ref] = c
        held, node_commits = set(tags.values()), set(commits.values())
        rc, out, _ = self._git(self.repo, "for-each-ref", "--format=%(refname:short) %(objectname)",
                               f"refs/heads/loop/{self._tag}-loop-*")
        tips = dict(line.split() for line in out.splitlines() if line.count(" ") == 1) if rc == 0 else {}
        stale = {f"loop/{self._tag}-{_safe('loop-' + r)}" for r in requests}
        branches = [b for b, tip in tips.items() if b in stale and (tip not in node_commits or tip in held)]
        recent = time.time() - WT_RECENT_S
        wts = [w for n in nodes if ((w := self.worktrees / _safe(n)) / ".git").exists() and self._wt_used.get(n, 0.0) < recent]
        if not dry_run:
            for b in branches:
                if halt():
                    break
                with self._wt_lock:
                    self.reap_worktree(Worktree(str(self.worktrees / "_research" / b.split(f"{self._tag}-", 1)[1]), "", ""))
                    self._git(self.repo, "branch", "-D", b)
            for w in wts:
                if halt():
                    break
                with self._wt_lock:
                    if self._wt_used.get(w.name, 0.0) < recent:         # not re-used since the plan
                        self.reap_worktree(Worktree(str(w), "", ""))
        stray, skipped = self._stray_worktrees(keep, recent) if keep is not None else ([], 0)
        removed = 0
        if not dry_run:
            for w in stray:
                if halt():
                    break
                with self._wt_lock:
                    if self._wt_used.get(w.name, 0.0) < recent:
                        rc, _, _ = self._git(self.repo, "worktree", "remove", "--force", "--force", str(w))
                        if rc != 0:
                            self._git(self.repo, "worktree", "prune")
                        shutil.rmtree(w, ignore_errors=True)
                        removed += not w.exists()
        return {"tagged": len(held), "branches": len(branches), "worktrees": len(wts),
                "stray": len(stray), "stray_removed": removed, "stray_kept": skipped}

    def _stray_worktrees(self, keep: Collection[str], recent: float) -> tuple[list[Path], int]:
        """(removable, kept) worktree dirs the node list never names: eval-only ``<node>_<commit10>``
        (``n0094_da22bf8472``, ``root_<c10>``) and wave baselines ``w###.x``. Kept: its node in ``keep`` (protected,
        the incumbent, a node with a RUNNING / SUBMITTING run), used or created within WT_RECENT_S, or the cwd
        of one of our live SLURM jobs (``squeue %Z``; an squeue failure keeps every one). ``_research``,
        ``_graft``, ``root`` and ``crn-*`` never match."""
        try:
            dirs = sorted(p for p in self.worktrees.iterdir() if STRAY_WT.match(p.name) and p.is_dir())
        except OSError:
            return [], 0
        if not dirs:
            return [], 0
        rc, out, _ = self.runner.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%Z"], timeout=60)
        if rc != 0:
            return [], len(dirs)
        cwds = {os.path.realpath(l.strip()) for l in out.splitlines() if l.strip()}
        go = []
        for p in dirs:
            m = STRAY_WT.match(p.name)
            node = m.group(1) or m.group(2)
            real = os.path.realpath(p)
            try:
                age_ts = max(p.stat().st_mtime, (p / ".git").stat().st_mtime if (p / ".git").exists() else 0.0)
            except OSError:
                age_ts = 0.0
            if node in keep or max(age_ts, self._wt_used.get(p.name, 0.0)) >= recent \
                    or any(c == real or c.startswith(real + os.sep) for c in cwds):
                continue
            go.append(p)
        return go, len(dirs) - len(go)

    def reap_orphans(self) -> list[str]:
        paths = sorted(str(p) for p in (self.worktrees / "_research").glob("*"))
        for p in paths:
            self.reap_worktree(Worktree(p, "", ""))
        return paths

    def cancel_smokes(self, name: str = "") -> list[str]:
        """scancel this loop's smoke jobs for the research worktree ``name`` ('' = every one no live smoke()
        call of this process waits on: an orphan): a restart orphans them, and a live one recreates ``slurm/``
        in the worktree a re-issue must re-create."""
        rc, out, _ = self.runner.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%j"], timeout=60)
        pre = f"lp{self._tag}-smoke-{_safe(name) if name else ''}"
        ids = [i for i, _, j in (line.strip().partition("|") for line in out.splitlines() if rc == 0)
               if j == pre or j.startswith(pre + "-") or not name and j.startswith(pre) and j not in self._smoke_names]
        if ids:
            self.runner.run(["scancel", *ids], timeout=60)
        return ids

    # ------------------------------------------------ smoke capacity (10-22: 4 Opus sessions lost to a starved smoke queue)
    def _smoke_notes_path(self) -> Path:
        return self.state / "smoke_waits.json"

    def _smoke_note(self, kind: str, queued_s: float) -> None:
        """Remember how a smoke fared in the queue: ``started`` (it got a slot after ``queued_s``) or ``timeout``."""
        with self._smoke_lock:
            try:
                rows = json.loads(self._smoke_notes_path().read_text())
            except (OSError, ValueError):
                rows = []
            rows = [r for r in rows if isinstance(r, dict)][-49:]
            rows.append({"ts": self._clock(), "kind": kind, "queued_s": round(float(queued_s), 1)})
            try:
                self.state.mkdir(parents=True, exist_ok=True)
                self._smoke_notes_path().write_text(json.dumps(rows))
            except OSError:
                pass

    def _our_pending(self) -> list[tuple[str, str, int, float, str]]:
        """(job id, name, nice, submit epoch, qos) of this loop's PENDING jobs (name prefix ``lp<tag>-``); [] on a squeue failure."""
        rc, out, _ = self.runner.run(["squeue", "-h", "-t", "PD", "-u", getpass.getuser(), "-o", "%i|%j|%y|%V|%q"], timeout=60)
        rows = []
        for line in out.splitlines() if rc == 0 else ():
            jid, _, rest = line.strip().partition("|")
            name, _, rest = rest.partition("|")
            nice, _, rest = rest.partition("|")
            sub, _, qos = rest.partition("|")
            if not name.startswith(f"lp{self._tag}-") or not jid.isdigit():
                continue
            try:
                ts = time.mktime(time.strptime(sub.strip(), "%Y-%m-%dT%H:%M:%S"))
            except ValueError:
                ts = self._clock()
            rows.append((jid, name, int(nice) if nice.strip().lstrip("-").isdigit() else 0, ts, qos.strip()))
        return rows

    def smokes_pending(self) -> dict[str, int]:
        """Pool -> number of our smoke jobs PENDING there (squeue; {} on a failure). The run book holds new screen / seed
        submits to a pool while one waits (smokes go first among our own jobs by GATING how many we keep pending, never
        by SLURM Nice: 10-02, Nice=10 put our jobs behind every other user's and 0 ran for ~9 h)."""
        by_qos: dict[str, str] = {}
        for p in self.ctx.cfg.pools:
            by_qos.setdefault(p.qos, p.name)
        out: dict[str, int] = {}
        for _, name, _, _, q in self._our_pending():
            if "-smoke-" in name and q in by_qos:
                out[by_qos[q]] = out.get(by_qos[q], 0) + 1
        return out

    def smoke_capacity(self) -> tuple[str, str]:
        """(state, why) -- may a new research session start? Its smoke has to get a slot within the session's own wait.
        ``closed``: one of our smokes has been PENDING longer than ``slurm.smoke_starve_s`` (a session started now waits
        behind it). ``probe``: a smoke queue-timed-out within ``slurm.smoke_probe_s`` and none has started since: one
        session at a time. ``open`` otherwise (and on any squeue failure: never block on a monitoring error)."""
        now = self._clock()
        if self._cap_cache is not None and now - self._cap_cache[0] < 30.0:
            return self._cap_cache[1]
        sl = self.ctx.cfg.slurm
        starve, mem = float(getattr(sl, "smoke_starve_s", 1500)), float(getattr(sl, "smoke_probe_s", 10800))
        out: tuple[str, str] = ("open", "")
        try:
            ages = [now - ts for _, name, _, ts, _q in self._our_pending() if "-smoke-" in name]
            notes = json.loads(self._smoke_notes_path().read_text()) if self._smoke_notes_path().is_file() else []
            last = lambda k: max((float(r.get("ts", 0)) for r in notes if isinstance(r, dict) and r.get("kind") == k), default=0.0)  # noqa: E731
            if ages and max(ages) > starve:
                out = ("closed", f"{len(ages)} smoke(s) pending, oldest {max(ages) / 60:.0f}m > {starve / 60:.0f}m")
            elif last("timeout") > last("started") and now - last("timeout") < mem:
                out = ("probe", f"smoke timed out {(now - last('timeout')) / 60:.0f}m ago, none started since")
        except (OSError, ValueError, TypeError):
            pass
        self._cap_cache = (now, out)
        return out

    def commit(self, worktree: Worktree, *, message: str) -> str:
        self._git(worktree.path, "rm", "-q", "--cached", "--ignore-unmatch", "--", *ENV_LINKS)
        self._git(worktree.path, "add", "-A", "--", ".", NO_VENV)
        self._git(worktree.path, "commit", "-m", message)      # rc 1 = nothing to commit: fine
        rc, out, err = self._git(worktree.path, "rev-parse", "HEAD")
        if rc != 0:
            raise RuntimeError(f"rev-parse HEAD failed in {worktree.path}: {err.strip()}")
        return out.strip()

    def commit_files(self, parent_commit: str, files: Mapping[str, str], *, message: str, branch: str = "") -> str:
        """v3-c8a B5: ``parent_commit`` with ``files`` (repo-relative path -> full text) replaced, committed WITHOUT a
        worktree (git plumbing on a private index: read-tree, hash-object, update-index, write-tree, commit-tree) --
        a config-only wave point / ladder dose / key graft cost a ~25 s ``worktree add`` + checkout on virtiofs
        each (live 10-14: launch.waves 130-333 s). ``parent_commit`` itself when nothing changed; ``branch`` is
        pointed at the commit when given. Raises on a git failure."""
        import tempfile
        ident = ("-c", "user.name=loop", "-c", "user.email=loop@localhost")
        tmp = self.state / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        fd, idx = tempfile.mkstemp(prefix="idx-", dir=tmp)
        os.close(fd)
        os.unlink(idx)                                    # git writes the index itself
        env = {"GIT_INDEX_FILE": idx}

        def git(*args: str) -> str:
            rc, out, err = self.runner.run(["git", "-C", self.repo, *ident, *args], timeout=120, env=env)
            if rc != 0:
                raise RuntimeError(f"git {args[0]} failed (rc={rc}): {(err or out).strip()[-300:]}")
            return out.strip()
        blobs: list[str] = []
        try:
            git("read-tree", parent_commit)
            for rel, text in files.items():
                fd, blob = tempfile.mkstemp(prefix="blob-", dir=tmp)
                with os.fdopen(fd, "w") as f:
                    f.write(text)
                blobs.append(blob)
                sha = git("hash-object", "-w", blob)
                mode = (git("ls-tree", parent_commit, "--", rel).split() or ["100644"])[0]
                git("update-index", "--add", "--cacheinfo", f"{mode},{sha},{rel}")
            tree = git("write-tree")
            if tree == git("rev-parse", f"{parent_commit}^{{tree}}"):
                return parent_commit
            out = git("commit-tree", tree, "-p", parent_commit, "-m", message)
            if branch:
                git("update-ref", f"refs/heads/{branch}", out)
            return out
        finally:
            for f in (idx, *blobs):
                try:
                    os.unlink(f)
                except OSError:
                    pass

    def diff(self, worktree: Worktree) -> tuple[str, tuple[str, ...], Mapping[str, Any]]:
        """Every tracked or untracked (non-ignored) change vs base, plus each changed ``file::`` entry;
        a file whose changes all sit inside its entries is reported by those entries alone."""
        wt, base = worktree.path, worktree.base_commit
        self._git(wt, "rm", "-q", "--cached", "--ignore-unmatch", "--", *ENV_LINKS)   # a researcher's own add -A
        self._git(wt, "add", "-A", "--", ".", NO_VENV)
        _, text, _ = self._git(wt, "diff", "--cached", base)
        _, names, _ = self._git(wt, "diff", "--cached", "--name-only", base)
        paths = [n for n in names.splitlines() if n]
        old = _yaml_flat(self._show(wt, base, self.config_rel))
        new = _yaml_flat(_read(Path(wt) / self.config_rel))
        cfg_diff = {k: new.get(k) for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}
        for rel in {l.split("::", 1)[0] for l in self.locks} & set(paths):
            old, new = self._show(wt, base, rel), _read(Path(wt) / rel)
            locks = [l for l in self.locks if l.startswith(rel + "::")]
            hit = [l for l in locks if locked_source(old, l) != locked_source(new, l)]
            if hit and outside_spans(old, locks) == outside_spans(new, locks):
                paths.remove(rel)
            paths += hit
        return text, tuple(paths), cfg_diff

    def merge(self, worktree: Worktree, *, other_commit: str) -> tuple[bool, str]:
        rc, out, err = self._git(worktree.path, "merge", "--no-edit", other_commit)
        if rc == 0:
            return True, ""
        self._git(worktree.path, "merge", "--abort")
        return False, (out + err).strip()[-2000:]

    # ------------------------------------------------------ run diagnostics (loop.diagnostics)
    def postprocess(self, run: Run) -> None:
        """A DONE run with a latest.pt: launch its detached low-priority feature extraction (loop/diag_ckpt.py)."""
        cfg = self.ctx.cfg.diagnostics
        if not cfg.enabled or not (Path(run.output_dir or "") / "latest.pt").is_file():
            return
        cands = [self.worktrees / _safe(run.node_id), *([] if cfg.python else [Path(self.repo)])]
        py = next((self._python(str(c)) for c in cands if any((c / ".venv" / "lib").glob("python*/site-packages/torch"))),
                  cfg.python)                       # the run's worktree venv when it carries torch, else the configured one
        node_code = self.worktrees / _safe(run.node_id)     # its model.type's own init (an init swap's drift)
        if py:
            DG.launch(self.ctx, run, py, cfg.init, str(node_code) if (node_code / "model.py").is_file() else "")

    # ------------------------------------------------------ smoke / init
    def _python(self, wt: str) -> str:
        return str(Path(wt) / ".venv" / "bin" / "python")    # never `uv run`: it would re-sync

    def smoke(self, worktree: Worktree, *, memory: bool = False, mitigation: str = "",
              queue_s: float | None = None) -> SmokeResult:
        """Data-subset training on one GPU through the run launcher (``sbatch --wait``, a short time
        limit; no probes, wandb off). Blocks the caller's thread, never the driver's. ``memory``: keep
        the config's batch size (a memory probe); ``mitigation``: an OOM ladder step, as a run."""
        smoke_dir = self.state / "smoke" / _safe(Path(worktree.path).name + (f"-{mitigation}" if mitigation else ""))
        smoke_dir.mkdir(parents=True, exist_ok=True)
        for stale in (smoke_dir / "out").rglob("latest.pt") if (smoke_dir / "out").is_dir() else ():
            try:                        # a killed earlier smoke of this worktree left its checkpoint (~1.35 GB each)
                stale.unlink()
            except OSError:
                pass
        try:
            cfg = yaml.safe_load((Path(worktree.path) / self.config_rel).read_text())
        except (OSError, yaml.YAMLError) as exc:
            return SmokeResult(False, f"cannot read {self.config_rel}: {exc}")
        for key, val in self.SMOKE_OVERRIDES.items():
            if not (memory and key in self.MEMORY_KEEP):
                set_dotted(cfg, key, val)
        if stand := DB.stand_in(self.ctx, (cfg.get("data") or {}).get("dataset_dir")):   # its dataset is building:
            set_dotted(cfg, "data.dataset_dir", stand)                                  # same schema, smaller tiles
        probe_why = self.probe_smoke_why(worktree)
        if probe_why:                   # latest.pt at the end of the smoke (after the readout fit): the probe leg's input
            set_dotted(cfg, "train.save_every", PROBE_SAVE_EVERY)
        if held := self._infra_hold(worktree, cfg):
            return held
        cfg_path = smoke_dir / "smoke.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        name = f"lp{self._tag}-smoke-{smoke_dir.name}"
        with self._smoke_lock:          # a slot like any run: preemptible pool when free, else primary
            pool = self.pool_picker() if self.pool_picker else self.ctx.cfg.pools[0].name
            self.smokes[pool] += 1
            self._smoke_names.add(name)
        jid, why = "", ""
        try:
            argv, env = self._sbatch(str(cfg_path), smoke_dir / "out", name, pool=pool)
            if probe_why:               # one job: the launcher's training smoke, then the probe leg
                argv = ["sbatch", str(self._probe_wrapper(worktree, smoke_dir, cfg_path)), *argv[1:]]
            env.pop("SBATCH_REQUEUE", None)     # a preempted smoke ends (no verdict, re-smoked): a requeued one
            env.update(OFFLINE_ENV)    # v3 review M1: a download attempt fails the smoke's construct
            env.update(WANDB_MODE="disabled", SBATCH_NO_REQUEUE="",                                    # pended on
                       SBATCH_TIMELIMIT=str(SMOKE_MIN + (PROBE_SMOKE_MIN if probe_why else 0)),     # forever, owned
                       **(ALLOC_ENV if mitigation else {}))                                         # by no thread
            rc, out, err = self.runner.run(argv, cwd=worktree.path, timeout=300, env=env)
            if m := re.search(r"Submitted batch job (\d+)", out):
                jid = m.group(1)
            if rc == 0 and jid:
                rc, why = self._smoke_wait(jid, run_s=SMOKE_RUN_S + (60 * PROBE_SMOKE_MIN if probe_why else 0),
                                             queue_s=queue_s)
        except BaseException:
            if jid:
                self.runner.run(["scancel", jid], timeout=60)
            raise
        finally:
            with self._smoke_lock:
                self.smokes[pool] -= 1
                self._smoke_names.discard(name)
        if rc == 124 and not jid:                               # the submission itself hung: never leave it
            self.runner.run(["scancel", "--name", name], timeout=60)
        logs = self._tail({"worktree": worktree.path}, jid) if jid else ""
        tail = "\n".join((out + "\n" + err + "\n" + logs).strip().splitlines()[-40:])
        if rc == 124:                   # never got (or kept) a slot in time: backend unavailable, not a verdict
            self.smoke_gate.release_canary()
            return SmokeResult(False, f"smoke_unavailable: {why or f'queue timeout after {self.smoke_queue_s(queue_s):.0f}s'} on {pool}",
                               tail, unavailable=True)
        full = "\n".join(_read(p) or "" for p in self._logs({"worktree": worktree.path}, jid)) if jid else ""
        if rc != 0 and (verdict := self._infra_verdict(worktree, tail, full)):
            return verdict
        if rc != 0 and PROBE_SMOKE_FAILED in full:         # the readout crashed in probe.py's own entry points
            line = next(l for l in reversed(full.splitlines()) if PROBE_SMOKE_FAILED in l)
            return SmokeResult(False, f"probe smoke failed ({probe_why}): "
                                      f"{line.split(PROBE_SMOKE_FAILED, 1)[1].strip()[:300]}", tail)
        if rc != 0:
            return SmokeResult(False, "smoke OOM" if is_oom(tail) else f"smoke exited rc={rc}", tail)
        self.smoke_gate.cleared()                           # it got past the environment
        if re.search(r"\b(nan|inf)\b", tail, re.I):
            return SmokeResult(False, "smoke log shows NaN/inf", tail)
        if probe_why and PROBE_SMOKE_OK not in full:      # the job passed but the probe leg never reported
            return SmokeResult(False, f"probe smoke did not report ok ({probe_why})", tail)
        m = self.FLOPS_RE.search(full) if self.FLOPS_RE is not None else None
        bs = (cfg.get("train") or {}).get("batch_size")
        path = self._flops_path(worktree)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": self._flop_key(worktree), "per_sample": int(m.group(1).replace(",", "")) / bs
                                    if m and isinstance(bs, int) and bs > 0 else None}))
        return SmokeResult(True, "", tail)

    def _infra_hold(self, worktree: Worktree, cfg: Mapping[str, Any]) -> SmokeResult | None:
        """Before a smoke is submitted: held while the gate is tripped (the work is kept: ``unavailable``), and the
        cheap environment preflight (torch hub dir resolves, the pretrained weights are cached or downloadable). A
        failed preflight trips the gate and raises an ALERT naming the path instead of a bare ``rc=1`` later."""
        if why := self.smoke_gate.blocked():
            return SmokeResult(False, f"smoke_unavailable: infra hold: {why}", "", unavailable=True, infra=True)
        if self.preflight is None:
            return None
        try:
            mpy = (Path(worktree.path) / "model.py").read_text()
        except OSError:
            mpy = ""
        try:
            bad = self.preflight(mpy, str((cfg.get("model") or {}).get("type") or ""))
        except Exception:  # noqa: BLE001 - a broken check never blocks a smoke
            self.smoke_gate.release_canary()
            return None
        if not bad:
            return None
        self._infra_alert(f"preflight: {bad}", "", Path(worktree.path).name)
        return SmokeResult(False, f"smoke_unavailable: infra (preflight): {bad}", "", unavailable=True, infra=True)

    def _infra_verdict(self, worktree: Worktree, tail: str, full: str) -> SmokeResult | None:
        """A failed smoke job's log: an environment signature (or the same traceback tail from >= CORRELATED different
        sessions within an hour) is INFRA -- no verdict on the idea, the gate trips, an ALERT goes out once. Anything
        else got past the environment: the gate clears."""
        text, wt = full or tail, Path(worktree.path).name
        sig, kind = INFRA.signature(text), INFRA.classify(text)
        if not kind and sig and not is_oom(tail) and not re.search(r"\b(nan|inf)\b", tail, re.I):
            n = self.smoke_gate.correlated(wt, sig)
            if n >= INFRA.CORRELATED and not self.smoke_gate.flag_correlated(wt, sig):
                kind = f"correlated: {n} sessions failed smoke with the same tail: {sig}"
        if not kind:
            self.smoke_gate.cleared()
            return None
        self._infra_alert(kind, sig, wt)
        return SmokeResult(False, f"smoke_unavailable: infra ({kind[:300]})", tail, unavailable=True, infra=True)

    def _infra_alert(self, reason: str, sig: str, wt: str) -> None:
        if self.smoke_gate.trip(reason, sig, wt):
            sessions = ",".join(self.smoke_gate.sessions(sig)) if sig else wt
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self._clock()))} ALERT: smoke INFRA failure "
                  f"({reason[:300]}); new smokes held, nodes kept for retry, sessions re-issued; sessions: {sessions}",
                  file=sys.stderr, flush=True)

    def smoke_hold_wait(self) -> float:
        """Seconds a held smoke should wait before trying again (the researcher's infra wait)."""
        return self.smoke_gate.wait_s()

    def probe_smoke_why(self, worktree: Worktree) -> str:
        """Why this smoke also runs the probe leg ('' = it does not): ``slurm.smoke_probe`` "always"; "readout" when the
        worktree's change vs its base (committed, uncommitted or untracked) touches PROBE_MODEL_FILES, a ``file::``
        span of the evalonly family in another file (train.py's post-training readout fit), or a PROBE_KEYS config
        key; "off" never. A memory smoke of a design point (its worktree IS the point's commit) has no diff."""
        mode = str(getattr(self.ctx.cfg.slurm, "smoke_probe", SMOKE_PROBE_DEFAULT) or "off").strip().lower()
        if mode not in SMOKE_PROBE_MODES:
            mode = SMOKE_PROBE_DEFAULT
        if mode != "readout":
            return "smoke_probe=always" if mode == "always" else ""
        wt, base = worktree.path, worktree.base_commit
        if not base:
            return ""
        rc, names, _ = self._git(wt, "diff", "--name-only", base, "--", ".", NO_VENV)
        _, extra, _ = self._git(wt, "ls-files", "--others", "--exclude-standard")
        if rc != 0:
            return ""
        paths = set(names.split()) | set(extra.split())
        for rel in sorted(paths & set(PROBE_MODEL_FILES)):
            if self._probe_funcs_changed(wt, base, rel):
                return f"{rel} changed"
        eo = getattr(self.ctx.cfg, "evalonly", None)
        fam = self.ctx.cfg.families.get(getattr(eo, "family", "eval_readout") or "eval_readout")
        for span in (p for p in (fam.paths if fam else ()) if "::" in p):
            rel = span.split("::", 1)[0]
            if rel in paths and locked_source(self._show(wt, base, rel), span) != locked_source(_read(Path(wt) / rel), span):
                return f"{span.split('...', 1)[0][:80]} changed"
        if self.config_rel in paths:
            old = _yaml_flat(self._show(wt, base, self.config_rel))
            new = _yaml_flat(_read(Path(wt) / self.config_rel))
            keys = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)
                          and any(k == p or (p.endswith(".") and k.startswith(p)) for p in PROBE_KEYS))
            if keys:
                return f"config {keys[0]} changed"
        return ""

    def _probe_funcs_changed(self, wt: str, base: str, rel: str) -> str:
        """The first probe-facing function (``slurm.smoke_probe_funcs``, default PROBE_FUNCS; matched by name or
        ``Class.name``) whose body the diff of ``rel`` vs ``base`` touches ('' = none), from the -U0 hunk line ranges
        mapped onto the new and old sources' ``def`` spans. An unparsable source counts as a hit (fail safe)."""
        funcs = set(getattr(self.ctx.cfg.slurm, "smoke_probe_funcs", None) or PROBE_FUNCS)
        rc, diff, _ = self._git(wt, "diff", "-U0", base, "--", rel)
        old, new = self._show(wt, base, rel), _read(Path(wt) / rel)
        if rc != 0 or old is None or new is None:       # new / deleted / unreadable: cannot attribute, fail safe
            return rel
        if not diff.strip():
            return ""
        olds, news = [], []
        for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", diff, re.M):
            a, b, c, d = int(m[1]), int(m[2] or 1), int(m[3]), int(m[4] or 1)
            olds.append((a, a + max(b, 1) - 1))
            news.append((c, c + max(d, 1) - 1))
        for src, spans in ((new, news), (old, olds)):
            try:
                tree = ast.parse(src or "")
            except SyntaxError:
                return rel
            def walk(node, cls=""):
                for ch in ast.iter_child_nodes(node):
                    if isinstance(ch, ast.ClassDef):
                        yield from walk(ch, ch.name)
                    elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        yield (f"{cls}.{ch.name}" if cls else ch.name), ch.name, ch.lineno, ch.end_lineno or ch.lineno
                        yield from walk(ch, cls)
                    else:
                        yield from walk(ch, cls)
            for qual, name, lo, hi in walk(tree):
                if (name in funcs or qual in funcs) and any(a <= hi and lo <= b for a, b in spans):
                    return qual
        return ""

    def _probe_wrapper(self, worktree: Worktree, smoke_dir: Path, cfg_path: Path) -> Path:
        """The probe smoke's sbatch script: the launcher's own #SBATCH header; runs the launcher in-job (it trains;
        its submit/ is locked), then loop/adapters/probe_smoke.py (a copy beside the smoke config) on the smoke's
        latest.pt in the worktree venv; latest.pt is removed either way, the job's exit code is the verdict."""
        launcher = _read(Path(worktree.path) / self.LAUNCHER) or ""
        heads = [l for l in launcher.splitlines() if l.startswith("#SBATCH")]
        script = smoke_dir / PROBE_SMOKE_SCRIPT.name
        shutil.copyfile(PROBE_SMOKE_SCRIPT, script)
        ckpt = smoke_dir / "out" / "latest.pt"
        (Path(worktree.path) / "slurm").mkdir(exist_ok=True)       # --output=slurm/%j.out: SLURM makes no dirs
        q = shlex.quote
        path = smoke_dir / "probe_smoke.sbatch"
        path.write_text("\n".join([
            "#!/usr/bin/env bash", *heads,
            "# loop probe smoke (loop.adapters.nanopath_v2.smoke): training smoke, then the readout through probe.py",
            "set -uo pipefail",
            f'cd "${{SLURM_SUBMIT_DIR:-{worktree.path}}}"',
            f'bash {self.LAUNCHER} "$@" || exit $?',
            'echo "probe_smoke: start"',
            f'"$PWD/.venv/bin/python" {q(str(script))} "$PWD" {q(str(cfg_path))} {q(str(ckpt))}',
            "rc=$?",
            f"rm -f {q(str(ckpt))}",
            "exit $rc", ""]))
        path.chmod(0o755)
        return path

    # the smoke's FLOP measurement (an init swap's flop_budget check, loop.adapters.initcompat)
    def _flops_path(self, worktree: Worktree) -> Path:
        return self.state / "smoke" / _safe(Path(worktree.path).name) / "flops.json"

    def _flop_key(self, worktree: Worktree) -> str:
        """What a run's FLOPs per sample depend on: the config + FLOP_FILES of the worktree, as they are now."""
        h = hashlib.sha1()
        for rel in (self.config_rel, *FLOP_FILES):
            h.update(((_read(Path(worktree.path) / rel) or "") + "\0").encode())
        return h.hexdigest()

    def _flops(self, worktree: Worktree) -> dict[str, Any] | None:
        """{per_sample, max_samples, max_flops} when a successful smoke measured THIS code / config, else None."""
        try:
            rec = json.loads(self._flops_path(worktree).read_text())
            cfg = yaml.safe_load((Path(worktree.path) / self.config_rel).read_text()) or {}
        except (OSError, ValueError, yaml.YAMLError):
            return None
        if rec.get("key") != self._flop_key(worktree):
            return None
        train = cfg.get("train") or {}
        return {"per_sample": rec.get("per_sample"), "max_samples": train.get("max_train_samples"),
                "max_flops": train.get("max_train_flops")}

    def smoke_queue_s(self, queue_s: float | None = None) -> float:
        """The PENDING time a smoke may wait for a slot: ``queue_s`` (the caller's, e.g. a session tool's
        ``slurm.smoke_session_wait_s``), else ``slurm.smoke_wait_s``, else SMOKE_WAIT_S."""
        if queue_s and float(queue_s) > 0:
            return float(queue_s)
        return float(getattr(self.ctx.cfg.slurm, "smoke_wait_s", 0.0) or 0.0) or float(SMOKE_WAIT_S)

    def _smoke_wait(self, jid: str, run_s: float = SMOKE_RUN_S, queue_s: float | None = None) -> tuple[int, str]:
        """Block until smoke job ``jid`` ends: (rc, why). PENDING time counts against :meth:`smoke_queue_s` (then
        scancelled: rc 124, no verdict -- a queue wait is never a smoke failure nor an infra strike); once RUNNING the
        job has its own wall SMOKE_RUN_S (SLURM's time limit normally ends it first). A preempted / node-failed smoke
        is no verdict either (rc 124)."""
        wait_s = self.smoke_queue_s(queue_s)
        queued = ran = 0.0
        last = self._clock()
        while True:
            rc, q, err = self.runner.run(["squeue", "-h", "-j", jid, "-o", "%T"], timeout=60)
            now = self._clock()
            dt, last = max(0.0, now - last), now
            state = (q.split() or [""])[0] if rc == 0 else "?"
            if rc != 0 and "invalid job id" in err.lower():
                state = ""
            if not state:
                break
            if state in SMOKE_RUN_STATES:
                if not ran:
                    self._smoke_note("started", queued)
                ran += dt
            else:
                queued += dt
            if queued > wait_s or ran > run_s:
                self.runner.run(["scancel", jid], timeout=60)
                if queued > wait_s:
                    self._smoke_note("timeout", queued)
                return (124, f"queue timeout after {queued:.0f}s pending") if queued > wait_s \
                    else (1, f"smoke ran past {run_s:.0f}s; scancelled")
            time.sleep(SMOKE_POLL_S)
        for _ in range(5):                                      # accounting lags squeue
            rc, acct, _ = self.runner.run(["sacct", "-n", "-X", "-P", "-j", jid, "-o", "State,ExitCode"], timeout=60)
            state, _, code = (acct.strip().splitlines() or [""])[0].partition("|")
            state = (state.split() or [""])[0]
            if state and state not in WAIT_STATES:
                break
            time.sleep(SMOKE_POLL_S)
        if state == "COMPLETED":
            return 0, ""
        if state in FREE_RETRY_STATES or state == "CANCELLED":     # the machine (or someone) took it: no verdict
            return 124, f"smoke job {jid} {state}"
        ec = code.split(":")[0]
        return (int(ec) or 1 if ec.isdigit() else 1), f"slurm {state or 'state unknown'}"

    def init_compat(self, worktree: Worktree, *, parent_commit: str) -> InitCompat:
        eo = getattr(self.ctx.cfg, "evalonly", None)
        if eo is not None and eo.enabled:           # scored on the parent's own weights: nothing to init
            rc, head, _ = self._git(worktree.path, "rev-parse", "HEAD")
            ok, _, shim = self.eval_only(parent_commit, head.strip()) if rc == 0 else (False, "", False)
            if ok:
                return InitCompat(True, "", {"eval_only": {"ok": True, "shim": shim}})
        ic = self.ctx.cfg.init_compat
        got = run_init_check(runner=self.runner, worktree_path=worktree.path,
                             parent_commit=parent_commit, state_dir=str(self.state),
                             limits=Limits(config_rel=self.config_rel, deny=tuple(ic.deny),
                                           allow=tuple(getattr(ic, "allow", ()) or ()), flop_cap=ic.flop_cap),
                             python=self._python(worktree.path), flops=self._flops(worktree), env=OFFLINE_ENV)
        if got.ok and "init_swap" in (got.checks or {}):
            self._swap_ok_record(worktree.path)
        return got

    def _swap_ok_path(self) -> Path:
        return self.state / "init_swap_ok.json"

    def _swap_ok_urls(self) -> set:
        try:
            return set(json.loads(self._swap_ok_path().read_text()))
        except (OSError, ValueError):
            return set()

    def _swap_ok_record(self, wt_path: str) -> None:
        """v3 review M1: remember the pretrained_url of a swap whose init-compat check passed (the O5 launch guard accepts it).
        A type mark is recorded only for a type matching ``init_compat.allow_types`` (never bare ``timm`` / ``timm_plain``,
        which load arbitrary ids) and is bound to the worktree's HEAD sha (``type:<t>@<sha>``), not global."""
        try:
            cfg = flatten(yaml.safe_load((Path(wt_path) / self.config_rel).read_text()) or {})
            url, typ = cfg.get("model.pretrained_url"), cfg.get("model.type")
            if typ == "timm" and isinstance(cfg.get("model.timm_name"), str):
                typ = f"timm:{cfg['model.timm_name']}"
            allow_types = tuple(getattr(self.ctx.cfg.init_compat, "allow_types", ()) or ())
            rc, head, _ = self._git(wt_path, "rev-parse", "HEAD")
            sha = head.strip() if rc == 0 else ""
            type_mark = (f"type:{typ}@{sha}" if sha and isinstance(typ, str) and any(
                re.search(p, typ, re.I) for p in allow_types) else None)
            marks = {x for x in (url, type_mark) if isinstance(x, str) and x}
            if marks:
                self.state.mkdir(parents=True, exist_ok=True)
                self._swap_ok_path().write_text(json.dumps(sorted(self._swap_ok_urls() | marks)))
        except Exception:  # noqa: BLE001 - best effort; a miss only makes the guard stricter
            pass

    # ------------------------------------------------------- eval-only
    def eval_only(self, parent_commit: str, commit: str) -> tuple[bool, str, bool]:
        """(eligible, why not, needs a resume shim): every change from ``parent_commit`` lies inside the READOUT-ONLY
        ``evalonly.family`` spans, except added / dropped fitted buffers (EO_BUFFER) and comments in the span files;
        no other file (so no config) changes.
        v3-c5 I5: with ``evalonly.readout_keys`` a changed config key or an outside-span literal constant (EO_CONST)
        is allowed too when its name is read ONLY inside the readout-only spans (:meth:`_readout_only`).
        v3-c5b M3: a span TRAINING can execute (:func:`train_reachable`: ``evalonly.train_spans`` and every span they
        call -- the EMA teacher runs ``forward`` in eval mode, so ``if not self.training`` bodies run during training
        too) may not change at all, and a changed symbol read inside one is a training-path change; an EXISTING
        buffer / attribute's declaration may not change (``rn_fitted = True`` would switch the teacher's readout on
        mid-training); an added / dropped buffer may be read only inside the readout-only spans."""
        eo = self.ctx.cfg.evalonly
        fam = self.ctx.cfg.families.get(eo.family)
        spans = [l for l in (fam.paths if fam else ()) if "::" in l]
        keys_ok = bool(getattr(eo, "readout_keys", False))
        rc, names, err = self._git(self.repo, "diff", "--name-only", parent_commit, commit)
        if rc != 0 or not names.split():
            return False, f"no diff ({err.strip()[-200:]})", False
        train = tuple(getattr(eo, "train_spans", ()) or ())
        reach: set[str] = set()
        if train:
            files = {l.split("::", 1)[0] for l in spans if l in train}
            reach = train_reachable([self._show(self.repo, c, rel) for rel in sorted(files)
                                     for c in (parent_commit, commit)], spans, train)
        readout = [l for l in spans if l not in reach]
        shim, symbols, buffers = False, {}, {}
        for rel in names.split():
            if keys_ok and rel == self.config_rel:
                old = _yaml_flat(self._show(self.repo, parent_commit, rel))
                new = _yaml_flat(self._show(self.repo, commit, rel))
                for k in sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)):
                    if any(k == p or (p.endswith(".") and k.startswith(p)) for p in PROBE_KEYS):
                        return False, f"{rel}:{k} is a probe key", False
                    symbols.setdefault(k.rsplit(".", 1)[-1], f"{rel}:{k}")
                continue
            locks = [l for l in spans if l.startswith(rel + "::")]
            if not locks:
                return False, f"{rel} is outside the readout spans", False
            src_a, src_b = self._show(self.repo, parent_commit, rel), self._show(self.repo, commit, rel)
            a, b = outside_spans(src_a, locks), outside_spans(src_b, locks)
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
                if op == "equal":
                    continue
                moved = a[i1:i2] + b[j1:j2]
                if bad := [l for l in moved if not (EO_TRIVIAL.match(l) or EO_BUFFER.match(l)
                                                     or (keys_ok and EO_CONST.match(l)))]:
                    return False, f"{rel}: `{bad[0].strip()[:120]}` is outside the readout spans", False
                shim = shim or any(EO_BUFFER.match(l) for l in moved)
                for l in moved:
                    if not (EO_TRIVIAL.match(l) or EO_BUFFER.match(l)) and (m := EO_CONST.match(l)):
                        symbols.setdefault(m.group(1), f"{rel}: `{l.strip()[:80]}`")
            da, db = _eo_decls(a), _eo_decls(b)
            for n in sorted(set(da) | set(db)):
                if n in da and n in db and da[n] != db[n]:
                    return False, (f"{rel}: `{n}` changes an existing buffer / attribute's initial value (it "
                                   f"exists in the trained model: training / the teacher may read it)"), False
                if (n in da) != (n in db):
                    buffers.setdefault(n, f"{rel}: buffer `{n}` {'added' if n in db else 'dropped'}")
            for lock in (l for l in locks if l in reach):
                if locked_source(src_a, lock) != locked_source(src_b, lock):
                    why = ("a training-path change" if lock in train else
                           f"a training-path change (reached from {', '.join(sorted(train))})")
                    return False, (f"{lock}: {why} -- training (and its eval-mode EMA teacher, for which "
                                   f"`not self.training` holds) runs this span"), False
        for sym, where in symbols.items():
            if why := self._readout_only(sym, (parent_commit, commit), readout):
                return False, f"{where}: {why}", False
        for sym, where in buffers.items():
            if why := self._readout_only(sym, (parent_commit, commit), readout, unread_ok=True):
                return False, f"{where}: {why}", False
        return True, "", shim

    def _readout_only(self, sym: str, commits: Sequence[str], spans: Sequence[str], unread_ok: bool = False) -> str:
        """'' when ``sym`` (a config key's leaf / a constant's name) is read in the *.py files of ``commits`` and
        every read lies inside a readout-only span (its own EO_CONST definition lines aside); else why not.
        ``unread_ok``: no read at all is fine (an added fitted buffer nothing reads yet)."""
        reads = 0
        for c in commits:
            rc, out, err = self._git(self.repo, "grep", "-n", "-w", "-e", sym, c, "--", "*.py")
            if rc not in (0, 1):
                return f"git grep {sym} failed: {err.strip()[-120:]}"
            ranges: dict[str, list[tuple[int, int]]] = {}
            for line in out.splitlines():
                parts = line.split(":", 3)
                if len(parts) < 4 or not parts[2].isdigit():
                    continue
                rel, no, text = parts[1], int(parts[2]) - 1, parts[3]
                if EO_TRIVIAL.match(text) or EO_BUFFER.match(text):    # a comment / a fitted buffer's declaration
                    continue
                if (m := EO_CONST.match(text)) and m.group(1) == sym:
                    continue
                if rel not in ranges:
                    src = self._show(self.repo, c, rel)
                    ranges[rel] = [s[:2] for l in spans if l.startswith(rel + "::") and (s := _span(src, l))]
                if not any(i <= no <= j for i, j in ranges[rel]):
                    return f"`{sym}` is read outside the readout-only spans ({rel}:{no + 1})"
                reads += 1
        return "" if reads or unread_ok else f"`{sym}` is read by no code (not a readout knob)"

    def trees(self, commits: Sequence[str]) -> dict[str, str]:
        """commit -> its git tree id (one rev-parse; an unknown commit is left out)."""
        cs = [c for c in dict.fromkeys(commits) if c]
        rc, out, _ = self._git(self.repo, "rev-parse", *(f"{c}^{{tree}}" for c in cs)) if cs else (0, "", "")
        if rc != 0:                                        # one bad commit fails the batch: one at a time
            return {} if len(cs) == 1 else {c: t for c in cs for t in self.trees([c]).values()}
        return dict(zip(cs, out.split()))

    def ancestry(self, commit: str, base: str) -> list[str]:
        """The commits ``commit`` contains above ``base`` (``git rev-list commit ^base``; itself included)."""
        rc, out, err = self._git(self.repo, "rev-list", commit, f"^{base}")
        if rc != 0:
            raise RuntimeError(f"git rev-list {commit[:12]} ^{base[:12]} failed: {err.strip()[-200:]}")
        return out.split()

    def changed_files(self, a: str, b: str) -> list[str]:
        """The paths ``git diff --name-only a b`` lists (raises on a git failure: never read as "no change")."""
        rc, out, err = self._git(self.repo, "diff", "--name-only", a, b)
        if rc != 0:
            raise RuntimeError(f"git diff --name-only {a[:12]} {b[:12]} failed: {err.strip()[-200:]}")
        return [x for x in out.split("\n") if x.strip()]

    def diff_commits(self, a: str, b: str) -> str:
        rc, out, err = self._git(self.repo, "diff", a, b)
        if rc != 0:
            raise RuntimeError(f"git diff {a[:12]} {b[:12]} failed: {err.strip()[-200:]}")
        return out

    def graft(self, parent_commit: str, commit: str, onto: str, name: str) -> str:
        """``commit``'s change over ``parent_commit`` re-applied on ``onto`` (:meth:`graft_port`); ``onto`` itself
        when it already holds the change, '' when it does not apply."""
        sha, how = self.graft_port(parent_commit, commit, onto, name)
        return sha or (onto if how == "contained" else "")

    def graft_port(self, parent_commit: str, commit: str, onto: str, name: str) -> tuple[str, str]:
        """(commit, 'applied') / ('', 'contained') / ('', 'conflict: ...'): ``commit``'s change over
        ``parent_commit`` on ``onto``, committed on branch ``loop/<tag>-eo-<name>``. The config merges per key
        (loop.adapters.merge_config: a key ``onto`` holds equal / subsumed is kept, one it moved elsewhere
        conflicts; live 2026-10-06 a line-level 3-way failed 9 of 10 ports onto a node on adjacent YAML lines);
        every other file by ``git apply --3way``."""
        if onto == parent_commit:
            return commit, "applied"
        branch = f"loop/{self._tag}-eo-{_safe(name)}"
        rc, out, _ = self._git(self.repo, "log", "-1", "--format=%H %P%n%s", branch)   # already grafted: reuse
        head, _, subject = out.partition("\n")
        if rc == 0 and head.split()[1:] == [onto] and subject.strip() == f"loop eval-only graft {name}":
            return head.split()[0], "applied"
        cfg = self.config_rel
        texts = [self._show(self.repo, c, cfg) for c in (parent_commit, commit, onto)]
        put, _, clash = merge_config(*(_yaml_flat(t) for t in texts))
        if clash:
            return "", "conflict: " + ", ".join(f"{cfg}:{k}" for k in clash)
        rc, names, err = self._git(self.repo, "diff", "--name-only", parent_commit, commit)
        if rc != 0:
            raise RuntimeError(f"git diff {parent_commit[:12]} {commit[:12]} failed: {err.strip()[-200:]}")
        code = [n for n in names.split() if n != cfg]
        if not code and not put:
            return "", "contained"
        dest = self.worktrees / "_graft" / _safe(name)
        self.reap_worktree(Worktree(str(dest), "", ""))
        self._add_worktree(dest, onto, "-B", branch)
        try:
            if code:
                patch = self.state / "evalonly" / "patches" / f"{_safe(name)}.patch"
                patch.parent.mkdir(parents=True, exist_ok=True)
                rc, text, err = self._git(self.repo, "diff", parent_commit, commit, "--", *code)
                patch.write_text(text)
                rc, _, err = self._git(dest, "apply", "--3way", str(patch))
                if rc != 0:
                    bad = re.findall(r"patch to '([^']+)' with conflicts", err)
                    return "", "conflict: " + (", ".join(bad) or err.strip()[-200:] or "git apply failed")
            if put:
                merged = yaml_apply(texts[2] or "", texts[1] or "", put)
                if merged is None:
                    return "", f"conflict: {cfg}: keys {sorted(put)} could not be placed"
                (dest / cfg).write_text(merged)
            self._git(dest, "add", "-A", "--", ".", NO_VENV)
            if self._git(dest, "diff", "--cached", "--quiet", onto)[0] == 0:
                return "", "contained"
            return self.commit(Worktree(str(dest), "", onto), message=f"loop eval-only graft {name}"), "applied"
        finally:
            self.reap_worktree(Worktree(str(dest), "", ""))

    def graft_keys(self, commit: str, keys: Sequence[str], onto: str, name: str) -> tuple[str, str]:
        """(commit, 'applied') / ('', 'contained') / ('', 'conflict: ...'): ONLY config ``keys`` as ``commit`` sets
        them, written onto ``onto``'s config (``yaml_apply``: a new key placed after the key it follows in
        ``commit``'s file), committed on branch ``loop/<tag>-eo-<name>`` -- a prerequisite port's key patch (v3
        review K3: never the introducer's whole change)."""
        cfg = self.config_rel
        src, dst = self._show(self.repo, commit, cfg), self._show(self.repo, onto, cfg)
        fs, fd = _yaml_flat(src), _yaml_flat(dst)
        if any(k not in fs for k in keys):
            return "", f"conflict: {cfg}: {commit[:12]} lacks {sorted(k for k in keys if k not in fs)}"
        put = {k: fs[k] for k in keys if fd.get(k, MISSING) != fs[k]}
        if not put:
            return "", "contained"
        branch = f"loop/{self._tag}-eo-{_safe(name)}"
        rc, out, _ = self._git(self.repo, "log", "-1", "--format=%H %P%n%s", branch)   # already grafted: reuse
        head, _, subject = out.partition("\n")
        if rc == 0 and head.split()[1:] == [onto] and subject.strip() == f"loop eval-only graft {name}":
            return head.split()[0], "applied"
        merged = yaml_apply(dst or "", src or "", put)
        if merged is None:
            return "", f"conflict: {cfg}: keys {sorted(put)} could not be placed"
        sha = self.commit_files(onto, {cfg: merged}, message=f"loop eval-only graft {name}", branch=branch)
        return ("", "contained") if sha == onto else (sha, "applied")    # v3-c8a B5: no worktree for a key patch

    def _plan_path(self, node_id: str, stage: Any, seed: int) -> Path:
        return self.state / "evalonly" / _safe(node_id) / f"{getattr(stage, 'value', stage)}_s{int(seed)}.json"

    def plan_eval(self, node_id: str, stage: Any, seed: int, plan: Mapping[str, Any]) -> None:
        """Make run (node, stage, seed) an eval-only run: ``plan`` = {commit, seed (the source run's),
        resume (its latest.pt), shim, source_run, model}; submit reads it back."""
        path = self._plan_path(node_id, stage, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(plan)))

    def eval_plan(self, run: Run) -> dict[str, Any] | None:
        return _json(self._plan_path(run.node_id, run.stage, run.seed))

    def clear_eval(self, node_id: str, stage: Any, seed: int) -> None:
        """Drop an unlaunched run's eval-only plan (v3-c5 I5: the lane routed it, it never launched, it now trains)."""
        self._plan_path(node_id, stage, seed).unlink(missing_ok=True)

    def _shim_script(self) -> Path:
        """state/evalonly/shim.py, written once per content (tmp + os.replace under a lock): submit() runs on
        slurm.submit_workers threads, and a plain write_text let one thread exec a half-written script."""
        script = self.state / "evalonly" / "shim.py"
        with self._shim_lock:
            if _read(script) != SHIM:
                script.parent.mkdir(parents=True, exist_ok=True)
                tmp = script.with_name(f".shim.{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text(SHIM)
                os.replace(tmp, script)
        return script

    def _shim(self, wt: Path, plan: Mapping[str, Any], out: Path) -> None:
        """Write ``out``/SHIM_NAME when the plan needs one (``SHIM``): what train.py strict-loads."""
        dst = out / SHIM_NAME
        if plan.get("shim") and not dst.is_file():
            out.mkdir(parents=True, exist_ok=True)
            script = self._shim_script()
            rc, so, se = self.runner.run([self._python(str(wt)), str(script), str(wt), str(plan["resume"]), str(dst)],
                                         cwd=str(wt), timeout=1800)
            if rc != 0 or not dst.is_file():
                raise RuntimeError(f"eval-only resume shim failed (rc={rc}): {(se or so).strip()[-300:]}")

    # ------------------------------------------------------------ runs
    def _rel(self, run: Run) -> str:
        return f"{_safe(run.node_id)}/{run.stage.value}_s{run.seed}_a{run.attempt}"

    def _receipt(self, rel: str) -> dict[str, Any]:
        return _json(self.state / "receipts" / f"{rel}.json") or {}

    def _save(self, rel: str, rec: Mapping[str, Any]) -> None:
        path = self.state / "receipts" / f"{rel}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec))

    def _job_id(self, rel: str, rec: dict[str, Any]) -> str:
        """The receipt's job id, else recovered by job name (the crash-before-record window)."""
        if rec.get("job_id") or not rec.get("job_name"):
            return rec.get("job_id", "")
        for argv in (["squeue", "-h", "-n", rec["job_name"], "-o", "%i"],
                     ["sacct", "-n", "-X", "-P", "--name", rec["job_name"], "-o", "JobID",
                      "-S", "now-30days"]):
            rc, out, _ = self.runner.run(argv, timeout=60)
            if rc == 0 and out.split():
                rec["job_id"] = out.split()[0]
                self._save(rel, rec)
                return rec["job_id"]
        return ""

    def _handle(self, run_id: str, rec: Mapping[str, Any]) -> JobHandle:
        return JobHandle(run_id=run_id, job_id=rec["job_id"], output_dir=rec["output_dir"],
                         submitted_ts=float(rec["submitted_ts"]))

    def _node_worktree(self, node_id: str, commit: str) -> Path:
        dest = self.worktrees / _safe(node_id)
        with self._dest_lock(dest):                       # v3-c8a B5: other nodes' adds run beside this one
            with self._wt_lock.shared():                  # stamped before tidy (exclusive) can look at it
                self._wt_used[_safe(node_id)] = time.time()
                there = (dest / ".git").exists()
            if not there:
                self._add_worktree(dest, commit, "--detach")
            if there and self._half_created(dest):        # an interrupted `worktree add`: rebuild it (_rebuild_worktree)
                self._rebuild_worktree(dest, commit)
            rc, head, _ = self._git(dest, "rev-parse", "HEAD")
            if there and rc == 0 and head.strip() != commit:
                self._heal_worktree(dest, head.strip(), commit)
                rc, head, _ = self._git(dest, "rev-parse", "HEAD")
        if rc != 0 or head.strip() != commit:
            raise RuntimeError(f"{dest} is at {head.strip() or '?'}, not {commit}")
        return dest

    def _heal_worktree(self, dest: Path, have: str, want: str) -> None:
        """Re-check-out the node worktree ``dest`` (at ``have``) at its node's commit ``want``. Live 11-09: the root
        was re-rooted (base_sha r12 45b541e9 -> fb 9035f616) while worktrees/<tag>/root, keyed by node id only, still
        stood at the old root, so every root replicate seed voided infra 8x on "is at 45b541e9, not 9035f616" and the
        noise-floor seeds never ran. Healed only when no tracked file is modified and no live SLURM job of ours has
        its cwd in it; otherwise one ALERT (per worktree/commit pair) and ``SubmissionRejected``: the row is void for
        good instead of retrying a deterministic mismatch. An squeue / git failure stays a plain (retried) error."""
        key = f"{dest}|{have}|{want}"
        rc, out, err = self._git(dest, "status", "--porcelain", "--untracked-files=no")
        if rc != 0:
            raise RuntimeError(f"{dest} is at {have}, not {want}; git status failed: {(err or out).strip()[-200:]}")
        why = f"{len(out.splitlines())} modified tracked file(s)" if out.strip() else ""
        if not why:
            jobs = self._live_jobs_in(dest, f"{dest} is at {have}, not {want}")
            why = f"live job(s) {','.join(jobs)} run in it" if jobs else ""
        if why:
            if key not in self._wt_alerted:
                self._wt_alerted.add(key)
                print(f"ALERT: node worktree {dest} is at {have}, not {want}, and was NOT re-checked-out ({why}); "
                      f"its submits are rejected until it is fixed by hand", file=sys.stderr, flush=True)
            raise SubmissionRejected(f"{dest} is at {have}, not {want} (not healed: {why})")
        with self._wt_lock.shared():
            rc, out, err = self._git(dest, "checkout", "--detach", "--quiet", want, timeout=600)
        if rc != 0:
            raise RuntimeError(f"{dest} is at {have}, not {want}; checkout failed: {(err or out).strip()[-300:]}")
        print(f"node worktree {dest} healed: {have[:12]} -> {want[:12]}", file=sys.stderr, flush=True)

    def _live_jobs_in(self, dest: Path, what: str) -> list[str]:
        """Our live SLURM jobs whose cwd is ``dest`` or below it (an squeue failure: a plain, retried error)."""
        rc, out, err = self.runner.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i %Z"], timeout=60)
        if rc != 0:
            raise RuntimeError(f"{what}; squeue failed: {(err or out).strip()[-200:]}")
        real = os.path.realpath(dest)
        return [j for j, _, c in (l.strip().partition(" ") for l in out.splitlines() if l.strip())
                if (r := os.path.realpath(c.strip())) == real or r.startswith(real + os.sep)]

    @staticmethod
    def _half_created(dest: Path) -> bool:
        """``dest`` is what a killed `git worktree add` leaves: its gitdir HEAD empty or missing, or its ``locked``
        file still "initializing" (git removes it when the add completes). Live 11-10 a node: HEAD written, checkout
        cut off before src/ (index.lock, no index), so rev-parse matched and both seeds crashed on the job guard
        ("nanobrain resolves to []"). Any other state (an unreadable .git, a rev-parse failure / timeout) is NOT
        this signature: never reaped (jobs write slurm/%j.out in the tree; a parallel submit may be in sbatch)."""
        try:
            gitdir = Path((dest / ".git").read_text().split("gitdir:", 1)[1].strip())
            gitdir = gitdir if gitdir.is_absolute() else dest / gitdir
            head = (gitdir / "HEAD").read_text().strip() if (gitdir / "HEAD").exists() else ""
            locked = (gitdir / "locked").read_text().strip() if (gitdir / "locked").exists() else ""
        except (OSError, IndexError):
            return False
        return not head or locked == "initializing"

    def _rebuild_worktree(self, dest: Path, commit: str) -> None:
        """Remove and re-add the half-created node worktree ``dest`` (:meth:`_half_created`). Live
        11-11: a restart at 00:09:12 killed the prewarm's `git worktree add` of a node mid-way, leaving ``.git``, an
        EMPTY gitdir HEAD and a ``locked`` "initializing" registration; every later submit then saw ``.git``, read
        HEAD as "?" and voided the node ("is at ?, not b3b66cb"). One with a valid HEAD can have run jobs (a node): a
        live job of ours in it blocks the rebuild (a plain, retried error); a dead one's slurm logs go with the tree."""
        what = f"{dest} is half-created (an interrupted worktree add), not at {commit}"
        if jobs := self._live_jobs_in(dest, what):
            raise RuntimeError(f"{what}; not rebuilt: live job(s) {','.join(jobs)} run in it")
        with self._wt_lock:                               # `remove --force --force` fails on a HEAD-less tree, and
            self._git(self.repo, "worktree", "unlock", str(dest))   # prune keeps a LOCKED registration: unlock it
        self.reap_worktree(Worktree(str(dest), "", ""))
        self._add_worktree(dest, commit, "--detach")
        print(f"node worktree {dest} rebuilt at {commit[:12]} (it was half-created)", file=sys.stderr, flush=True)

    def warm_worktree(self, node_id: str, commit: str) -> bool:
        """Create ``node_id``'s run worktree ahead of its first submit (the tick's housekeeping, on a background
        thread): live 10-14 tick 138 spent 193 s in launch.submit, ~25 s per NEW node's `git worktree add` under the
        worktree lock. True when it was created now; never raises."""
        if not commit or (self.worktrees / _safe(node_id) / ".git").exists():
            return False
        try:
            self._node_worktree(node_id, commit)
        except Exception:  # noqa: BLE001 - the submit creates it (and reports) as before
            return False
        return True

    def _sbatch(self, config: str, out: Path, name: str, *extra: str,
                pool: str = "") -> tuple[list[str], dict[str, str]]:
        """The repo launcher (stdin /dev/null, so it never prompts); Labless autosubmit disarmed."""
        env = {"WANDB_MODE": "offline", "SBATCH_JOB_NAME": name, "LABLESS_AUTOSUBMIT_FILE": ""}
        pools = self.ctx.cfg.pools                              # env overrides the script's #SBATCH lines
        p = next((q for q in pools if q.name == pool), pools[0])
        env.update({k: v for k, v in (("SBATCH_ACCOUNT", p.account), ("SBATCH_QOS", p.qos),
                                      ("SBATCH_PARTITION", p.partition),
                                      ("SBATCH_TIMELIMIT", self.ctx.cfg.slurm.time_limit)) if v})
        if p.preemptible:                   # = --requeue (slurm 26.05: any other value is refused)
            env["SBATCH_REQUEUE"] = ""
        return [self.LAUNCHER, config, f"output_dir={out}", *extra], env

    def oom_ladder(self, commit: str) -> tuple[str, ...]:
        """The numerics-neutral OOM mitigations a run can take (see EXPANDABLE)."""
        return (EXPANDABLE,)

    def _launch(self, run: Run) -> tuple[list[str], dict[str, str], Path]:
        """(argv, env, output dir) of the sbatch launch, run from the node's worktree; an OOM step adds
        the allocator env (the receipt keeps it). An eval-only run (``plan_eval``) resumes its source's
        latest.pt (or its shim, ``_shim``) from a config copy that sets train.resume and saves
        no checkpoint: the training loop is skipped, the readout is refit, the locked probe runs."""
        rel = self._rel(run)
        out = self.runs / rel
        cfg, seed = self.config_rel, run.seed
        if plan := self.eval_plan(run):
            tree = yaml.safe_load(self._show(self.repo, plan["commit"], self.config_rel) or "") or {}
            set_dotted(tree, "train.resume", str(out / SHIM_NAME) if plan.get("shim") else plan["resume"])
            set_dotted(tree, "train.save_every", None)
            path = self._plan_path(run.node_id, run.stage, run.seed).with_suffix(f".a{run.attempt}.yaml")
            path.write_text(yaml.safe_dump(tree, sort_keys=False))
            cfg, seed = str(path), plan["seed"]
        elif (self.resume_ok or run.id in self.resume_runs) and not self.no_resume(run) \
                and (src := self.worktrees / _safe(run.node_id) / self.config_rel).is_file():
            tree = yaml.safe_load(src.read_text()) or {}    # resumable: a config copy, armed once latest.pt exists
            if self.ctx.cfg.slurm.resume_save_every > 0:
                set_dotted(tree, "train.save_every", int(self.ctx.cfg.slurm.resume_save_every))
            if (out / "latest.pt").is_file():               # a re-launch of this attempt (repool) resumes too
                set_dotted(tree, "train.resume", str(out / "latest.pt"))
            if run.mitigation == ACT_CKPT:
                set_dotted(tree, ACT_CKPT_KEY, True)
            path = self._resume_cfg(rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(tree, sort_keys=False))
            cfg = str(path)
        elif run.mitigation == ACT_CKPT and (src := self.worktrees / _safe(run.node_id) / self.config_rel).is_file():
            tree = yaml.safe_load(src.read_text()) or {}    # a plain config copy with the flag on
            set_dotted(tree, ACT_CKPT_KEY, True)
            path = self.state / "oom" / f"{rel}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(tree, sort_keys=False))
            argv, env = self._sbatch(str(path), out, f"lp{self._tag}-{rel.replace('/', '-')}", f"seed={seed}",
                                     pool=run.pool)
            return argv, {**env, **ALLOC_ENV}, out
        argv, env = self._sbatch(cfg, out, f"lp{self._tag}-{rel.replace('/', '-')}",
                                 f"seed={seed}", pool=run.pool)
        if cfg != self.config_rel:
            env["WANDB_MODE"] = "disabled"          # resume forces wandb resume="must"
        if run.mitigation:
            env.update(ALLOC_ENV)
        return argv, env, out

    def _resume_cfg(self, rel: str) -> Path:
        return self.state / "resume" / f"{rel}.yaml"

    def _resume_rec(self, rel: str, rec: dict[str, Any], argv: Sequence[str], out: Path) -> None:
        """Record a resumable launch on its receipt: the config copy, armed (train.resume set) if latest.pt exists."""
        if argv[1] == str(self._resume_cfg(rel)):
            rec["resume_cfg"] = argv[1]
            if (out / "latest.pt").is_file():
                rec.update(resume_armed=rec.get("resume_armed") or self._clock(), resumed=rec.get("resumed", 0) + 1)

    def _arm(self, rel: str, rec: dict[str, Any], out: Path) -> None:
        """The run saved its first latest.pt: every later start of this job (a SLURM requeue, a free retry)
        resumes from it instead of wiping the output dir (train.py re-reads the config copy at each start)."""
        path = Path(rec["resume_cfg"])
        tree = yaml.safe_load(path.read_text()) or {}
        set_dotted(tree, "train.resume", str(out / "latest.pt"))
        path.write_text(yaml.safe_dump(tree, sort_keys=False))
        rec["resume_armed"] = self._clock()
        self._save(rel, rec)

    def resume_state(self, handle: JobHandle) -> dict[str, Any]:
        """{armed, resumed (times a start resumed from latest.pt), fraction (sample_fraction of the last train log),
        resumable (launched from a resume config copy: it arms at its first save -- v3 review O3)}."""
        rec, frac = self._receipt(self._rel_of(handle)), 0.0
        for line in reversed((_read(Path(handle.output_dir) / "metrics.jsonl") or "").splitlines()[-50:]):
            try:
                frac = float(json.loads(line)["sample_fraction"])
                break
            except (ValueError, KeyError, TypeError):
                continue
        return {"armed": bool(rec.get("resume_armed")), "resumed": int(rec.get("resumed", 0)), "fraction": frac,
                "resumable": bool(rec.get("resume_cfg"))}

    def interrupt(self, handle: JobHandle) -> str:
        """A forced preemption (the resume calibration): scancel, then the same launch as a new job, which resumes
        from latest.pt through the armed config copy. Returns the new job id."""
        rel = self._rel_of(handle)
        rec = self._receipt(rel)
        if not rec.get("resume_armed"):
            raise RuntimeError(f"{handle.run_id}: no latest.pt to resume from yet")
        self._scancel(handle)
        _, so, se = self.runner.run(rec["argv"], cwd=rec["worktree"], timeout=300, env=rec["env"])
        if not (m := re.search(r"Submitted batch job (\d+)", so)):
            raise RuntimeError(f"interrupt relaunch failed: {(se or so).strip()[-300:]}")
        rec.update(job_id=m.group(1), resumed=rec.get("resumed", 0) + 1, restarts=0, lost_s=0.0)
        self._save(rel, rec)
        return m.group(1)

    def command(self, run: Run) -> str:
        """The exact launch as one shell line (``--dry-run`` prints it; nothing runs)."""
        argv, env, _ = self._launch(run)
        return (f"cd {shlex.quote(str(self.worktrees / _safe(run.node_id)))} && "
                + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + shlex.join(argv))

    def lock_violations(self, commit: str) -> list[str]:
        """The ``surface.locked`` entries whose content at ``commit`` differs from the canonical copy at
        ``lock_ref``: whole files / trees byte-identical (``git diff``), ``file::`` spans by :func:`locked_source`.
        Fail closed: an unreadable commit or ref is a violation. Cached per commit (a commit is immutable)."""
        if (got := self._lock_ok.get(commit)) is not None:
            return list(got)
        locked = tuple(self.ctx.spec.surface.locked)
        whole, spans = [p.rstrip("/") for p in locked if "::" not in p], [p for p in locked if "::" in p]
        bad: list[str] = []
        if whole:
            rc, out, err = self._git(self.repo, "diff", "--name-only", self.lock_ref, commit, "--", *whole)
            bad += [f"<unreadable: {(err or out).strip()[-120:]}>"] if rc != 0 else [n for n in out.splitlines() if n]
        for span in spans:
            rel = span.split("::", 1)[0]
            new, ref_src = self._show(self.repo, commit, rel), self._show(self.repo, self.lock_ref, rel)
            want = locked_source(ref_src, span)
            if new is None or (want is None and ref_src is not None) or want != locked_source(new, span):
                bad.append(span.split("...", 1)[0][:100])      # an ambiguous canonical pin fails closed too
        bad += self.config_violations(commit)
        self._lock_ok[commit] = bad
        return list(bad)

    def config_violations(self, commit: str) -> list[str]:
        """v3 review O5, the launch-time config guard: the launched commit's config flattened and diffed against
        ``lock_ref``'s -- any ``surface.protected_keys`` dotpath (segment-safe) added, removed or changed, or a
        ``surface.cap_ceilings`` value ABOVE its ceiling (a shrink is fine: the smoke leg), is a violation. Every
        route into the YAML (a hand edit, a merge, a graft) meets it, not only the checked diffs. Fail closed: an
        unreadable / unparsable / missing config is a violation (a ref without one: no check)."""
        surf = self.ctx.spec.surface
        prot = tuple(getattr(surf, "protected_keys", ()) or ())
        ceil = dict(getattr(surf, "cap_ceilings", {}) or {})
        if not prot and not ceil:
            return []
        if (ref_text := self._show(self.repo, self.lock_ref, self.config_rel)) is None:
            return []                               # no reference config (a repo without one): nothing to diff
        try:
            new = flatten(yaml.safe_load(self._show(self.repo, commit, self.config_rel) or "") or {})
            ref = flatten(yaml.safe_load(ref_text) or {})
        except Exception as exc:  # noqa: BLE001 - fail closed
            return [f"config:<unreadable {self.config_rel}: {repr(exc)[:80]}>"]
        if not new and ref:
            return [f"config:<missing {self.config_rel}>"]
        bad = []
        okurls = self._swap_ok_urls()
        ic = self.ctx.cfg.init_compat
        allow, allow_types = tuple(getattr(ic, "allow", ()) or ()), tuple(getattr(ic, "allow_types", ()) or ())
        for k in sorted(set(new) | set(ref)):
            if any(k == pk or k.startswith(pk + ".") for pk in prot) and new.get(k, MISSING) != ref.get(k, MISSING):
                v = new.get(k, MISSING)
                if k == "model.pretrained_url" and (v in okurls or (
                        allow and isinstance(v, str) and any(re.search(p, v, re.I) for p in allow))):
                    continue            # v3 review M1: an init swap whose check passed, or an allowlisted Meta source
                bad.append(f"config:{k}")
        # 10-22 WEIGHTS ALLOWLIST: model.type is not a protected key, but it names the weights the model loads. With an
        # allowlist configured a changed type is accepted only when it is an allowlisted Meta type or a type whose
        # init-swap check (init_legal incl. the allowlist) passed and was recorded; anything else is a violation.
        def _eff(c: Mapping[str, Any]) -> Any:     # model.type as train.py folds it (`timm` + timm_name -> `timm:<id>`)
            t = c.get("model.type", MISSING)
            return f"timm:{c['model.timm_name']}" if t == "timm" and isinstance(c.get("model.timm_name"), str) else t
        if allow_types and _eff(new) != _eff(ref):
            t = _eff(new)
            if not (isinstance(t, str) and (f"type:{t}@{commit}" in okurls or any(re.search(p, t, re.I) for p in allow_types))):
                bad.append("config:model.type")
        for k, cap in ceil.items():
            v = new.get(k, MISSING)
            try:
                over = v is not MISSING and v is not None and float(v) > float(cap)
            except (TypeError, ValueError):
                over = True                         # a non-number where the budget is: fail closed
            if over:
                bad.append(f"cap:{k}={v!r}>{cap}")
        return bad

    def _lock_guard(self, run: Run, commit: str) -> None:
        """Refuse (SubmissionRejected: the row is void) a launch whose locked files differ from the canonical copy."""
        if not (bad := self.lock_violations(commit)):
            return
        msg = (f"LOCKED-FILE REFUSAL: {run.id} commit {commit[:12]} differs from {self.lock_ref[:12]} in locked "
               f"{bad[:6]}; not launched (void)")
        print(f"[loop] {msg}", file=sys.stderr, flush=True)
        try:
            self.state.mkdir(parents=True, exist_ok=True)
            with (self.state / "lock_refusals.jsonl").open("a") as f:
                f.write(json.dumps({"ts": self._clock(), "run": run.id, "node": run.node_id, "commit": commit,
                                    "ref": self.lock_ref, "paths": bad}) + "\n")
        except OSError:
            pass
        raise SubmissionRejected(msg)

    def submit(self, *, run: Run, commit: str) -> JobHandle:
        rel = self._rel(run)
        rec = self._receipt(rel)
        if self._job_id(rel, rec):                              # already launched: never twice
            return self._handle(run.id, rec)
        plan = self.eval_plan(run)
        self._lock_guard(run, plan["commit"] if plan else commit)
        wt = self._node_worktree(run.node_id + (f"~{plan['commit'][:10]}" if plan else ""),
                                 plan["commit"] if plan else commit)
        if plan:
            self._shim(wt, plan, self.runs / rel)
        argv, env, out = self._launch(run)
        rec = {"run_id": run.id, "attempt": run.attempt, "commit": plan["commit"] if plan else commit, "job_id": "",
               "job_name": env["SBATCH_JOB_NAME"], "output_dir": str(out),
               "worktree": str(wt), "submitted_ts": self._clock(), "argv": argv, "env": env}
        self._resume_rec(rel, rec, argv, out)
        self._save(rel, rec)                                    # receipt BEFORE sbatch
        rc, so, se = self.runner.run(argv, cwd=str(wt), timeout=300, env=env)
        m = re.search(r"Submitted batch job (\d+)", so)
        if m:
            rec["job_id"] = m.group(1)
            self._save(rel, rec)
        elif not self._job_id(rel, rec):
            msg = (se or so).strip()
            if rc != 0 and REJECTED.search(msg) and not TRANSIENT.search(msg):
                raise SubmissionRejected(f"sbatch rejected the submission (configuration error, "
                                         f"not retried): {msg[-300:]}")
            raise RuntimeError(f"sbatch launch failed (rc={rc}): {(se or so).strip()[-300:]}")
        return self._handle(run.id, rec)

    def find(self, run: Run) -> JobHandle | None:
        rel = self._rel(run)
        rec = self._receipt(rel)
        return self._handle(run.id, rec) if self._job_id(rel, rec) else None

    def _rel_of(self, handle: JobHandle) -> str:
        return Path(handle.output_dir).relative_to(self.runs).as_posix()

    def _scancel(self, handle: JobHandle) -> None:
        """Fire-and-forget scancel (the adapter's own freeze / requeue / repool paths: they relaunch or void anyway)."""
        rel = self._rel_of(handle)
        jid = self._job_id(rel, self._receipt(rel)) or handle.job_id    # the receipt follows free retries
        if jid:
            self.runner.run(["scancel", jid], timeout=60)

    def cancel(self, handle: JobHandle, only_pending: bool = False) -> None:
        """The loop's own cancel (``RunBook.cancel_run``): scancel, then CONFIRM via squeue that the job is gone or
        ending; raises when it is not (the runner then keeps the row RUNNING and retries next tick). ``only_pending``
        (the stale-pending release): ``scancel --state=PENDING``, so a job that started since the caller looked is left
        running -- and the confirm then raises, leaving the row RUNNING."""
        rel = self._rel_of(handle)
        jid = self._job_id(rel, self._receipt(rel)) or handle.job_id    # the receipt follows free retries
        if not jid:
            return
        rc, so, se = self.runner.run(["scancel", *(["--state=PENDING"] if only_pending else []), jid], timeout=60)
        # confirm: the job is gone (squeue: no row / invalid id) or ending (CG / CA); a scancel that failed AND a job
        # still live raises -- the runner then keeps the row RUNNING (never VOID) and retries it next tick
        qrc, q, qerr = self.runner.run(["squeue", "-h", "-j", jid, "-o", "%T"], timeout=60)
        states = {x.strip().upper() for x in (q or "").split() if x.strip()}
        gone = (qrc == 0 and not states) or (qrc != 0 and "invalid job id" in (qerr or q or "").lower())
        if gone or (qrc == 0 and states <= {"COMPLETING", "CANCELLED", "COMPLETED", "FAILED", "TIMEOUT"}):
            return
        raise RuntimeError(f"scancel {jid} not confirmed (scancel rc={rc}: {(se or so).strip()[-200:]}; "
                           f"squeue rc={qrc} state={sorted(states)} {(qerr or '').strip()[-100:]})")

    def release_receipt(self, handle: JobHandle) -> None:
        """Set the (cancelled) job's receipt aside so the SAME attempt can be submitted again: ``submit`` returns the
        receipt's job for an attempt it already launched ("never twice"), which after a stale-pending scancel would
        re-attach the row to the dead job. The receipt stays on disk as ``<name>.released-<ts>.json`` (nothing deleted)."""
        rel = self._rel_of(handle)
        path = self.state / "receipts" / f"{rel}.json"
        if path.is_file():
            path.rename(path.with_name(f"{path.stem}.released-{int(self._clock())}.json"))

    def repool(self, handle: JobHandle, run: Run) -> JobHandle:
        """Move a still-PENDING job to ``run.pool``: scancel it, launch the same run there as a new
        job (a free retry: same receipt, same attempt, same output dir). The old job's scancel is CONFIRMED first
        (:meth:`cancel`: squeue shows it gone / ending; raises otherwise, nothing relaunched -- never two live jobs
        writing one output dir)."""
        rel = self._rel_of(handle)
        rec = self._receipt(rel)
        self.cancel(handle)
        argv, env, out = self._launch(run)
        rc, so, se = self.runner.run(argv, cwd=rec["worktree"], timeout=300, env=env)
        if not (m := re.search(r"Submitted batch job (\d+)", so)):
            raise RuntimeError(f"repool sbatch failed (rc={rc}): {(se or so).strip()[-300:]}")
        self._resume_rec(rel, rec, argv, out)
        rec.update(job_id=m.group(1), argv=argv, env=env, pool=run.pool, restarts=0, lost_s=0.0, nice=0)
        self._save(rel, rec)
        return self._handle(run.id, rec)

    def pool_capacity(self) -> dict[str, dict[str, int]]:
        """The REAL schedulable GPU capacity of every pool whose association carries a ``GrpTRES gres/gpu`` limit
        (live 10-21: the ``training`` user association caps user at 8 GPUs; our other jobs there -- the SPECTRA
        sweeps -- held 6, so the 6 declared primary slots left 2 and loop screens / smokes sat at AssocGrpGRES for
        30-120 min). Per pool: ``{limit, others, loop, effective}`` = the limit, the GPUs held by this user's RUNNING
        jobs in the pool's account that are NOT the loop's (``lp<tag>-*`` runs and smokes, ``diag-*``), the loop's own,
        and ``limit - others`` (floored at 0). A pool without a limit is absent (no cap); any probe failure returns {}
        (the declared slots stand). Read-only: sacctmgr + squeue; never touches a job."""
        user, out = getpass.getuser(), {}
        mine = (f"lp{self._tag}", "diag-")
        now = self._clock()
        if now < self._cap_retry_at:                 # slurmdbd / slurmctld slow or down: not asked again for a while
            return {}
        for p in self.ctx.cfg.pools:
            if not p.account:
                continue
            rc, so, _ = self.runner.run(["sacctmgr", "-n", "-P", "show", "assoc", "where", f"user={user}",
                                         f"account={p.account}", "format=Account,GrpTRES"], timeout=CAP_PROBE_TIMEOUT_S)
            if rc != 0:
                self._cap_retry_at = now + CAP_PROBE_BACKOFF_S
                return {}
            limits = [int(m.group(1)) for line in so.splitlines() if line.split("|")[0].strip() == p.account
                      for m in [re.search(r"gres/gpu=(\d+)", line)] if m]
            if not limits:
                continue
            rc, sq, _ = self.runner.run(["squeue", "-h", "-u", user, "-A", p.account, "-t", "RUNNING,COMPLETING,SUSPENDED",
                                         "-O", "JobID:16|,Name:200|,tres-alloc:200"], timeout=CAP_PROBE_TIMEOUT_S)
            if rc != 0:
                self._cap_retry_at = now + CAP_PROBE_BACKOFF_S
                return {}
            held, ours = 0, 0
            for line in sq.splitlines():
                parts = [f.strip() for f in line.split("|")]
                if len(parts) < 3:
                    continue
                m = re.search(r"gres/gpu=(\d+)", parts[2])
                n = int(m.group(1)) if m else 0
                if parts[1].startswith(mine):
                    ours += n
                else:
                    held += n
            lim = min(limits)
            out[p.name] = {"limit": lim, "others": held, "loop": ours, "effective": max(0, lim - held)}
        return out

    def pending_reason(self, handle: JobHandle) -> str:
        """SLURM's PENDING reason of the run's job ('' = not pending / unknown): squeue only."""
        rel = self._rel_of(handle)
        jid = self._job_id(rel, self._receipt(rel)) or handle.job_id
        if not jid:
            return ""
        rc, q, _ = self.runner.run(["squeue", "-h", "-j", jid, "-O", "State:20|,Reason:100"], timeout=60)
        if rc != 0 or not q.strip():
            return ""
        state, _, reason = q.strip().splitlines()[0].partition("|")
        return reason.strip().rstrip("|").strip() if state.strip() == "PENDING" else ""

    _NOT_STARTABLE = ("drain", "drng", "down", "fail", "maint", "resv", "reserved", "inval", "unk", "noresp", "power", "plan")

    def free_gpus(self) -> dict[str, int]:
        """Partition -> GPUs that could START a job now (10-22: 50 screens went to an empty idle pool in one tick while a
        congested partition had ~3 free). ``sinfo`` per node: configured minus allocated GPUs of every node whose state
        is not drained / draining / down / failed / reserved / planned / unresponsive. Read-only. Raises on a probe
        failure (the run book falls back to ``slurm.max_new_per_tick``)."""
        out: dict[str, int] = {}
        for part in sorted({p.partition for p in self.ctx.cfg.pools if p.partition}):
            rc, so, err = self.runner.run(["sinfo", "-p", part, "-h", "-N", "-O",
                                           "NodeList:24,Gres:60,GresUsed:80,StateLong:24"], timeout=60)
            if rc != 0:
                raise RuntimeError(f"sinfo -p {part} failed rc={rc}: {err.strip()[:200]}")
            free = 0
            for line in so.splitlines():
                m = re.match(r"\s*(\S+)\s+(\S.*?)\s{2,}(\S.*?)\s{2,}(\S+)\s*$", line)
                if not m:
                    continue
                state = m.group(4).lower()
                if any(k in state for k in self._NOT_STARTABLE) or state[-1:] in "*~#$@-":
                    continue
                tot = re.search(r"gpu(?::[^:(\s]+)?:(\d+)", m.group(2))
                used = re.search(r"gpu(?::[^:(\s]+)?:(\d+)", m.group(3))
                free += max(0, int(tot.group(1)) - (int(used.group(1)) if used else 0)) if tot else 0
            out[part] = free
        return out

    def pending_jobs(self, names: Sequence[str]) -> list[str]:
        """The job names among ``names`` (anyone's) SLURM holds waiting -- PENDING, or preempted and
        requeued (a requeue passes through these states before it is PENDING again) -- that could plausibly
        start soon: a PENDING reason in ``slurm.protect_reasons`` (a requeue state counts) and fewer than
        ``slurm.protect_max_restarts`` requeues. A job SLURM cannot place (ReqNodeNotAvail, a reservation)
        or one that is requeue-looping gains nothing from our idle slots."""
        rc, out, err = self.runner.run(["squeue", "-h", "-t", PROTECT_WAITING, "-n", ",".join(names),
                                        "-O", "Name:200|,State:20|,Reason:200|,RestartCnt:8"], timeout=60)
        if rc != 0:
            raise RuntimeError(f"squeue failed: {err.strip()[-200:]}")
        sl, found = self.ctx.cfg.slurm, set()
        for line in out.splitlines():
            name, state, reason, restarts = ([f.strip() for f in line.split("|")] + ["", "", ""])[:4]
            if name not in names:
                continue
            looping = sl.protect_max_restarts > 0 and restarts.isdigit() and int(restarts) >= sl.protect_max_restarts
            if not looping and (state != "PENDING" or not sl.protect_reasons
                                or reason.split(",")[0] in sl.protect_reasons):
                found.add(name)
        return sorted(found)

    # ------------------------------------------------------------ poll
    def _read_scores(self, out: Path, since: float) -> tuple[dict[str, float], float | None] | None:
        """Axes from ``summary.json``, only if written after the run was submitted (never stale)."""
        path = out / "summary.json"
        data = _json(path) if path.is_file() and path.stat().st_mtime >= since else None
        if not isinstance(data, dict):
            return None
        scores: dict[str, float] = {}
        for axis in self.ctx.cfg.axes:
            try:
                scores[axis] = float(data[axis])
            except (KeyError, TypeError, ValueError):
                return None
            if not math.isfinite(scores[axis]):
                return None
        agg = data.get(self.ctx.spec.slots.score_key)
        agg = float(agg) if isinstance(agg, (int, float)) and math.isfinite(agg) else None
        return scores, agg

    def _logs(self, rec: Mapping[str, Any], jid: str) -> list[Path]:
        return [Path(rec.get("worktree", "")) / "slurm" / f"{jid}.{ext}" for ext in ("out", "err")]

    def _heartbeat(self, out: Path, rec: Mapping[str, Any], jid: str) -> float | None:
        paths = self._logs(rec, jid) + [Path(d) / f for d, _, fs in os.walk(out) for f in fs]
        return max((p.stat().st_mtime for p in paths if p.exists()), default=None)

    def _tail(self, rec: Mapping[str, Any], jid: str) -> str:
        return "\n".join(line for p in self._logs(rec, jid)
                         for line in (_read(p) or "").splitlines()[-40:])

    def _lost_s(self, jid: str) -> float:
        """Seconds the job ran before each of its SLURM requeues (sacct -D: one row per instance)."""
        rc, out, _ = self.runner.run(["sacct", "-n", "-D", "-X", "-P", "-j", jid, "-o", "State,ElapsedRaw"], timeout=60)
        rows = [line.split("|") for line in out.splitlines() if "|" in line] if rc == 0 else []
        return float(sum(int(e) for s, e in rows if s.split()[:1] and s.split()[0] in REQUEUE_STATES and e.isdigit()))

    def nice(self, handle: JobHandle, nice: int) -> None:
        """Lower a queued job's priority (a LOW run on a preemptible pool); free retries keep it."""
        rel = self._rel_of(handle)
        rec = self._receipt(rel)
        jid = self._job_id(rel, rec) or handle.job_id
        rc, out, err = self.runner.run(["scontrol", "update", f"JobId={jid}", f"Nice={int(nice)}"], timeout=60)
        if rc != 0:
            raise RuntimeError(f"scontrol nice failed: {(err or out).strip()[-200:]}")
        if rec:
            rec["nice"] = int(nice)
            self._save(rel, rec)

    def _free_retry(self, rel: str, rec: dict[str, Any], jid: str, why: str, gpu_h: float = 0.0) -> JobStatus | None:
        """The machine was taken away: the same launch again as a new job (not an infra void)."""
        if not rec.get("argv") or rec.get("requeues", 0) >= MAX_FREE_RETRIES:
            return None
        self.runner.run(["scancel", jid], timeout=60)           # never two copies (a SLURM requeue)
        _, so, _ = self.runner.run(rec["argv"], cwd=rec["worktree"], timeout=300, env=rec["env"])
        if not (m := re.search(r"Submitted batch job (\d+)", so)):
            return None
        rec.update(job_id=m.group(1), requeues=rec.get("requeues", 0) + 1, restarts=0, lost_s=0.0,
                   resumed=rec.get("resumed", 0) + bool(rec.get("resume_armed")))
        self._save(rel, rec)                                    # a new job: the freeze clock restarts
        if rec.get("nice"):
            self.runner.run(["scontrol", "update", f"JobId={m.group(1)}", f"Nice={rec['nice']}"], timeout=60)
        return JobStatus(RunStatus.RUNNING, gpu_h=gpu_h, detail=f"slurm {why}: free retry {rec['requeues']} as job {m.group(1)}")

    def _resumed(self, out: Path, rec: Mapping[str, Any]) -> int:
        """Starts of the run that resumed from a checkpoint so far (JobStatus.resumed: seeds / world noise treat a
        resumed run as a non-exact replicate). nanopath: the receipt's count; a project whose trainer resumes on its
        own (nanobrain) counts the job's resumed instances."""
        return int(rec.get("resumed", 0))

    def _classify(self, out: Path, rec: Mapping[str, Any], jid: str, state: str, gpu_h: float, tail: str,
                  since: float) -> JobStatus | None:
        """A finished job without scores, before the TIMEOUT / COMPLETED / CRASH / INFRA split: a project adapter's
        own verdict (None = the split decides; nanopath has none)."""
        return None

    def poll(self, handle: JobHandle) -> JobStatus:
        rel = self._rel_of(handle)
        rec = self._receipt(rel) or {"submitted_ts": handle.submitted_ts}
        jid = self._job_id(rel, rec) or handle.job_id
        if not jid:
            return JobStatus(RunStatus.VOID, void_type=VoidType.INFRA, detail="no slurm job id")
        out, now = Path(handle.output_dir), self._clock()
        rc, q, err = self.runner.run(["squeue", "-h", "-j", jid, "-O", SQUEUE_FIELDS], timeout=60)
        if rc != 0 and "invalid job id" not in err.lower():
            raise RuntimeError(f"squeue failed: {err.strip()[-200:]}")  # unknown is not a verdict
        hb = self._heartbeat(out, rec, jid)
        if rc == 0 and q.strip():
            state, start, reason, restarts = ([f.strip() for f in q.strip().splitlines()[0].split("|")] + ["", "", ""])[:4]
            restarts = int(restarts) if restarts.isdigit() else 0
            if "restarts" not in rec and rec.get("run_id"):     # the first reading is the baseline: a receipt from
                rec.update(restarts=restarts, lost_s=self._lost_s(jid) if restarts else 0.0)   # before the
                self._save(rel, rec)                            # telemetry never backfills its old requeues
            new, lost = restarts - int(rec.get("restarts", 0)), 0.0
            if new > 0 and rec.get("run_id"):                   # SLURM requeued it under the same id: preempted
                lost = self._lost_s(jid) - float(rec.get("lost_s", 0.0))
                rec.update(restarts=restarts, lost_s=float(rec.get("lost_s", 0.0)) + max(0.0, lost),
                           resumed=rec.get("resumed", 0) + (new if rec.get("resume_armed") else 0))
                self._save(rel, rec)
            if rec.get("resume_cfg") and not rec.get("resume_armed") and (out / "latest.pt").is_file():
                self._arm(rel, rec, out)
            requeue = {"restarts": restarts, "requeued": max(0, new), "gpu_h": max(0.0, lost) / 3600,
                       "resumed": self._resumed(out, rec)}
            if state == "PENDING" and reason.startswith("JobHeld"):   # held after MaxBatchRequeue requeues
                if st := self._free_retry(rel, rec, jid, reason):
                    return st
                self.runner.run(["scancel", jid], timeout=60)   # retries spent: never hold a slot forever
                return JobStatus(RunStatus.VOID, void_type=VoidType.INFRA, log_tail=self._tail(rec, jid),
                                 detail=f"slurm held ({reason}), free retries exhausted; scancelled")
            try:                                                # PENDING has no start
                started = time.mktime(time.strptime(start, "%Y-%m-%dT%H:%M:%S"))
            except ValueError:
                started = now
            if state == "RUNNING" and now - max(started, hb or 0.0) > self.freeze_s:
                self._scancel(handle)
                return JobStatus(RunStatus.VOID, void_type=VoidType.INFRA,
                                 log_tail=self._tail(rec, jid),
                                 detail=f"frozen: no progress for {self.freeze_s:.0f}s; scancelled")
            return JobStatus(RunStatus.RUNNING, detail=state, **requeue)
        _, acct, _ = self.runner.run(["sacct", "-n", "-X", "-P", "-j", jid, "-o",
                                      "State,ElapsedRaw"], timeout=60)
        state, _, elapsed = (acct.strip().splitlines() or [""])[0].partition("|")
        state = (state.split() or [""])[0]
        gpu_h = int(elapsed) / 3600 if elapsed.strip().isdigit() else 0.0
        scored = self._read_scores(out, handle.submitted_ts)
        if scored:
            return JobStatus(RunStatus.DONE, scores=scored[0], aggregate=scored[1],
                             gpu_h=gpu_h, detail=f"slurm {state}", resumed=self._resumed(out, rec))
        if state in WAIT_STATES:                                # accounting lags squeue
            return JobStatus(RunStatus.RUNNING, detail=f"sacct {state}")
        if state in FREE_RETRY_STATES and (st := self._free_retry(rel, rec, jid, state, gpu_h)):
            return st
        tail = self._tail(rec, jid)
        if st := self._classify(out, rec, jid, state, gpu_h, tail, handle.submitted_ts):
            return st                                           # a project's own failure typing (nanobrain)
        if state == "TIMEOUT":                                  # the candidate is too slow: a verdict
            why = f"TOO SLOW: killed at the {self.ctx.cfg.slurm.time_limit or 'SLURM'} time limit after {gpu_h:.2f} h"
            return JobStatus(RunStatus.VOID, void_type=VoidType.TOO_SLOW, gpu_h=gpu_h,
                             log_tail=f"{why}\n{tail}", detail=why)
        if state == "COMPLETED":
            return JobStatus(RunStatus.VOID, void_type=VoidType.EVAL, gpu_h=gpu_h, log_tail=tail,
                             detail=f"completed without axes {list(self.ctx.cfg.axes)}")
        if state in CRASH_STATES and (is_oom(tail) or not any(sig in tail.lower() for sig in HARDWARE)):
            return JobStatus(RunStatus.CRASH, gpu_h=gpu_h, log_tail=tail, detail=f"slurm {state}")
        return JobStatus(RunStatus.VOID, void_type=VoidType.INFRA, gpu_h=gpu_h, log_tail=tail,
                         detail=f"slurm {state or 'state unknown'}")
