"""loop.adapters.nanobrain -- ExecutionAdapter for nanobrain (MAE-3D pretraining on FOMO300K, scored by the
brainmarks-smri "open" task group). Multi-project Stage 1 (design-multiproject §3).

Everything git / worktree / receipt / poll / free-retry is :class:`NanopathV2Adapter`'s. What differs:

- the job: the loop's own ``nanobrain_job.sbatch`` (never the repo's scripts/train.sbatch, which calls
  scripts/submit.py): ``CFG OUT SEED AUTO_RESUME REPO [MODE]``; train then eval in one job, requeue-safe;
- resume: nanobrain resumes in-trainer (``auto_resume``); TEST runs pass ``auto_resume=false`` (``no_resume``);
  no resume config copies, no eval-only lane, no OOM ladder;
- smoke: a short train (:attr:`SMOKE_OVERRIDES`) + a checkpoint load / forward check in the same job; no probe
  smoke, no FLOP measurement; steps/s recorded;
- scores: ``OUT/eval/aggregate.json``: family means of the project axes (``loop.project.axes[*].tasks``), the
  aggregate = brainmarks ``summary.open.score``, which the weighted family composite must reproduce;
- failure typing: an l-bfgs non-convergence AssertionError in the eval = VOID(eval, "eval_lbfgs"); a completed
  job with an incomplete aggregate = VOID(eval); a training crash = CRASH (the base split);
- init compat: no construct probe; a denied pretrained-weights source (``init_compat.deny`` + :data:`NB_INIT_DENY`)
  in an added line is init_illegal.

Execution integrity (F1): the job script keeps ``OUT.loopmeta/instances.jsonl`` (train / eval phase starts and ends
with ``date +%s``); from it the adapter enforces the 60-minute wall budget across every SLURM instance
(:meth:`budget`), counts resumed instances (:meth:`_resumed`), refuses an eval dir the training code wrote
(``eval_fabricated``), and scores ONLY from the per-task jsons, each validated (task / checkpoint / commit / time)
and RE-SCORED here (:func:`task_score`, brainmarks' point estimates) -- aggregate.json is a cross-check, never the
source. A launch refuses binary / oversized files and foreign top-level packages under src/ (:meth:`lock_violations`).
TEST runs go to the non-preemptible pool; a TIMEOUT after training is an eval continuation into the same OUT; an
l-bfgs non-convergence gets one eval-only retry, then one fresh training attempt, then is the eval_lbfgs verdict."""

from __future__ import annotations

import copy
import datetime
import json
import os
import math
import posixpath
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from loop.adapters import flatten, set_dotted
from loop.adapters.nanopath_v2 import (ALLOC_ENV, NO_VENV, OFFLINE_ENV, NanopathV2Adapter, _json,
                                       _project_value, _read, _safe)
from loop.contracts import InitCompat, JobHandle, JobStatus, Run, RunStatus, SmokeResult, VoidType, Worktree, is_oom
from loop.spec import LoopContext

JOB_SCRIPT = Path(__file__).with_name("nanobrain_job.sbatch")
#: the brainmarks l-bfgs probe fit gave up (logistic.py: ``assert n_iter < max_iter``): an eval verdict, not a crash
LBFGS = re.compile(r"AssertionError.*(lbfgs|l-bfgs|converg)", re.I)
LOADCHECK_OK = "nanobrain_loadcheck: ok"
LOADCHECK_FAILED = "nanobrain_loadcheck: FAILED"
#: train.py's throughput meters (SmoothedValue window 1) print ``nan`` on step 0 of EVERY run (no elapsed time yet):
#: ``vol/s: nan (nan)  tflop/s: nan (nan)  watts: nan (nan)``. 11-08: the smoke's NaN scan read those as a diverged
#: loss and failed a node / a node, whose training finished with finite losses. They are dropped before the scan.
THROUGHPUT_METERS = re.compile(r"\b(?:vol/s|tflop/s|watts|vol_per_sec|tflop_per_sec)\"?\s*:\s*(?:nan|-?inf(?:inity)?|-?[\d.]+)"
                               r"(?:\s*\((?:nan|-?inf(?:inity)?|-?[\d.]+)\))?", re.I)
#: a non-finite number as python / json / torch print it: nan, inf, -inf, Infinity, -Infinity (review fix: Infinity)
NONFINITE = re.compile(r"\b(?:nan|inf(?:inity)?)\b", re.I)


def nonfinite(text: str) -> bool:
    """A smoke log shows a NaN / inf outside the throughput meters (a loss / grad / lr that diverged)."""
    return bool(NONFINITE.search(THROUGHPUT_METERS.sub(" ", text or "")))


#: the weighted family composite must equal brainmarks' own open score to this (the panel IS the leaderboard number)
OPEN_TOL = 1e-6
#: the job script's bookkeeping dir beside OUT (instances.jsonl, heartbeat): written by the loop's script, not train.py
META_SUFFIX = ".loopmeta"
#: C1 wall budget. Per train instance the job script's `date +%s` brackets also hold what train.py's own clock (from
#: the start of main()) does not: interpreter start + `import torch / nanobrain / wandb / matplotlib` (21-24 s warm on
#: a login node, measured 11-08; cold on a compute node up to ~60 s) and, after the last step, the summary write and
#: process exit (the final plot + checkpoint save are INSIDE train.py's wall). A real end-to-end CPU run of this job
#: script (11-08, login node) measured 28 s before main() + 3 s after the summary = 31 s. 120 s per instance covers a
#: cold start twice over and leaks at most 3% of the hour to pre-main work.
SETUP_ALLOWANCE_S = 120.0
#: train.py stops at the first step with wall >= time_limit, then plots and saves the final checkpoint before it reads
#: its clock for summary.wall_seconds: the smoke 579012 ran 307.3 s for a 300 s limit (+7 s); 60 s is 8x that.
SUMMARY_ALLOWANCE_S = 60.0
#: H2: a TIMEOUT after training finished = the eval ran out of job time; the same job again into the same OUT skips
#: the training (instances.jsonl train_done) and every finished task. Twice, then it is a verdict (the candidate's
#: encoder is too slow to evaluate).
EVAL_CONTINUATIONS = 2
#: C3: no candidate may add weights / data blobs to the repo
MAX_FILE_BYTES = 1_000_000
#: C3: weight-loading idioms refused in every added line of every writable file (data.py / misc.py included) on top
#: of the task's init_compat.deny; module-to-module copies (an EMA teacher's load_state_dict(student.state_dict()))
#: and torch.utils.checkpoint (activation checkpointing) stay legal.
NB_INIT_DENY = (
    r"torch\.load|safetensors|\bload_file\s*\(|pretrained\s*=\s*True|/data/|\.pth\b|\.ckpt\b|checkpoint[-_]last"
    r"|np\.load\s*\(|pickle\.load|joblib|urllib|requests\.(get|post)|\bwget\b|\bcurl\b",
    r"(?<!def )(?<!opt\.)(?<!optimizer\.)load_state_dict\((?!\s*[\w.]+\.state_dict\(\))",
)
#: the brainmarks task json fields that identify the run (C4)
EVAL_NAME, EVAL_MODEL, EVAL_SUBSET = "eval", "vit_mae_3d", "dev"
FAILED_TASK = re.compile(rf"\b{EVAL_NAME} (\w+) failed\b")


# ------------------------------------------------------------ brainmarks point scores (aggregate.py, re-implemented)
def _ranks(xs: Sequence[float]) -> list[float]:
    """1-based ranks, ties averaged (scipy.stats.rankdata 'average')."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def _auroc(labels: Sequence[int], probs: Sequence[Sequence[float]]) -> float:
    """brainmarks metrics.auroc: binary = class 1; multiclass = macro one-vs-rest; NaN if a class is missing."""
    n_classes = len(probs[0])
    if len(set(labels)) < n_classes:
        return float("nan")
    vals = []
    for c in ([1] if n_classes == 2 else range(n_classes)):
        pos = [lab == c for lab in labels]
        n_pos = sum(pos)
        n_neg = len(labels) - n_pos
        ranks = _ranks([float(p[c]) for p in probs])
        u = sum(r for r, is_pos in zip(ranks, pos) if is_pos) - n_pos * (n_pos + 1) / 2
        vals.append(u / (n_pos * n_neg))
    return sum(vals) / len(vals)


def task_score(result: Mapping[str, Any]) -> float:
    """The brainmarks task score of one task json (aggregate.score_auroc / score_r2 / score_dice point estimate):
    classification 2*AUROC-1, regression R^2, segmentation nanmean Dice. ValueError on anything else."""
    kind, samples, target = result.get("type"), result.get("eval_samples"), result.get("target")
    if not isinstance(samples, list) or not samples:
        raise ValueError("no eval_samples")
    if kind == "classification":
        classes = list(result["classes"])
        labels = [classes.index(s[target]) for s in samples]
        probs = [list(s["probabilities"].values()) for s in samples]
        if any(len(p) != len(classes) for p in probs):
            raise ValueError("probabilities do not match the classes")
        return 2 * _auroc(labels, probs) - 1
    if kind == "regression":
        t = [float(s[target]) for s in samples]
        pr = [float(s["prediction"]) for s in samples]
        mean = sum(t) / len(t)
        return 1 - sum((a - b) ** 2 for a, b in zip(pr, t)) / sum((a - mean) ** 2 for a in t)
    if kind == "segmentation":
        d = [float(s["dice"]) for s in samples if s.get("dice") is not None]
        d = [x for x in d if not math.isnan(x)]
        if not d:
            raise ValueError("no finite dice")
        return sum(d) / len(d)
    raise ValueError(f"task type {kind!r} is not scored by the loop")


#: the job guard's line when the nanobrain package is not importable at all (see NanobrainAdapter._tree_incomplete)
GUARD_EMPTY = "nanobrain_guard: FAILED nanobrain resolves to [],"


class NanobrainAdapter(NanopathV2Adapter):
    # fallbacks; the task's loop.project block (base_sha / runs_root / worktrees_root / config_rel) wins
    BASE_SHA = "440fc92a0f949c5874db64ec6491f397bcbda37c"           # 1 hour training budget, account training
    RUNS_ROOT = os.environ.get("NANOBRAIN_RUNS_ROOT") or "/path/to/runs"
    WORKTREES_ROOT = os.environ.get("NANOBRAIN_WORKTREES") or "/path/to/worktrees"
    CONFIG_REL = "src/nanobrain/config/default_pretrain.yaml"
    LAUNCHER = str(JOB_SCRIPT)                                       # absolute: Path(wt) / LAUNCHER == LAUNCHER
    #: warmup / decay are NOT fixed here: smoke_config scales every step-counted key by 300 / total_steps (H4).
    #: plot_period 0 = plotting OFF (train.py: ``is_plot_step = args.plot_period and ...``). 11-08: it was 1_000_000,
    #: which is truthy, and train.py:114 asserts ``not (native and plot_period)``: every smoke of the native root
    #: (the leader recipe, plot_period 0) died at startup rc=1 (a node / a node and 6 later smokes)
    SMOKE_OVERRIDES: dict[str, Any] = {"time_limit": 3, "total_steps": 300, "checkpoint_period": 300,
                                       "num_workers": 4, "plot_period": 0}
    FLOPS_RE = None
    MEMORY_KEEP = ("batch_size",)
    # loop.selfcheck: train.py and the default config it merges the run's config over (``train_config``), and a
    # passing smoke's log the NaN scan must accept (``nan_scan``; it carries the step-0 ``vol/s: nan`` meters)
    TRAIN_REL = "src/nanobrain/train.py"
    # loop.selfcheck train_config: build the model from the merged config (binds ``m``) + the eval-data env var
    SELFCHECK_MODEL = "from nanobrain.model import ViTMAE3D\nm = ViTMAE3D.from_config(cfg)"
    SELFCHECK_ENV = {"BRAINMARKS_SMRI_DATA": "{data}"}
    SELFCHECK_OVERRIDES = ("output_dir={out}", "name=selfcheck", "seed={seed}", "auto_resume=false")
    DEFAULT_CONFIG_REL = "src/nanobrain/config/default_pretrain.yaml"
    KNOWN_GOOD_SMOKE_LOG = str(Path(__file__).with_name("fixtures") / "nanobrain_smoke_ok.log")
    SMOKE_MIN = 20                     # smoke SLURM minutes: ~1-2 min setup + compile, <=3 min train, ~30 s check
    #: brainmarks open = the plain mean of these 11 tasks; the family weights 6/2/3 are their counts
    DEFAULT_FAMILIES: tuple[tuple[str, float, tuple[str, ...]], ...] = (
        ("nb_cls_mean", 6.0, ("abide_diagnosis", "adhd_diagnosis", "cnp_diagnosis", "ucsf_idh", "upenn_survival_1y",
                              "soop_mrs_poor")),
        ("nb_reg_mean", 2.0, ("ixi_age", "openbhb_age")),
        ("nb_seg_mean", 3.0, ("brats_enhancing_tumor", "brats_whole_tumor", "soop_lesion")),
    )
    AGGREGATE_GROUP = "open"

    def __init__(self, ctx: LoopContext, **kw: Any) -> None:
        super().__init__(ctx, **kw)
        self.preflight = None              # nanopath's torch-hub / pretrained-weights preflight does not apply
        # loop.project.python: a re-rooted base whose deps differ (FA3 / synthseg) has its own venv; "" = <repo>/.venv
        self.python = _project_value(ctx, "python", "")
        if self.python:
            self.venv_root = Path(self.python).parent.parent

    # ------------------------------------------------------------ project
    def families(self) -> tuple[tuple[str, float, tuple[str, ...]], ...]:
        """(summary key, raw weight, brainmarks tasks) per axis, from ``loop.project.axes`` (DEFAULT_FAMILIES
        when the task declares none)."""
        def g(a: Any, k: str, d: Any = None) -> Any:
            return a.get(k, d) if isinstance(a, Mapping) else getattr(a, k, d)
        axes = tuple(g(getattr(self.ctx.cfg, "project", None), "axes", ()) or ())
        fams = tuple((str(g(a, "key")), float(g(a, "weight", 0.0) or 0.0), tuple(g(a, "tasks", ()) or ()))
                     for a in axes if g(a, "tasks"))
        return fams or self.DEFAULT_FAMILIES

    # ------------------------------------------------------------ launch
    def _sbatch(self, config: str, out: Path, name: str, *extra: str,
                pool: str = "") -> tuple[list[str], dict[str, str]]:
        """``sbatch nanobrain_job.sbatch CFG OUT SEED AUTO_RESUME REPO MODE`` (extra = seed, auto_resume, mode);
        the env is the base's (job name, pool account / qos / partition, time limit, requeue on a preemptible pool)."""
        _, env = super()._sbatch(config, out, name, pool=pool)
        if self.python:
            env["NB_PYTHON"] = self.python             # the job script's PY (and the GUARD's venv) instead of REPO/.venv
        seed, auto, mode = (list(extra) + ["0", "true", "full"][len(extra):])[:3]
        return ["sbatch", self.LAUNCHER, config, str(out), str(seed), str(auto), str(self.repo), str(mode)], env

    def _node_worktree(self, node_id: str, commit: str) -> Path:
        dest = super()._node_worktree(node_id, commit)
        (dest / "slurm").mkdir(exist_ok=True)          # the job's --output=slurm/%j.out: SLURM makes no dirs
        return dest

    #: the FOMO300K image lists a decay-phase list (``anneal_filelist``) may be drawn from (whichever the base has)
    IMAGE_LISTS = ("src/nanobrain/config/fomo300k_images.txt", "src/nanobrain/config/fomo300k_images_qc.txt")
    ANNEAL_DIR = "src/nanobrain/config/"

    def _image_universe(self) -> set[str]:
        """The union of the base's FOMO300K image lists (read at ``lock_ref``; the lists are locked, so every
        commit carries the same ones). Cached."""
        if (got := getattr(self, "_universe", None)) is None:
            got = set()
            for rel in self.IMAGE_LISTS:
                text = self._show(self.repo, self.lock_ref, rel)
                got |= {l.strip() for l in (text or "").splitlines() if l.strip()}
            self._universe = got
        return got

    def anneal_violations(self, commit: str) -> list[str]:
        """``anneal_filelist``, when set, must be a path under ``src/nanobrain/config/`` whose lines are a subset of
        the FOMO300K image lists (the decay phase may re-weight the data, never add non-FOMO300K images). Fail
        closed: a missing / unreadable list, a path escaping the directory, or an empty universe is a violation."""
        try:
            cfg = flatten(yaml.safe_load(self._show(self.repo, commit, self.config_rel) or "") or {})
        except Exception:  # noqa: BLE001 - the parent's check names an unreadable config
            return []
        val = cfg.get("anneal_filelist")
        if val in (None, "", False):
            return []
        if not isinstance(val, str):
            return [f"anneal_filelist:<not a path: {val!r}>"]
        norm = posixpath.normpath(val)
        if not norm.startswith(self.ANNEAL_DIR) or norm != val.rstrip("/") or ".." in val.split("/"):
            return [f"anneal_filelist:<{val!r} is not a file under {self.ANNEAL_DIR}>"]
        text = self._show(self.repo, commit, norm)
        if text is None:
            return [f"anneal_filelist:<{norm} does not exist at the commit>"]
        universe = self._image_universe()
        if not universe:
            return ["anneal_filelist:<no FOMO300K image list in the base to check against>"]
        extra = [l.strip() for l in text.splitlines() if l.strip() and l.strip() not in universe]
        return [f"anneal_filelist:<{len(extra)} line(s) outside the FOMO300K lists, e.g. {extra[0][:80]!r}>"] if extra else []

    def config_violations(self, commit: str) -> list[str]:
        """The shared protected-key / cap-ceiling guard plus the anneal list's subset check."""
        return super().config_violations(commit) + self.anneal_violations(commit)

    def oom_ladder(self, commit: str) -> tuple[str, ...]:
        return ()

    def _launch(self, run: Run) -> tuple[list[str], dict[str, str], Path]:
        """The node worktree's config as is; ``auto_resume=false`` for a run that must never resume (TEST), which
        also never goes to a preemptible pool (L3: a preempted TEST run restarts from step 0 and pays the hour
        twice): it is launched on the first non-preemptible pool."""
        rel = self._rel(run)
        out = self.runs / rel
        auto = "false" if self.no_resume(run) else "true"
        pool = run.pool
        if self.no_resume(run):
            pools = self.ctx.cfg.pools
            mine = next((q for q in pools if q.name == pool), pools[0])
            if mine.preemptible:
                pool = next((q.name for q in pools if not q.preemptible), pool)
        argv, env = self._sbatch(self.config_rel, out, f"lp{self._tag}-{rel.replace('/', '-')}", str(run.seed), auto,
                                 pool=pool)
        if run.mitigation:
            env.update(ALLOC_ENV)
        return argv, env, out

    def submit(self, *, run: Run, commit: str) -> JobHandle:
        """The base submit; a FIRST launch of this (run, attempt) clears a leftover ``OUT.loopmeta`` (its events
        would count another job's instances into this run's budget)."""
        rel = self._rel(run)
        if not self._job_id(rel, self._receipt(rel)):
            shutil.rmtree(meta_dir(self.runs / rel), ignore_errors=True)
        return super().submit(run=run, commit=commit)

    # ------------------------------------------------------------ smoke
    def _throughput_path(self, worktree: Worktree) -> Path:
        return self.state / "smoke" / _safe(Path(worktree.path).name) / "throughput.json"

    # ------------------------------------------------------------ no-op fingerprint (intake.noop_fingerprint, 11-08)
    #: keys a CPU 1-step fingerprint cannot see (data pipeline, runtime speed, bookkeeping): a change touching one is
    #: never fingerprint-bounced (it may act through the data or the step rate)
    FP_BLIND_RX = re.compile(r"^(clip_quantiles|train_filelist|anneal_filelist|num_workers|prefetch_factor|prep_\w+|"
                             r"amp|amp_dtype|compile|cudagraphs|fused_adamw|muon_fused|gc_freeze|fast_startup|"
                             r"high_priority_stream|decoder_prune_last|flash_attention_impl|time_limit|checkpoint_\w+|"
                             r"print_freq|plot_\w+|seed|name|notes|output_dir|wandb\w*|ckpt|resume|auto_resume|"
                             r"device|distributed|debug)$")
    FP_TIMEOUT_S = 420.0               # per fingerprint (CPU; ~70 s measured for the root on 8 threads)
    FP_SCRIPT = Path(__file__).with_name("nb_fingerprint.py")

    def _fp_run(self, worktree: str, config: Path, out: Path, base: str = "") -> dict[str, Any]:
        """One fingerprint process (the project's python, isolated, no GPU); {} on timeout / failure. Cached per
        (base commit, config text): a config-only candidate runs the base's code."""
        import hashlib
        import os
        import subprocess
        cache = self.state / "fingerprint" / "cache" / (hashlib.sha1(
            f"{base}\0{config.read_text()}".encode()).hexdigest()[:20] + ".json") if base else None
        if cache is not None and cache.exists():
            try:
                return json.loads(cache.read_text())
            except ValueError:
                pass
        py = self.python or str(Path(self.repo) / ".venv" / "bin" / "python")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NB_FP_THREADS": "8"}
        try:
            subprocess.run([py, "-I", str(self.FP_SCRIPT), worktree, str(config), str(out)], cwd=worktree, env=env,
                           capture_output=True, timeout=self.FP_TIMEOUT_S, stdin=subprocess.DEVNULL)
            got = json.loads(out.read_text())
            if cache is not None and not got.get("errors"):
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(got))
            return got
        except (OSError, subprocess.SubprocessError, ValueError):
            return {}

    def noop_fingerprint(self, worktree: Worktree, work: Path) -> str:
        """'' or the ``noop_knob`` reason: a CONFIG-ONLY change whose keys a fingerprint can see, and whose parent and
        child fingerprints (nb_fingerprint: resolved LR schedule, loss / grad norm of one fixed CPU batch, the loss and
        parameter sum after one optimizer step) are identical within 1e-6. Fails open (any doubt: '')."""
        from concurrent.futures import ThreadPoolExecutor
        from loop.adapters import nb_fingerprint as FP
        if not getattr(getattr(self.ctx.cfg, "intake", None), "noop_fingerprint", False):
            return ""
        base = worktree.base_commit
        rc, out, _ = self._git(worktree.path, "status", "--porcelain")
        rc2, out2, _ = self._git(worktree.path, "diff", "--name-only", base)
        if rc or rc2:
            return ""
        changed = {ln[3:].strip() for ln in out.splitlines() if ln.strip()} | {ln.strip() for ln in out2.splitlines() if ln.strip()}
        if changed != {self.config_rel}:
            return ""                      # code changed (or nothing): not a config-only candidate
        parent_text = self._show(worktree.path, base, self.config_rel)
        try:
            child_text = (Path(worktree.path) / self.config_rel).read_text()
            pc, cc = flatten(yaml.safe_load(parent_text or "") or {}), flatten(yaml.safe_load(child_text) or {})
        except (OSError, yaml.YAMLError):
            return ""
        keys = sorted(k for k in set(pc) | set(cc) if pc.get(k, "<absent>") != cc.get(k, "<absent>"))
        if not keys or any(self.FP_BLIND_RX.match(k.rsplit(".", 1)[-1]) or self.FP_BLIND_RX.match(k) for k in keys):
            return ""
        work.mkdir(parents=True, exist_ok=True)
        (work / "parent.yaml").write_text(parent_text or "")
        (work / "child.yaml").write_text(child_text)
        with ThreadPoolExecutor(2) as ex:
            fa = ex.submit(self._fp_run, worktree.path, work / "parent.yaml", work / "parent.json", base)
            fb = ex.submit(self._fp_run, worktree.path, work / "child.yaml", work / "child.json", base)
            a, b = fa.result(), fb.result()
        same, parts = FP.same(a, b)
        (work / "verdict.json").write_text(json.dumps({"keys": keys, "same": same, "parts": parts,
                                                       "lr_final": [a.get("lr_final"), b.get("lr_final")],
                                                       "errors": [a.get("errors"), b.get("errors")]}))
        if not same:
            return ""
        return (f"noop_knob: {', '.join(keys)} changed but the training numerics did not (identical {', '.join(parts)} "
                "on a fixed CPU batch and LR schedule): the code never reads it here; change a key the code uses")

    def smoke(self, worktree: Worktree, *, memory: bool = False, mitigation: str = "",
              queue_s: float | None = None) -> SmokeResult:
        """Short training (SMOKE_OVERRIDES) through the loop's job script, then the checkpoint load / forward check
        in the same job (MODE smoke); ``sbatch`` + wait, no eval, no requeue, hub offline. ``intake.noop_fingerprint``:
        a config-only change first gets the CPU no-op fingerprint (:meth:`noop_fingerprint`); inert -> failed smoke
        ``noop_knob``, no GPU."""
        if not memory and not mitigation:
            try:
                why = self.noop_fingerprint(worktree, self.state / "fingerprint" / _safe(Path(worktree.path).name))
            except Exception:  # noqa: BLE001 - the guard fails open
                why = ""
            if why:
                return SmokeResult(False, why)
        smoke_dir = self.state / "smoke" / _safe(Path(worktree.path).name + ("-mem" if memory else "")
                                                 + (f"-{mitigation}" if mitigation else ""))
        out = smoke_dir / "out"
        shutil.rmtree(out, ignore_errors=True)        # a stale summary.json (done) / config.yaml would skip / refuse
        shutil.rmtree(meta_dir(out), ignore_errors=True)   # ... and a stale train_done event would skip the training
        smoke_dir.mkdir(parents=True, exist_ok=True)
        try:
            cfg = yaml.safe_load((Path(worktree.path) / self.config_rel).read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            return SmokeResult(False, f"cannot read {self.config_rel}: {exc}")
        cfg = self.smoke_config(cfg, memory=memory)
        if held := self._infra_hold(worktree, cfg):
            return held
        cfg_path = smoke_dir / "smoke.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        name = f"lp{self._tag}-smoke-{smoke_dir.name}"
        (Path(worktree.path) / "slurm").mkdir(exist_ok=True)
        with self._smoke_lock:
            pool = self.pool_picker() if self.pool_picker else self.ctx.cfg.pools[0].name
            self.smokes[pool] += 1
            self._smoke_names.add(name)
        jid, why, out_s, err_s = "", "", "", ""
        try:
            argv, env = self._sbatch(str(cfg_path), out, name, str(cfg.get("seed", 0)), "false", "smoke", pool=pool)
            env.pop("SBATCH_REQUEUE", None)
            env.update(OFFLINE_ENV)
            env.update(WANDB_MODE="disabled", SBATCH_NO_REQUEUE="", SBATCH_TIMELIMIT=str(self.SMOKE_MIN),
                       **(ALLOC_ENV if mitigation else {}))
            rc, out_s, err_s = self.runner.run(argv, cwd=worktree.path, timeout=300, env=env)
            if m := re.search(r"Submitted batch job (\d+)", out_s):
                jid = m.group(1)
            if rc == 0 and jid:
                rc, why = self._smoke_wait(jid, run_s=60 * self.SMOKE_MIN + 300, queue_s=queue_s)
        except BaseException:
            if jid:
                self.runner.run(["scancel", jid], timeout=60)
            raise
        finally:
            with self._smoke_lock:
                self.smokes[pool] -= 1
                self._smoke_names.discard(name)
        if rc == 124 and not jid:
            self.runner.run(["scancel", "--name", name], timeout=60)
        logs = self._tail({"worktree": worktree.path}, jid) if jid else ""
        tail = "\n".join((out_s + "\n" + err_s + "\n" + logs).strip().splitlines()[-40:])
        if rc == 124:
            self.smoke_gate.release_canary()
            return SmokeResult(False, f"smoke_unavailable: {why or f'queue timeout after {self.smoke_queue_s(queue_s):.0f}s'} on {pool}",
                               tail, unavailable=True)
        full = "\n".join(_read(p) or "" for p in self._logs({"worktree": worktree.path}, jid)) if jid else ""
        if rc != 0 and (verdict := self._infra_verdict(worktree, tail, full)):
            return verdict
        if rc != 0 and (line := next((l for l in reversed(full.splitlines())
                                      if "nanobrain_guard: FAILED" in l or "nanobrain_job: REFUSED" in l), "")):
            return SmokeResult(False, f"job refused the worktree: {line.strip()[:300]}", tail)
        if rc != 0 and LOADCHECK_FAILED in full:
            line = next(l for l in reversed(full.splitlines()) if LOADCHECK_FAILED in l)
            return SmokeResult(False, f"checkpoint load check failed: "
                                      f"{line.split(LOADCHECK_FAILED, 1)[1].strip()[:300]}", tail)
        if rc != 0:
            return SmokeResult(False, "smoke OOM" if is_oom(tail) else f"smoke exited rc={rc}", tail)
        self.smoke_gate.cleared()
        if self.nan_scan(tail):
            return SmokeResult(False, "smoke log shows NaN/inf", tail)
        if LOADCHECK_OK not in full:
            return SmokeResult(False, "checkpoint load check did not report ok", tail)
        if memory:
            return SmokeResult(True, "", tail)        # a memory smoke's batch is not the point's: no throughput
        summ = _json(out / "summary.json") or {}
        try:
            sps = float(summ["step"]) / float(summ["wall_seconds"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            sps = None
        path = self._throughput_path(worktree)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"steps_per_s": sps, "steps": summ.get("step"),
                                    "wall_seconds": summ.get("wall_seconds"), "ts": self._clock()}))
        return SmokeResult(True, "", tail)

    @staticmethod
    def nan_scan(text: str) -> bool:
        """The smoke's NaN / inf verdict on a log (:func:`nonfinite`; loop.selfcheck runs it on a known-good log)."""
        return nonfinite(text)

    def smoke_config(self, cfg: Mapping[str, Any], *, memory: bool = False) -> dict[str, Any]:
        """The point's config shrunk for the smoke (H4): :attr:`SMOKE_OVERRIDES`, and every step-counted key scaled
        by the same factor ``smoke total_steps / point total_steps`` -- ``*_steps`` (warmup / decay and any the
        candidate added), ``*_epochs`` and ``*_period`` ints, int lists under ``*_steps`` (checkpoint_steps) --
        so a schedule's shape survives (warmup + decay <= total_steps: make_lr_schedule's np.full raises on a
        negative stable phase) and an annealing candidate never crashes the smoke on a schedule the full run is
        fine with. checkpoint_period = the smoke's total_steps (one save), plot_period stays off (overrides)."""
        cfg = copy.deepcopy(dict(cfg))
        total = cfg.get("total_steps")
        smoke_total = int(self.SMOKE_OVERRIDES.get("total_steps", 300))
        f = smoke_total / float(total) if isinstance(total, (int, float)) and not isinstance(total, bool) \
            and total > 0 else 1.0
        fixed = set(self.SMOKE_OVERRIDES) | {"total_steps"}

        def scale(v: Any, floor: int) -> Any:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                return v
            return max(floor, int(round(v * f)))
        for key, val in flatten(cfg).items():
            leaf = key.rsplit(".", 1)[-1]
            if key in fixed or not leaf.endswith(("_steps", "_epochs", "_period")):
                continue
            if isinstance(val, list):
                set_dotted(cfg, key, [scale(v, 1) for v in val] if all(
                    isinstance(v, int) and not isinstance(v, bool) for v in val) else val)
            else:
                set_dotted(cfg, key, scale(val, 1 if leaf.endswith("_period") else 0))
        for key, val in self.SMOKE_OVERRIDES.items():
            if not (memory and key in self.MEMORY_KEEP):
                set_dotted(cfg, key, val)
        tot = int(cfg.get("total_steps") or smoke_total)
        w, d = int(cfg.get("warmup_steps") or 0), int(cfg.get("decay_steps") or 0)
        if w + d > tot:                                 # rounding: keep the stable phase >= 0
            cfg["decay_steps"] = max(0, tot - w) if w <= tot else 0
            cfg["warmup_steps"] = min(w, tot)
        cfg["checkpoint_period"] = tot
        return cfg

    def probe_smoke_why(self, worktree: Worktree) -> str:
        return ""                          # no probe smoke: the load check is part of every smoke

    # ------------------------------------------------------------ init
    def init_compat(self, worktree: Worktree, *, parent_commit: str) -> InitCompat:
        """No construct probe (nanobrain trains from random init; ckpt / resume are protected keys): the change's
        added lines (committed, uncommitted and untracked, vs ``parent_commit``; every file, data.py / misc.py too)
        must match no ``init_compat.deny`` regex (a pretrained brain-MRI weights source) nor :data:`NB_INIT_DENY`
        (a weight-loading idiom) -> else init_illegal."""
        deny = tuple(getattr(self.ctx.cfg.init_compat, "deny", ()) or ()) + NB_INIT_DENY
        wt = worktree.path
        rc, diff, err = self._git(wt, "diff", "-U0", parent_commit, "--", ".", NO_VENV)
        if rc != 0:
            return InitCompat(False, f"init check: cannot diff vs {parent_commit[:12]}: {err.strip()[-200:]}")
        added = [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
        _, extra, _ = self._git(wt, "ls-files", "--others", "--exclude-standard")
        for rel in extra.split():
            added += (_read(Path(wt) / rel) or "").splitlines()
        for pat in deny:
            rx = re.compile(pat, re.I)
            if hit := next((l for l in added if rx.search(l)), None):
                return InitCompat(False, f"init_illegal: {pat!r} matches {hit.strip()[:160]!r}",
                                  {"init_legal": {"ok": False, "pattern": pat, "hit": hit.strip()[:300]}})
        return InitCompat(True, "", {"init_legal": {"ok": True}})

    # ------------------------------------------------------------ launch-time refusals
    def lock_violations(self, commit: str) -> list[str]:
        """The base locks / config guard, plus (C3 / M1) every file changed from ``lock_ref`` that is binary
        (``binary:<path>``) or over :data:`MAX_FILE_BYTES` (``large:<path>``: weights / data blobs), and every
        top-level entry under ``src/`` the canonical tree does not have (``foreign:<path>``: a second package or a
        sitecustomize next to nanobrain would be importable by the train and the eval). Fail closed."""
        bad = super().lock_violations(commit)
        if (got := self._nb_ok.get(commit)) is not None:
            return bad + list(got)
        extra: list[str] = []
        rc, out, err = self._git(self.repo, "diff", "--numstat", self.lock_ref, commit, "--", ".", NO_VENV)
        if rc != 0:
            extra.append(f"<unreadable diff: {(err or out).strip()[-120:]}>")
        changed = []
        for line in out.splitlines() if rc == 0 else ():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                if parts[0] == "-" and parts[1] == "-":
                    extra.append(f"binary:{parts[2]}")
                changed.append(parts[2])
        if changed:
            rc, tree, err = self._git(self.repo, "ls-tree", "-r", "-l", commit, "--", *changed)
            for line in tree.splitlines() if rc == 0 else ():
                meta, _, path = line.partition("\t")
                size = meta.split()[-1] if meta.split() else ""
                if size.isdigit() and int(size) > MAX_FILE_BYTES:
                    extra.append(f"large:{path}")
        tops = []
        for ref in (self.lock_ref, commit):
            rc, names, _ = self._git(self.repo, "ls-tree", "--name-only", ref, "src/")
            tops.append(set(names.split()) if rc == 0 else None)
        if tops[1] is None:
            extra.append("<unreadable src/ tree>")
        elif tops[0] is not None:
            extra += [f"foreign:{n}" for n in sorted(tops[1] - tops[0])]
        self._nb_ok[commit] = extra
        return bad + extra

    @property
    def _nb_ok(self) -> dict[str, list[str]]:
        if not hasattr(self, "_nb_lock_cache"):
            self._nb_lock_cache: dict[str, list[str]] = {}
        return self._nb_lock_cache

    # ------------------------------------------------------------ job bookkeeping (OUT.loopmeta)
    def events(self, out: Path) -> list[dict[str, Any]]:
        """The job script's events for ``out`` (instances.jsonl), oldest first; unreadable lines skipped."""
        rows = []
        for line in (_read(meta_dir(out) / "instances.jsonl") or "").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and isinstance(row.get("ts"), (int, float)):
                rows.append(row)
        return rows

    def train_instances(self, out: Path) -> list[tuple[float, float | None, bool]]:
        """(start, end or None for an instance killed before its train phase returned, resumed) per train phase
        since the last FRESH start (a start without resume -- a TEST run restarted from step 0 -- resets train.py's
        own clock, so the hour starts again there)."""
        inst: list[tuple[float, float | None, bool]] = []
        for e in self.events(out):
            if e.get("event") == "train_start":
                if not e.get("resume"):
                    inst = []
                inst.append((float(e["ts"]), None, bool(e.get("resume"))))
            elif e.get("event") == "train_end" and inst and inst[-1][1] is None:
                inst[-1] = (inst[-1][0], float(e["ts"]), inst[-1][2])
        return inst

    def _resumed(self, out: Path, rec: Mapping[str, Any]) -> int:
        """H1: train phases that resumed from checkpoint-last.pth (train.py auto_resume after a preemption): the job
        script's count, so seeds / world noise see a resumed (non-exact) run."""
        n = sum(1 for e in self.events(out) if e.get("event") == "train_start" and e.get("resume"))
        return max(int(rec.get("resumed", 0)), n)

    def resume_state(self, handle: JobHandle) -> dict[str, Any]:
        """{armed, resumed, fraction, resumable} for the runner's idle churn / preempt backoff. nanobrain resumes
        IN the trainer (the job script forwards SIGTERM, train.py saves OUT/checkpoint-last.pth, a requeue with
        auto_resume=true continues from it), not through nanopath's resume config copy / latest.pt that the
        inherited version reads -- so every nanobrain run looked "not resumable" and the preempt backoff re-ran each
        run preempted 3x on idle from step 0 on primary (live 11-10: 16 screens, ~23 GPU-h a day of finished
        training thrown away; a node s2 was dropped at 3505 of its 3570 s). ``resumable``: launched by the job script
        with auto_resume=true (never a TEST run); ``armed``: that, and the checkpoint exists; ``fraction``:
        train.py's own summary.wall_seconds (written at every stop) over the wall budget."""
        try:
            rel = self._rel_of(handle)
        except ValueError:
            return {}
        rec, out = self._receipt(rel), Path(handle.output_dir)
        argv = [str(x) for x in (rec.get("argv") or [])]
        auto = len(argv) > 5 and argv[1].endswith(JOB_SCRIPT.name) and argv[5] == "true"
        summ = _json(out / "summary.json") or {}
        wall = summ.get("wall_seconds")
        frac = 1.0 if summ.get("done") is True else (
            float(wall) / (60.0 * self.budget_minutes())
            if isinstance(wall, (int, float)) and not isinstance(wall, bool) else 0.0)
        return {"armed": auto and (out / "checkpoint-last.pth").is_file(), "resumed": self._resumed(out, rec),
                "fraction": max(0.0, min(1.0, frac)), "resumable": auto}

    def _heartbeat(self, out: Path, rec: Mapping[str, Any], jid: str) -> float | None:
        """The base (logs + OUT's files) and the job script's CPU-progress heartbeat / events (M4)."""
        base = super()._heartbeat(out, rec, jid)
        paths = [meta_dir(out) / "heartbeat", meta_dir(out) / "instances.jsonl"]
        return max([p.stat().st_mtime for p in paths if p.exists()] + ([base] if base is not None else []),
                   default=None)

    def budget_minutes(self) -> float:
        caps = dict(getattr(self.ctx.spec.surface, "cap_ceilings", {}) or {})
        try:
            return float(caps.get("time_limit", 60))
        except (TypeError, ValueError):
            return 60.0

    def budget(self, out: Path) -> str:
        """'' or why the run broke the wall budget (C1): train.py's own summary.wall_seconds above the cap +
        :data:`SUMMARY_ALLOWANCE_S`, or the job script's train-phase wall summed over every instance (since the last
        fresh start; an instance killed mid-phase has no end and is not counted: its unsaved steps were lost too)
        above the cap + :data:`SETUP_ALLOWANCE_S` per instance. time_limit counts from process start, so compile /
        startup are inside the hour."""
        cap = 60.0 * self.budget_minutes()
        wall = (_json(out / "summary.json") or {}).get("wall_seconds")
        if isinstance(wall, (int, float)) and not isinstance(wall, bool) and wall > cap + SUMMARY_ALLOWANCE_S:
            return f"budget: summary.wall_seconds {wall:.0f} s > {cap:.0f} + {SUMMARY_ALLOWANCE_S:.0f} s"
        done = [(a, b) for a, b, _ in self.train_instances(out) if b is not None]
        ext = sum(b - a for a, b in done)
        if done and ext > cap + SETUP_ALLOWANCE_S * len(done):
            return (f"budget: train phases ran {ext:.0f} s over {len(done)} instance(s) (job clock) > "
                    f"{cap:.0f} + {SETUP_ALLOWANCE_S:.0f} s x {len(done)}")
        return ""

    # ------------------------------------------------------------ scores
    def _commit_of(self, out: Path) -> str:
        try:
            rel = str(out.relative_to(self.runs))
        except ValueError:
            return ""
        return str(self._receipt(rel).get("commit") or "")

    def task_results(self, out: Path, tasks: Sequence[str]) -> tuple[dict[str, float] | None, str]:
        """({task: brainmarks score}, '') re-scored from OUT/eval/<task>.json, each validated (C4): the task / name /
        model / subset fields, model_kwargs.checkpoint == OUT/checkpoint-last.pth, cwd_git.sha a prefix of the run's
        commit (the worktree that was evaluated), ``date`` and the file's mtime not before summary.json (written when
        training ended) nor before the job script's first eval_start."""
        ev = out / "eval"
        summ = out / "summary.json"
        if not summ.is_file():
            return None, "no summary.json"
        t_train = summ.stat().st_mtime
        starts = [float(e["ts"]) for e in self.events(out) if e.get("event") == "eval_start"]
        if not starts:
            return None, "no eval_start in the job's instances.jsonl (the eval did not run under the loop's job)"
        commit = self._commit_of(out)
        ckpt = f"{out}/checkpoint-last.pth"
        scores: dict[str, float] = {}
        for t in tasks:
            path = ev / f"{t}.json"
            d = _json(path)
            if not isinstance(d, dict):
                return None, f"eval incomplete: no readable eval/{t}.json"
            fields = {k: d.get(k) for k in ("task", "name", "model", "subset")}
            if fields != {"task": t, "name": EVAL_NAME, "model": EVAL_MODEL, "subset": EVAL_SUBSET}:
                return None, f"eval integrity: eval/{t}.json identifies {fields}"
            got = ((d.get("config") or {}).get("model_kwargs") or {}).get("checkpoint")
            if got != ckpt:
                return None, f"eval integrity: eval/{t}.json evaluated checkpoint {got!r}, not {ckpt}"
            sha = str((d.get("cwd_git") or {}).get("sha") or "")
            if not commit or len(sha) < 7 or not commit.startswith(sha):
                return None, f"eval integrity: eval/{t}.json cwd_git.sha {sha!r} is not the run commit {commit[:12]!r}"
            try:
                when = datetime.datetime.fromisoformat(str(d.get("date"))).timestamp()
            except (TypeError, ValueError):
                return None, f"eval integrity: eval/{t}.json date {d.get('date')!r} unreadable"
            first = min(starts)
            if when < t_train - 2 or when < first - 2 or path.stat().st_mtime < max(t_train, first) - 2:
                return None, f"eval integrity: eval/{t}.json predates the training end / the eval start"
            try:
                v = task_score(d)
            except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
                return None, f"eval integrity: eval/{t}.json cannot be re-scored: {exc!r}"[:300]
            if not math.isfinite(v):
                return None, f"eval incomplete: eval/{t}.json scores non-finite {v}"
            scores[t] = v
        return scores, ""

    def integrity(self, out: Path) -> tuple[str, str]:
        """('fabricated' | 'budget', why) or ('', '')."""
        if any(e.get("event") == "eval_fabricated" for e in self.events(out)):
            return "fabricated", ("eval_fabricated: OUT/eval existed before the loop's job started the eval "
                                  "(written by the training code); moved aside, not evaluated")
        if why := self.budget(out):
            return "budget", why
        return "", ""

    def eval_report(self, out: Path, since: float) -> tuple[tuple[dict[str, float], float] | None, str]:
        """((scores, aggregate), '') else (None, why). Scores are RE-COMPUTED from the per-task jsons
        (:meth:`task_results`); ``eval/aggregate.json`` (fresh: mtime >= ``since``) must agree with them to
        :data:`OPEN_TOL` per task and on summary.open. Scores: every family mean (the axes), ``nb_task_<task>``, the
        aggregate (= open = the plain task mean, which the weighted family composite must reproduce), and
        ``nb_steps`` / ``nb_images`` / ``nb_wall_s`` from summary.json. The wall budget and a fabricated eval dir
        refuse the scores (:meth:`integrity`)."""
        kind, why = self.integrity(out)
        if kind:
            return None, why
        path = out / "eval" / "aggregate.json"
        if not path.is_file():
            return None, "no eval/aggregate.json"
        if path.stat().st_mtime < since:
            return None, "eval/aggregate.json is older than the submission (stale)"
        raw = _json(path)
        summ = raw.get("summary") if isinstance(raw, dict) else None
        if not isinstance(summ, dict):
            return None, "eval/aggregate.json has no summary"

        def score(key: str) -> float | None:
            row = summ.get(key)
            v = row.get("score") if isinstance(row, dict) else None
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None

        fams = self.families()
        tasks = list(dict.fromkeys(t for _, _, ts in fams for t in ts))
        if missing := [t for t in tasks if score(t) is None]:
            return None, f"eval incomplete: {len(missing)}/{len(tasks)} tasks missing or non-finite: {missing}"
        agg = score(self.AGGREGATE_GROUP)
        if agg is None:
            return None, f"eval/aggregate.json has no summary.{self.AGGREGATE_GROUP}"
        per, why = self.task_results(out, tasks)
        if per is None:
            return None, why
        if off := [t for t in tasks if not abs(per[t] - (score(t) or 0.0)) <= OPEN_TOL]:
            return None, (f"eval integrity: aggregate.json disagrees with the re-scored task jsons on {off[:4]} "
                          f"(e.g. {off[0]}: {score(off[0])} vs {per[off[0]]:.8f})")
        mean = sum(per.values()) / len(per)
        if not abs(mean - agg) <= OPEN_TOL:
            return None, f"eval integrity: re-scored task mean {mean:.8f} != aggregate.json {self.AGGREGATE_GROUP} {agg:.8f}"
        agg = mean
        scores = {key: sum(per[t] for t in ts) / len(ts) for key, _, ts in fams}
        wsum = sum(w for _, w, _ in fams)
        comp = sum(w * scores[key] for key, w, _ in fams) / wsum if wsum > 0 else float("nan")
        if not abs(comp - agg) <= OPEN_TOL:
            return None, (f"weighted family composite {comp:.8f} != {self.AGGREGATE_GROUP} {agg:.8f} "
                          f"(family weights / tasks do not reproduce the leaderboard score)")
        if miss := [a for a in self.ctx.cfg.axes if a not in scores]:
            return None, f"axes {miss} are no family of loop.project.axes"
        if key := str(getattr(getattr(self.ctx.spec, "slots", None), "score_key", "") or ""):
            scores.setdefault(key, agg)                 # nb_open: the aggregate under the task's score_key too
        scores.update({f"nb_task_{t}": v for t, v in per.items()})
        run = _json(out / "summary.json") or {}
        for key, src in (("nb_steps", "step"), ("nb_images", "images"), ("nb_wall_s", "wall_seconds")):
            v = run.get(src)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                scores[key] = float(v)
        return (scores, agg), ""

    def _read_scores(self, out: Path, since: float) -> tuple[dict[str, float], float | None] | None:
        return self.eval_report(out, since)[0]

    def _train_done(self, out: Path) -> bool:
        return (_json(out / "summary.json") or {}).get("done") is True

    def _resubmit(self, out: Path, rec: dict[str, Any], jid: str, key: str, why: str) -> JobStatus | None:
        """The receipt's launch again as a new job into the SAME OUT (an eval continuation: the job script skips the
        finished training, brainmarks every finished task); ``rec[key]`` counts them. Not a 'free retry' (that is a
        preemption in the runner's telemetry)."""
        try:
            rel = str(out.relative_to(self.runs))
        except ValueError:
            return None
        if not rec.get("argv"):
            return None
        self.runner.run(["scancel", jid], timeout=60)
        _, so, _ = self.runner.run(rec["argv"], cwd=rec["worktree"], timeout=300, env=rec["env"])
        if not (m := re.search(r"Submitted batch job (\d+)", so)):
            return None
        rec.update(job_id=m.group(1), restarts=0, lost_s=0.0, **{key: int(rec.get(key, 0)) + 1})
        self._save(rel, rec)
        if rec.get("nice"):
            self.runner.run(["scontrol", "update", f"JobId={m.group(1)}", f"Nice={rec['nice']}"], timeout=60)
        return JobStatus(RunStatus.RUNNING, detail=f"{why}: eval continuation {rec[key]} as job {m.group(1)}"[:400],
                         resumed=self._resumed(out, rec))

    def _lbfgs_note(self, out: Path, rec: Mapping[str, Any], jid: str, tasks: Sequence[str], verdict: str) -> None:
        """One line per l-bfgs non-convergence (state/lbfgs.jsonl): the failure rate per task / checkpoint is
        countable (how often an eval retry or a fresh training attempt converged)."""
        try:
            self.state.mkdir(parents=True, exist_ok=True)
            with (self.state / "lbfgs.jsonl").open("a") as f:
                f.write(json.dumps({"ts": self._clock(), "run": rec.get("run_id"), "out": str(out), "job": jid,
                                    "attempt": rec.get("attempt"), "eval_retries": int(rec.get("lbfgs_evals", 0)),
                                    "tasks": list(tasks), "verdict": verdict}) + "\n")
        except OSError:
            pass

    def _tree_incomplete(self, rec: Mapping[str, Any], jid: str) -> str:
        """The job guard found no nanobrain package (``GUARD_EMPTY``) because the worktree's checkout is incomplete,
        not because of the node's code: src/nanobrain is missing on disk while the run's commit has it. Live 11-11
        a node: a restart cut its `git worktree add` off before src/ (11-10 17:26); both seeds crashed rc=3 at 02:56
        and were scored CRASH. That is INFRA (retried; the submit rebuilds the half-created tree). A commit without
        src/nanobrain is the node's own bug: '' (the base split's CRASH)."""
        wt, commit = str(rec.get("worktree") or ""), str(rec.get("commit") or "")
        if not wt or not commit or (Path(wt) / "src" / "nanobrain").is_dir():
            return ""
        if not any(GUARD_EMPTY in (_read(p) or "") for p in self._logs(rec, jid)):
            return ""
        if self._git(self.repo, "cat-file", "-e", f"{commit}:src/nanobrain")[0] != 0:
            return ""
        return (f"worktree checkout incomplete: {wt}/src/nanobrain is missing though {commit[:12]} has it "
                f"(an interrupted worktree add); the job guard found no nanobrain package")

    def _classify(self, out: Path, rec: Mapping[str, Any], jid: str, state: str, gpu_h: float, tail: str,
                  since: float) -> JobStatus | None:
        """A finished job without scores.

        Training done: a fabricated eval dir is VOID(eval, eval_fabricated); a broken wall budget VOID(too_slow,
        budget); an l-bfgs non-convergence (H3) first gets ONE eval-only retry into the same OUT (the eval is seeded
        but its GPU embedding is not bit-exact; ~5 min: only the failed task re-runs), then -- repeated on the same
        checkpoint, i.e. a property of THAT checkpoint -- ONE fresh training attempt (VOID(infra) on attempt 0: the
        core re-runs the row, same seed, a new checkpoint), then it is the verdict VOID(eval, eval_lbfgs). Least
        bias: a node is never killed by a single probe fit (survival conditioned on a fluke), and a run is retrained
        at most once (no retrain-until-it-converges); every occurrence is logged (state/lbfgs.jsonl). A TIMEOUT
        (H2) is an eval continuation into the same OUT, at most :data:`EVAL_CONTINUATIONS`, then TOO_SLOW. A
        COMPLETED job with an incomplete / inconsistent eval is VOID(eval).

        Training not done: the job stopped training on a signal (exit 75, summary.stopped_by_signal: a TERM that
        was no SLURM requeue) is a free retry into the same OUT (it resumes), INFRA once those are spent; a training
        crash falls through to the base split (CRASH / INFRA / TOO_SLOW)."""
        rec = dict(rec)
        if not self._train_done(out):
            ends = [e for e in self.events(out) if e.get("event") == "train_end"]
            if state == "FAILED" and ends and ends[-1].get("stopped") and ends[-1].get("rc") == 0:
                try:
                    rel = str(out.relative_to(self.runs))
                except ValueError:
                    rel = ""
                if rel and (st := self._free_retry(rel, rec, jid, "training stopped by a signal (exit 75)", gpu_h)):
                    return st
                return JobStatus(RunStatus.VOID, void_type=VoidType.INFRA, gpu_h=gpu_h, log_tail=tail,
                                 detail="training stopped by a signal (exit 75); free retries spent")
            if why := self._tree_incomplete(rec, jid):
                return JobStatus(RunStatus.VOID, void_type=VoidType.INFRA, gpu_h=gpu_h, log_tail=tail, detail=why)
            return None
        kind, why = self.integrity(out)
        if kind == "fabricated":
            return JobStatus(RunStatus.VOID, void_type=VoidType.EVAL, gpu_h=gpu_h, log_tail=tail, detail=why)
        if kind == "budget":
            return JobStatus(RunStatus.VOID, void_type=VoidType.TOO_SLOW, gpu_h=gpu_h, log_tail=f"{why}\n{tail}",
                             detail=why)
        full = "\n".join(_read(p) or "" for p in self._logs(rec, jid))
        if m := LBFGS.search(full):
            line = full[m.start():].splitlines()[0].strip()
            tasks = list(dict.fromkeys(FAILED_TASK.findall(full)))
            _, why = self.eval_report(out, since)
            what = f"eval_lbfgs: {line[:160]} (tasks {tasks})"
            if int(rec.get("lbfgs_evals", 0)) < 1:
                self._lbfgs_note(out, rec, jid, tasks, "eval_retry")
                if st := self._resubmit(out, rec, jid, "lbfgs_evals", what):
                    return st
            if int(rec.get("attempt", 0) or 0) < 1:
                self._lbfgs_note(out, rec, jid, tasks, "retrain")
                return JobStatus(RunStatus.VOID, void_type=VoidType.INFRA, gpu_h=gpu_h, log_tail=tail,
                                 detail=f"{what}; repeated on an eval-only retry: one fresh training attempt"[:400])
            self._lbfgs_note(out, rec, jid, tasks, "verdict")
            return JobStatus(RunStatus.VOID, void_type=VoidType.EVAL, gpu_h=gpu_h, log_tail=tail,
                             detail=f"{what}; repeated after a fresh training attempt; {why}"[:400])
        if state == "TIMEOUT":
            if int(rec.get("eval_continuations", 0)) < EVAL_CONTINUATIONS and \
                    (st := self._resubmit(out, rec, jid, "eval_continuations",
                                          "slurm TIMEOUT after training finished")):
                return st
            why = (f"TOO SLOW: the eval did not finish in {1 + int(rec.get('eval_continuations', 0))} job(s) "
                   f"after training")
            return JobStatus(RunStatus.VOID, void_type=VoidType.TOO_SLOW, gpu_h=gpu_h, log_tail=f"{why}\n{tail}",
                             detail=why)
        if state == "COMPLETED":
            _, why = self.eval_report(out, since)
            return JobStatus(RunStatus.VOID, void_type=VoidType.EVAL, gpu_h=gpu_h, log_tail=tail,
                             detail=f"eval: {why or 'no scores'}")
        return None

    # ------------------------------------------------------------ dry check
    def render(self, run: Run) -> dict[str, Any]:
        """The exact launch of ``run`` without submitting it (argv, env, out, cwd): ``--dry-run`` / operator check."""
        argv, env, out = self._launch(run)
        return {"argv": argv, "env": env, "out": str(out), "cwd": str(self.worktrees / _safe(run.node_id))}


def meta_dir(out: Path) -> Path:
    """The job script's bookkeeping dir of ``out``: ``OUT.loopmeta`` (instances.jsonl, heartbeat)."""
    return Path(str(out) + META_SUFFIX)


__all__: Sequence[str] = ("NanobrainAdapter", "JOB_SCRIPT", "LBFGS", "meta_dir", "task_score")
