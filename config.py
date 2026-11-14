"""overfitX SDK — Configuration: paths, TRACKS registry, model routing.

Every path-like setting has an environment override so the SDK is movable;
literal defaults preserve today's behavior on this machine.

Every override is OPTIONAL: with NO environment variable set the SDK runs
entirely from this repo checkout (tools from ./overfitx_tools/) against the
default machine state dir. Vars below are overrides, not requirements.

Environment overrides:
    NANOPATH_JEPA_ROOT   training repo root  (default: the nanopath-jepa checkout)
    OVERFITX_HOME        overfitX state home: MANIFEST.json + batches
                         (default: <NANOPATH_JEPA_ROOT>/overfitX). The tools no
                         longer have to live here — see OVFX_TOOLS_DIR.
    OVFX_TOOLS_DIR       deterministic-tool root holding tools/ + tools-v2/
                         (default: ./overfitx_tools in this repo if complete,
                         else the legacy <OVERFITX_HOME> state-dir copy)
    NANOPATH_VENV        training venv (default: ~/nanopath/.venv)
    OVFX_QWEN_PROXY_URL  sonnet-tier backend (default: the NO-FALLBACK proxy :12346)
    OVFX_QWEN_MODEL      served model id (default: Qwen/Qwen3.6-27B — confirmed
                         via GET /v1/models on the live proxy)
    OVFX_OPUS_URL        opus-tier backend base URL (default: https://api.anthropic.com)
    OVFX_OPUS_MODEL      opus model id (default: claude-opus-5-5)
    OVFX_OPUS_PATH       opus billing path: api_key | oauth | credit_pool | qwen
                         (default: api_key if ANTHROPIC_API_KEY is set, else oauth
                         if logged into Claude (~/.claude/.credentials.json), else
                         qwen)
    OVFX_QWEN_WAIT_MAX   max seconds to pause-and-wait for qwen (0 = unbounded,
                         the default)
    OVFX_FORCE_OPUS_ROLES  comma-list of runtime judgment roles to force onto
                         opus, e.g. "l2,analyst" ("if qwen is doing too bad use
                         opus"). Roles: analyst | l2 | l3. No-op if no opus route.
    OVFX_MODEL_<ROLE>    per-role model override, e.g. OVFX_MODEL_L2=opus (wins
                         over OVFX_FORCE_OPUS_ROLES; value opus | qwen)
    OVFX_ESCALATION_CAP_PER_CYCLE  max automatic qwen->opus escalations per
                         cycle before staying on qwen (default 20)
    OVFX_PAUSE_BACKENDS  comma-list of backends that PAUSE-AND-RETRY on a
                         transient backend-down (503 / connection-refused)
                         instead of failing or degrading to a fallback route
                         (default "qwen"). Add "opus" so the L3 editor's
                         opus-PRIMARY tool loop backs off and retries a
                         transient opus 503 rather than immediately degrading
                         to qwen. Values: any of {qwen, opus}.
    OVFX_OPUS_BACKEND_RETRIES  max bounded backoff retries for an opus backend
                         listed in OVFX_PAUSE_BACKENDS before the L3 editor
                         gives up the opus attempt and degrades to qwen
                         (default 6). Only consulted when "opus" is paused.
    OVFX_SDK_FAKE_TOOLS  "1" = select the OFFLINE task (tasks/fake.yaml), whose
                         adapter substitutes canned fixture JSON for every
                         SLURM-touching tool call (tests / dry runs). Read in
                         main.py ONLY, as a legacy spelling of --task; this
                         module no longer reads it, and neither do tool.py,
                         state.py or agents/l3.py.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# The ONE per-base MANIFEST resolver. Imported at module level (it is a leaf --
# stdlib only, no back-import of config) so there is exactly one implementation
# of "which artifacts belong to which base" in the process.
import manifest_bases
from atomicio import atomic_write_text

# THE noise model. Every sigma, floor and bar below is READ from it rather than
# written here: config.py used to carry four of the five conflicting noise
# literals in the tree (:915 NOISE_FLOOR, :940, :971, :1212) and there is now
# exactly one place those numbers live. kernel/ imports nothing above it, so
# this is a leaf import like manifest_bases above -- no back-import of config.
from kernel.model import TaskSpec as _TaskSpec
from kernel.stats import NoiseModel as _NoiseModel

# ---------------------------------------------------------------------------
# Emergency operator overrides — config.py is the SINGLE SOURCE OF TRUTH.
#
# Every tunable below is written as ``CONST = _env_override("OVFX_X", <default>)``
# with the default the plainly-visible config value. The env var is an EMERGENCY
# OPERATOR OVERRIDE only (path vars double as relocation knobs). EDIT THE DEFAULT
# HERE, not your shell: the live driver runs with NO OVFX_*/OVERFITX_*/NANOPATH_*
# env set, so a missing var always yields exactly the default on its line.
#
# _env_override(name, default) returns os.environ[name] coerced to type(default)
# when the var is SET, else `default` unchanged. bool defaults use the falsey-
# token set below (preserves the historical `== "0"` / `.lower() in (...)` off-
# switch semantics); a malformed int/float override falls back to the default
# (never raises). COMBINED_PREPROBE is DEAD from the SDK side — only stale c6
# worktrees read it; the SDK never sets or reads it.
# ---------------------------------------------------------------------------

_ENV_FALSEY = frozenset({"0", "false", "no", "off", ""})


def _env_override(name: str, default):
    """Emergency-override reader (see the block comment above). type(default)
    drives coercion: bool | int | float | Path | str."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(default, bool):          # NB: bool before int (bool<:int)
        return raw.strip().lower() not in _ENV_FALSEY
    if isinstance(default, int):
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default
    if isinstance(default, Path):
        return Path(raw)
    return raw


# ---------------------------------------------------------------------------
# Paths — env-overridable, literal defaults preserved
# ---------------------------------------------------------------------------

REPO_ROOT = _env_override(
    "NANOPATH_JEPA_ROOT",
    Path.home() / "nanopath" / "base-models" / "nanopath-jepa")
OVERFITX_ROOT = _env_override("OVERFITX_HOME", REPO_ROOT / "overfitX")
SDK_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# THE NOISE MODEL, READ FROM THIS TASK'S OWN SPEC.
#
# This used to be ``from kernel.stats import DEFAULT_NOISE``: a noise model
# ``kernel`` built from an EMPTY spec, which meant kernel/model.Noise carried
# this task's three measured sigmas as its defaults. That put one task's
# measurements in the task-independent core, and -- far worse -- it meant any
# OTHER spec that simply failed to mention ``noise:`` silently ran on this
# task's calibration. Two shipped specs were doing exactly that.
#
# The numbers have not moved; the AUTHORITY has. They live in the v1 task YAML
# that already declared every one of them, and they are read here, which is the
# layer allowed to name a task: this module owns ``DEFAULT_TASK_SPEC`` and the
# ``--task`` selection below, and ``tests/test_kernel_model.py`` already asserts
# each constant in this file against the value that YAML declares.
#
# PINNED TO THE V1 SPEC, NOT TO THE SELECTED ONE. Every constant below it
# (``NOISE_FLOOR``, ``SEED_TRANSFER_SIGMA``, ``EXPLOIT_*``, the first-trial-kill
# deltas) is a MODULE-LEVEL v1 constant whose comments cite v1 cycle evidence by
# name. Re-pointing them at whatever ``--task`` happens to select would resolve
# them at import time, before the selection is made, and would silently hand v2
# a set of thresholds derived from v1's evidence under v2's name -- the same
# substitution in the other direction. A task that wants its own thresholds
# builds a ``NoiseModel`` from its own spec; these stay v1's, and say so.
#: The default task, relative to the repo root. ONE copy of this string: the
#: noise model below and ``select_task_spec`` further down both read it, and a
#: second spelling of "which task is the default" is the kind of duplicate this
#: module exists to not have.
DEFAULT_TASK_SPEC = "tasks/fake.yaml"
#: THE v1 noise model. Other v1-era modules import THIS rather than minting a
#: second reader of the same file.
TASK_NOISE = _NoiseModel.from_spec(_TaskSpec.load(SDK_ROOT / DEFAULT_TASK_SPEC))
_NOISE = TASK_NOISE

# ---------------------------------------------------------------------------
# TOOLS_ROOT — where the deterministic tools (tools/ + tools-v2/) live.
#
# The SDK now ships a tracked mirror of the tools at SDK_ROOT/overfitx_tools/,
# so the DEFAULT is the repo copy: the whole system runs from the checkout with
# NO env var set. Resolution order (env var is an OPTIONAL override, never a
# requirement):
#   1. OVFX_TOOLS_DIR (explicit override — used verbatim, no fallback).
#   2. the repo-vendored copy SDK_ROOT/overfitx_tools IF it is COMPLETE
#      (has the critical tools below) — the authoritative single-repo default.
#   3. else the legacy state-dir copy at OVERFITX_ROOT (tools/ + tools-v2/ live
#      directly under the overfitX home). SAFE FALLBACK: if the vendored copy is
#      ever incomplete mid-transition, a live run still finds a working toolset
#      instead of erroring.
# TOOLS_ROOT holds the tools/ and tools-v2/ subdirs regardless of which copy
# won, so every downstream tool constant derives uniformly from it.
# ---------------------------------------------------------------------------
_REPO_TOOLS_ROOT = SDK_ROOT / "overfitx_tools"
# Sentinels that must ALL be present for a tools root to count as "complete".
_CRITICAL_TOOLS = ("tools-v2/state.py", "tools-v2/trial.py",
                   "tools-v2/preprobe_run.py", "tools/launch_full.py",
                   "tools/commit_full.py")


def _tools_root_complete(root: Path) -> bool:
    """True iff *root* holds every critical tool (tools/ + tools-v2/ subdirs)."""
    return all((root / rel).exists() for rel in _CRITICAL_TOOLS)


def _resolve_tools_root() -> Path:
    """Pick the tools root: OVFX_TOOLS_DIR override, else the complete repo copy,
    else the legacy state-dir copy (fallback so a partial vendored copy can never
    break a live run)."""
    override = os.environ.get("OVFX_TOOLS_DIR")
    if override:
        return Path(override)
    if _tools_root_complete(_REPO_TOOLS_ROOT):
        return _REPO_TOOLS_ROOT
    return OVERFITX_ROOT


TOOLS_ROOT = _resolve_tools_root()
TOOLS_V2 = TOOLS_ROOT / "tools-v2"
TOOLS = TOOLS_ROOT / "tools"

# Deterministic tools (shelled out to — never re-implemented)
STATE_PY = TOOLS_V2 / "state.py"
TRIAL_PY = TOOLS_V2 / "trial.py"
PREPROBE_RUN_PY = TOOLS_V2 / "preprobe_run.py"
WORKTREE_PY = TOOLS_V2 / "worktree.py"
COMMIT_FULL_PY = TOOLS / "commit_full.py"

# ---------------------------------------------------------------------------
# The nanopath ADAPTER package (plan commit 9).
#
# Two deterministic tools moved OUT of TOOLS and into adapters/nanopath/, and
# the difference is not cosmetic: TOOLS can still resolve to an EXTERNAL copy
# (OVFX_TOOLS_DIR, or the legacy state-dir fallback above), while these two are
# nanopath-task rules that must be THIS checkout's. A leakage scanner or a
# mechanism-KIND map read from a stale external mirror is the same defect
# commit_full.py already documents for labless_bundle_check: the wrong copy runs
# and "verified" means nothing.
#
# They are SDK_ROOT-anchored, never TOOLS_ROOT-anchored, for that reason. The
# constants live here so the in-process caller (agents/l3.py) and the subprocess
# caller (launch_full.py, which imports this module off REPO) resolve ONE path.
# ---------------------------------------------------------------------------
ADAPTER_NANOPATH = SDK_ROOT / "adapters" / "nanopath"
#: Was overfitx_tools/tools/smoke_assert.py.
SMOKE_ASSERT_PY = ADAPTER_NANOPATH / "smoke.py"
#: Was overfitx_tools/tools/readout_leakage_verifier.py.
LEAKAGE_VERIFIER_PY = ADAPTER_NANOPATH / "leakage.py"
MINE_RESULTS_PY = TOOLS_V2 / "mine_results.py"
# Greedy-accumulate exploit support (tools-v1 advisory scripts):
#  * exploit_parent.py picks a within-cycle positive parent to build on.
#  * state_reconcile.py --fix keeps MANIFEST.tracks.<track>.committed_since_reset
#    in sync with the commit ledger (the reset/consolidation trigger reads it).
EXPLOIT_PARENT_PY = TOOLS / "exploit_parent.py"
STATE_RECONCILE_PY = TOOLS / "state_reconcile.py"

DATA_DIR = _env_override("OVFX_DATA_DIR", Path("/data"))

# Nanopath venv for training-side Python (torch, openslide, ...)
NANOPATH_VENV = _env_override("NANOPATH_VENV", Path.home() / "nanopath" / ".venv")

# Agent definition files (~/.claude/agents/overfitx-*.md)
AGENT_DEFS_DIR = _env_override("OVFX_AGENT_DEFS", Path.home() / ".claude" / "agents")

# kgraph SQLite DB (ideator/L2 evidence). OVFX_KGRAPH_DB is an emergency path
# override (tests point it at a fixture), read at CALL time for the
# reload/patch contract in agents.briefs / agents.l2.
#
# THE DEFAULT IS THE RUNNING TASK'S OWN HOME, not the module-level
# ``OVERFITX_ROOT``. The constant this replaced (``KGRAPH_DB``) was bound at
# IMPORT time from nanopath's ``overfitX/`` for every task. MEASURED on the
# live graph (5,783 nodes / 18,972 mechanism rows): a localcv ideation brief
# came back with 98,387 chars of evidence, 226 of its 493 distinct strings
# naming pathology SSL work (``c7-preprobe-p-lit-featup-jbu-dense-seg-rea``,
# ``crops.local_crops_size``, ``stain_aug``), and not one of them measured on
# the task being planned. An evidence base borrowed from another task is worse
# than no evidence base, because it reads as measurement.
#
# ``task_home()`` is defined further down this module; it is called, never
# closed over, so the ordering is irrelevant. It is also the ONE resolver that
# already honors the ``OVERFITX_HOME`` relocation override, so pointing here at
# it neither adds nor drops an override. A task that declares no home (``fake``)
# falls back to ``OVERFITX_ROOT`` inside ``task_home`` itself, so the offline
# path is byte-identical.


def kgraph_db_path() -> Path:
    """kgraph DB path for the RUNNING task, re-reading OVFX_KGRAPH_DB at CALL
    time (fixture/reload contract). Default: ``task_home()/kgraph.sqlite``."""
    env = os.environ.get("OVFX_KGRAPH_DB")
    if env:
        return Path(env)
    try:
        return task_home() / "kgraph.sqlite"
    except Exception:  # noqa: BLE001 — an unreadable task spec must not make
        # "where is the graph" raise; reading the graph is best-effort
        # everywhere it is asked, and every caller already degrades on a
        # missing file.
        return OVERFITX_ROOT / "kgraph.sqlite"

# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

# Sonnet tier -> a locally served open model (default Qwen) behind an
# Anthropic-messages-compatible proxy. Point OVFX_QWEN_PROXY_URL at your own; a
# proxy must NOT silently fall back to a paid route when the model is down.
QWEN_PROXY_URL = _env_override("OVFX_QWEN_PROXY_URL", "http://127.0.0.1:12346")
QWEN_MODEL = _env_override("OVFX_QWEN_MODEL", "Qwen/Qwen3.6-27B")

# Pause-and-wait bounds for qwen preemption (0 = unbounded)
QWEN_WAIT_MAX = _env_override("OVFX_QWEN_WAIT_MAX", 0)
QWEN_POLL_INTERVAL = _env_override("OVFX_QWEN_POLL_INTERVAL", 30)
# Wall-clock bound on the POST serving probe (llm.tiny_completion_ok). The probe
# runs per dispatch wave and per backend-hold tick, so it must be cheap AND it
# must never inherit the 600s production client timeout — a hung POST would
# otherwise block the probe, and with it the whole hold loop, indefinitely.
QWEN_PROBE_TIMEOUT = _env_override("OVFX_QWEN_PROBE_TIMEOUT", 20.0)

# Opus tier — the 2 ideators only
OPUS_MODEL = _env_override("OVFX_OPUS_MODEL", "claude-opus-5-5")
OPUS_URL = _env_override("OVFX_OPUS_URL", "https://api.anthropic.com")

# Hard wall-clock timeout (seconds) on a SINGLE opus call via the oauth
# claude_agent_sdk subprocess. Load-bearing: the bundled Claude Code CLI can
# spawn then hang/die silently (observed 7-15 min freezes with no directions
# emitted), and anyio.run() over its async query would otherwise block the whole
# run FOREVER. On expiry the call raises OpusTimeoutError so the caller can retry
# once / degrade gracefully instead of hanging. 0 disables the wall (unbounded).
OPUS_TIMEOUT = _env_override("OVFX_OPUS_TIMEOUT", 600.0)

# Opus billing path: api_key | oauth | credit_pool | qwen.
# oauth drives real opus through claude_agent_sdk on the logged-in Max/Pro
# subscription (no API key); it becomes the default when logged into Claude
# (~/.claude/.credentials.json present) and no ANTHROPIC_API_KEY is set.
_has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
_has_oauth = (Path.home() / ".claude" / ".credentials.json").exists()
_default_opus_path = ("api_key" if _has_key else "oauth" if _has_oauth else "qwen")
OPUS_PATH = _env_override("OVFX_OPUS_PATH", _default_opus_path)

# NO FAKE_TOOLS HERE. It used to be
# ``FAKE_TOOLS = os.environ.get("OVFX_SDK_FAKE_TOOLS", "") == "1"``, evaluated
# on this line at IMPORT time and re-bound by value in ``tool.py``,
# ``state.py`` and ``agents/l3.py``. That single line is why task selection
# could not be a parameter: three modules had to be told the answer before
# anything imported them, so ``tests/conftest.py`` set the variable on line 13,
# ahead of every SDK import. Selection now happens once, in
# ``main.resolve_adapter``, off the ``adapter:`` string in the task spec, and
# the resolved adapter is passed DOWN. See the note in ``tool.py``.
#
# ``FIXTURES_DIR`` went with it, to ``adapters/fake``. The canned-JSON directory
# is the OFFLINE adapter's own business and nothing else in the tree read it;
# leaving it declared here would be a constant this module resolves for a task
# it is not allowed to know about.

# ---------------------------------------------------------------------------
# Per-role model tiers — explicit, no string-matching on role names.
# The LOW-volume, HIGH-judgment roles run on opus by default: the ideators
# (what-to-try), the L2 lineage-owner (what-to-try-next: derive_spec /
# should_stop / render_verdict) and the analyst (survivor selection). These are
# the cheap, high-value opus spend + the plateau-hypothesis test. The L3 editor
# stays on qwen (high-volume mechanical tool loop — opus for it is the api_key
# bare-client path only, wired separately). The user can force any runtime
# judgment role onto a different tier with OVFX_FORCE_OPUS_ROLES /
# OVFX_MODEL_<ROLE>, and OVFX_L2_OPUS=0 is a kill-switch that drops L2+analyst
# back to qwen. Because the OAuth opus subprocess is flaky, the L2/analyst
# structured-JSON calls degrade to qwen (rather than stranding a lineage) when
# opus is unavailable or a call fails/times out — see llm.query_json.
# ---------------------------------------------------------------------------

ROLE_TIERS: dict[str, str] = {
    "ideator": "opus",
    "analyst": "opus",
    "l2": "opus",
    "l3": "qwen",
}

# Roles whose default opus routing can be dropped back to qwen with the single
# OVFX_L2_OPUS=0 kill-switch (the pair moved onto opus together in this change).
L2_OPUS_KILL_SWITCH_ROLES: frozenset[str] = frozenset({"l2", "analyst"})


def _role_model_override(role: str) -> str | None:
    """Manual per-role model override from the environment (read at call time).

    OVFX_MODEL_<ROLE> (e.g. OVFX_MODEL_L2=opus) wins; otherwise a role listed in
    OVFX_FORCE_OPUS_ROLES (comma-separated) is forced to opus. Returns
    'opus' | 'qwen' | None (None => use the ROLE_TIERS default).
    """
    per_role = os.environ.get(f"OVFX_MODEL_{role.upper()}")
    if per_role and per_role.strip().lower() in ("opus", "qwen"):
        return per_role.strip().lower()
    forced = os.environ.get("OVFX_FORCE_OPUS_ROLES", "")
    if role.lower() in {r.strip().lower() for r in forced.split(",") if r.strip()}:
        return "opus"
    return None


def tier_for_role(role: str) -> str:
    """Effective model tier for a role, honoring manual opus overrides.

    `role` must be a key of ROLE_TIERS. OVFX_MODEL_<ROLE> / OVFX_FORCE_OPUS_ROLES
    can override the default tier (whole-role routing; distinct from the
    automatic single-call escalation in llm.py).
    """
    if role not in ROLE_TIERS:
        raise KeyError(f"Unknown role {role!r}; expected one of {sorted(ROLE_TIERS)}")
    override = _role_model_override(role)
    if override:
        return override
    # OVFX_L2_OPUS=0 kill-switch: force the L2/analyst judgment roles back to
    # qwen without needing a per-role override (an explicit OVFX_MODEL_<ROLE>
    # still wins above). Default (unset / "1") keeps the opus routing.
    if (role in L2_OPUS_KILL_SWITCH_ROLES
            and os.environ.get("OVFX_L2_OPUS", "1").strip() == "0"):
        return "qwen"
    return ROLE_TIERS[role]


def escalation_cap() -> int:
    """Max automatic qwen->opus escalations per cycle (env, default 20)."""
    return _env_override("OVFX_ESCALATION_CAP_PER_CYCLE", 20)


def pause_backends() -> frozenset[str]:
    """Backends that PAUSE-AND-RETRY on a transient backend-down instead of
    failing / degrading to a fallback route (from OVFX_PAUSE_BACKENDS).

    Default "qwen": the no-fallback qwen route already pauses-and-waits on 503
    (that rule is unconditional). Adding "opus" makes the L3 editor's
    opus-PRIMARY tool loop back off + retry a transient opus 503 rather than
    immediately degrading to qwen (issue: don't degrade a preferred backend on
    a transient outage). Values: any of {qwen, opus}; unknown tokens ignored.
    """
    raw = os.environ.get("OVFX_PAUSE_BACKENDS", "qwen")
    return frozenset(b.strip().lower() for b in raw.split(",") if b.strip())


def opus_backend_retry_max() -> int:
    """Bounded opus backend-down backoff retries before the L3 editor gives up
    the opus attempt and degrades to qwen (env, default 6). Only consulted when
    "opus" is in pause_backends()."""
    return _env_override("OVFX_OPUS_BACKEND_RETRIES", 6)


# ---------------------------------------------------------------------------
# EXTERNAL LAUNCH GATE — the ceiling every internal cap must stay under.
#
# `launch_full.py` blocks on `safe_launch_gate.sh wait 24` before each sbatch,
# and `job_cap.sh` enforces CAP=24 on concurrent overfitX GPU jobs
# (main_1gpu + pp_eval) globally. That 24 is the EXTERNAL gate.
#
# THE DEADLOCK (observed 2026-07-12): if a track's INTERNAL semaphore is larger
# than the external gate, surplus worker threads acquire an internal full-slot
# and THEN block at the external gate while still holding it. One stalled GPU
# job then cascades into a full futex deadlock of the orchestrator. Staying at
# or BELOW the gate is the safe direction — surplus lineages simply block on
# the internal semaphore, which releases normally.
#
# `_validate_track_caps()` below turns that from a comment into a hard import-
# time failure, and tests/test_tracks.py pins this constant to the literals in
# job_cap.sh + launch_full.py so a script edit cannot silently desync it.
# ---------------------------------------------------------------------------
EXTERNAL_LAUNCH_GATE = 24

# Adaptive train-track full-slot caps. The operator's rule: 8 concurrent
# `main_1gpu` trainings when the cluster is free, 5 when it is contended.
# "Free" is decided by cluster_load.gpu_headroom() — see that module for why
# idle-GPU headroom (and not queue depth or idle CPUs) is the signal.
TRAIN_FULLS_FREE = 8
TRAIN_FULLS_CONTENDED = 5

# A cluster is "free" only if it could absorb the whole free-branch cap right
# now. Tying the threshold to TRAIN_FULLS_FREE keeps the two self-consistent:
# we never claim "free" while unable to actually place 8 jobs.
TRAIN_FREE_GPU_HEADROOM = TRAIN_FULLS_FREE


# ---------------------------------------------------------------------------
# TRACKS registry — the single source of truth for per-track BEHAVIOR, and for
# THE SHIPPED TASK'S names only. It is NOT the CLI vocabulary: what
# tools-v2/state.py accepts for --track is `valid_tracks()` below, read from the
# running task's spec. It used to be `lib/common.VALID_TRACKS`, a literal copy of
# these two names in the shared tool library, and a second task's `--track main`
# died on "invalid choice" before any of its code ran.
# The agent-def files are named with the -pretrain/-preprobe suffix;
# `agent_suffix` maps to them.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackConfig:
    name: str                  # canonical name accepted by state.py --track
    agent_suffix: str          # suffix used in ~/.claude/agents/overfitx-*-<suffix>.md
    budget: int                # trial budget per cycle
    lineage_concurrency: int   # concurrent L2 lineages
    max_concurrent_fulls: int  # concurrent full trainings / evals in flight
                               # (the FREE-cluster value; also the hard ceiling)
    runner: str                # "trial" (trial.py run) | "preprobe" (preprobe_run.py)
    worktree_mode: str         # "per_trial" (worktree.py setup) | "shared" (consolidator-provisioned)
    max_fix_attempts: int      # L3 fix-and-rerun attempts (0 = never retry)
    # run_cycle dispatch shape (L1 agent contract): "waves" = foreground
    # batches up to lineage_concurrency (pretrain L1: "dispatch all 24 ...
    # in one message"); "sequential" = strictly ONE lineage in flight at a
    # time (preprobe L1: "exactly ONE L2 in flight at any time"). Registry-
    # driven so no code branches on raw track literals.
    dispatch_mode: str = "waves"
    # CONTENDED-cluster full-slot cap. None = this track is NOT adaptive and
    # always uses max_concurrent_fulls (preprobe: a flat 3). When set, it must
    # be <= max_concurrent_fulls; resolve_max_concurrent_fulls() picks between
    # the two from the live cluster reading and falls back to THIS (the
    # conservative) value whenever the cluster state cannot be determined.
    min_concurrent_fulls: int | None = None
    # RETIREMENT. Empty string = the track is LIVE. A non-empty string is the
    # stated reason the track no longer runs, and it is the message every
    # refusal prints. A retired track is deliberately KEPT in the registry --
    # deleting the entry would make ``get_track(<name>)`` raise at ~40 call
    # sites, un-index every historical lineage row (``state``'s run-name track
    # index iterates TRACKS), and break the runner-scoped predicates that
    # commit ace9810 deliberately keyed off ``get_track(track).runner`` rather
    # than a raw track literal. Retirement is therefore a FIELD, not a
    # deletion: history stays readable, dispatch refuses. See
    # :func:`assert_track_live`.
    retired: str = ""

    @property
    def is_retired(self) -> bool:
        """True iff this track no longer dispatches trials."""
        return bool(self.retired)

    @property
    def adaptive_fulls(self) -> bool:
        """True iff this track's full-slot cap varies with cluster load."""
        return self.min_concurrent_fulls is not None

    @property
    def ideator_agent_def(self) -> str:
        return f"overfitx-ideator-{self.agent_suffix}.md"

    @property
    def l2_agent_def(self) -> str:
        return f"overfitx-l2-{self.agent_suffix}.md"

    @property
    def l3_agent_def(self) -> str:
        return f"overfitx-l3-editor-{self.agent_suffix}.md"


# ---------------------------------------------------------------------------
# PREPROBE RETIREMENT (operator decision, 2026-08-07).
# ---------------------------------------------------------------------------
# This string is the single source of the reason. It is printed by every
# refusal, so anything that tries to dispatch a preprobe trial is told WHY in
# the numbers that decided it -- not just "disabled".
PREPROBE_RETIREMENT = (
    "the preprobe track is RETIRED (operator decision, 2026-08-07): the "
    "readout is now TRAINED inside the train cycle instead of searched "
    "post-hoc. MEASURED reason - re-minting the panel on a BYTE-IDENTICAL "
    "frozen checkpoint moves panel_mean_overall by pooled sd ~0.0053, so a "
    "single-mint vs single-mint comparison carries SE ~0.0075. An observed "
    "+0.0100 is therefore 1.33 sigma (p~0.18), i.e. not a result. Resolving "
    "the 0.01 graduation bar at 80% power needs n=5 mints per arm (~19h per "
    "candidate) = ~4 candidates per CYCLE. c9 scored ~10 preprobe trials with "
    "ZERO positives, and its two 'best' trials were ablation CONTROLS that "
    "recovered baseline exactly. The track is arithmetically incapable of the "
    "job it was given, so it is retired rather than left to burn budget. "
    "NOTE what this does NOT retire: the PANEL. panel.json, the rolling "
    "3-model panel, panel_mean_overall, baseline_readout_panel.json, "
    "mint_baseline.py and panel_delta.py are how TRAIN trials are scored and "
    "are untouched. The registry entry is KEPT so historical preprobe rows "
    "stay readable and every runner-scoped predicate still resolves."
)


class TrackRetiredError(RuntimeError):
    """A retired track was asked to dispatch work.

    This is a REFUSAL, never a no-op. A retirement that silently returned
    "nothing to do" would be indistinguishable from a track that ran and found
    nothing -- which is precisely the confusion the measured numbers in
    :data:`PREPROBE_RETIREMENT` exist to end.
    """


def assert_track_live(track: str, *, action: str = "dispatch") -> "TrackConfig":
    """Return the TrackConfig for ``track``, or REFUSE if it is retired.

    Call this at every point that would START work for a track. Reading
    historical rows for a retired track is always allowed and must NOT call
    this -- retirement stops dispatch, not history.
    """
    tcfg = get_track(track)
    if tcfg.is_retired:
        raise TrackRetiredError(
            f"REFUSED: {action} requested for track {track!r} - {tcfg.retired}")
    return tcfg


def live_tracks() -> tuple[str, ...]:
    """Canonical names of the tracks that still dispatch trials."""
    return tuple(n for n, t in TRACKS.items() if not t.is_retired)


def retired_tracks() -> tuple[str, ...]:
    """Canonical names of the tracks that are retired but still readable."""
    return tuple(n for n, t in TRACKS.items() if t.is_retired)


def retirement_exclusion_note(names: Iterable[str] | None) -> str:
    """One line a planner logs when it dropped retired tracks from a plan.

    Empty when nothing was dropped, so a caller can append it unconditionally
    -- the same discipline as ``anchor_attribution.exclusion_note`` and
    ``base_join.exclusion_note``. A plan that quietly shrank would repeat, in
    miniature, the failure that whole register exists to prevent.
    """
    dropped = [n for n in (names or ()) if n in TRACKS and TRACKS[n].is_retired]
    if not dropped:
        return ""
    return (f"[{len(dropped)} track(s) EXCLUDED from this plan as RETIRED: "
            f"{', '.join(sorted(dropped))}. Their historical trials are real, "
            f"are NOT deleted, and stay readable - the registry entry is kept "
            f"on purpose. Reason: "
            + " | ".join(TRACKS[n].retired for n in sorted(dropped)) + "]")


class TrackBudgetDivergence(RuntimeError):
    """A persisted trial_budget disagrees with TrackConfig.budget.

    Callers must treat this as a REFUSAL, never as a value to pick between."""


def track_trial_budget(track: str, recorded: object = None) -> int:
    """THE trial budget for a track. ``TrackConfig.budget`` is authoritative.

    WHY THIS EXISTS. The budget was written down twice: ``TrackConfig.budget``
    (60 for train), which ``run_rounds`` actually enforces, and
    ``MANIFEST.tracks.train.trial_budget`` (120), which only the dashboard and
    ``state.py allocate`` ever read. c9 therefore reported a 62/120 run that was
    in fact bound at 60 — the operator was reading a plan the driver was not
    executing. Two definitions of one rule is this codebase's recurring bug
    class, and editing the 120 to a 60 would only reset the drift clock.

    So: the code constant is the ONE definition and the persisted field is a
    DERIVED MIRROR. Pass whatever a document recorded as ``recorded``; a value
    that disagrees raises :class:`TrackBudgetDivergence` naming both numbers,
    because a document that disagrees with the enforcer is exactly the condition
    nobody was noticing. ``None`` / non-int (nothing recorded) is not a
    disagreement — it simply resolves to the authoritative value.

    The mirror is kept in step at the one place that owns cycle bookkeeping:
    ``rollover.update_manifest_live_batch`` re-derives it every cycle.
    """
    tcfg = TRACKS.get(track)
    if tcfg is None:
        raise TrackBudgetDivergence(
            f"track {track!r} is not in the registry {sorted(TRACKS)} — there "
            "is no authoritative budget to validate against")
    if isinstance(recorded, bool) or not isinstance(recorded, int):
        return tcfg.budget
    if recorded != tcfg.budget:
        raise TrackBudgetDivergence(
            f"recorded trial_budget {recorded} for track {track!r} disagrees "
            f"with the authoritative config.TRACKS[{track!r}].budget "
            f"{tcfg.budget}. The orchestrator enforces {tcfg.budget}; a "
            f"document saying {recorded} makes every progress readout a plan "
            "nobody is executing. Repair the document (the rollover re-derives "
            "it each cycle) rather than the constant.")
    return tcfg.budget


TRACKS: dict[str, TrackConfig] = {
    "train": TrackConfig(
        name="train",
        agent_suffix="pretrain",
        budget=60,
        # ADAPTIVE FULL-SLOT CAP: 8 concurrent `main_1gpu` trainings when the
        # cluster is free, 5 when it is contended (operator rule, 2026-08-02 —
        # 24 in flight on a busy cluster just queues and starves everyone).
        # resolve_max_concurrent_fulls() picks between them from live GPU
        # headroom and uses the CONSERVATIVE 5 whenever the cluster cannot be
        # read (unknown is never treated as free).
        #
        # BOTH values stay far UNDER the external gate (EXTERNAL_LAUNCH_GATE=24:
        # job_cap.sh CAP + `safe_launch_gate.sh wait 24` in launch_full), which
        # is the SAFE direction — see the EXTERNAL_LAUNCH_GATE block above for
        # the 2026-07-12 futex deadlock this ordering prevents.
        # _validate_track_caps() enforces it at import.
        #
        # lineage_concurrency stays higher than the cap so a lineage is always
        # ready to take a freed slot; surplus lineages block on the INTERNAL
        # semaphore (safe) rather than the external gate.
        lineage_concurrency=26,
        max_concurrent_fulls=TRAIN_FULLS_FREE,
        min_concurrent_fulls=TRAIN_FULLS_CONTENDED,
        runner="trial",
        worktree_mode="per_trial",
        max_fix_attempts=3,
    ),
    # RETIRED 2026-08-07 -- see PREPROBE_RETIREMENT for the measured reason.
    # The entry is KEPT, not deleted: every historical preprobe lineage, ledger
    # row, journal event and kgraph node stays attributable to a track that
    # exists, and ``get_track("preprobe").runner`` still resolves for the
    # runner-scoped predicates (agents/ideator.py, spec_guard, orchestrator).
    # budget=0 is the HONEST mirror of "this track dispatches nothing"; the
    # REFUSAL, not the zero, is what stops a dispatch (assert_track_live).
    "preprobe": TrackConfig(
        name="preprobe",
        agent_suffix="preprobe",
        budget=0,
        retired=PREPROBE_RETIREMENT,
        lineage_concurrency=3,
        max_concurrent_fulls=3,
        runner="preprobe",
        worktree_mode="shared",
        # L1 preprobe contract: "strictly sequential — exactly ONE L2 in
        # flight at any time" (run_cycle honors this; max_concurrent_fulls
        # above still caps the legacy run_track wave path).
        dispatch_mode="sequential",
        # PREPROBE FORGIVENESS (was 0): the readout is a single model.py edit and
        # the legal-label set is now surfaced to the editor's prompt BEFORE
        # launch, so a bounded fix-loop lets the editor correct a gated-off /
        # mislabeled readout instead of bleeding a void per attempt (the c6
        # gate_blocked starvation). Still small — the readout edit is cheap.
        max_fix_attempts=2,
    ),
}


# ---------------------------------------------------------------------------
# STRUCTURAL invariant: internal cap <= EXTERNAL_LAUNCH_GATE.
# ---------------------------------------------------------------------------
class TrackCapInvariantError(ValueError):
    """A track's internal full-slot cap violates the external-gate invariant."""


def _validate_track_caps(tracks: dict[str, TrackConfig] | None = None) -> None:
    """Fail LOUDLY (at import) if any track would out-issue the external gate.

    This is the guard the 2026-07-12 futex deadlock earned. A future edit that
    raises max_concurrent_fulls above EXTERNAL_LAUNCH_GATE — or sets a
    contended cap above the free cap, or a non-positive cap — now raises here
    instead of silently reproducing a whole-orchestrator deadlock in
    production.
    """
    for name, t in (tracks if tracks is not None else TRACKS).items():
        if t.max_concurrent_fulls < 1:
            raise TrackCapInvariantError(
                f"track {name!r}: max_concurrent_fulls="
                f"{t.max_concurrent_fulls} must be >= 1")
        if t.max_concurrent_fulls > EXTERNAL_LAUNCH_GATE:
            raise TrackCapInvariantError(
                f"track {name!r}: max_concurrent_fulls="
                f"{t.max_concurrent_fulls} EXCEEDS the external launch gate "
                f"({EXTERNAL_LAUNCH_GATE}). Surplus worker threads would take "
                f"an internal full-slot and then block AT safe_launch_gate.sh "
                f"while holding it — one stalled GPU job then futex-deadlocks "
                f"the whole orchestrator (observed 2026-07-12). Lower the "
                f"internal cap; do NOT raise it to match, and do NOT lower the "
                f"external gate to match.")
        if t.min_concurrent_fulls is not None:
            if t.min_concurrent_fulls < 1:
                raise TrackCapInvariantError(
                    f"track {name!r}: min_concurrent_fulls="
                    f"{t.min_concurrent_fulls} must be >= 1")
            if t.min_concurrent_fulls > t.max_concurrent_fulls:
                raise TrackCapInvariantError(
                    f"track {name!r}: min_concurrent_fulls="
                    f"{t.min_concurrent_fulls} exceeds max_concurrent_fulls="
                    f"{t.max_concurrent_fulls}; the contended cap must be the "
                    f"SMALLER of the two")


_validate_track_caps()


def resolve_max_concurrent_fulls(tcfg: TrackConfig, *,
                                 logger=None) -> tuple[int, dict]:
    """Resolve THIS track's full-slot cap for one semaphore construction.

    Returns ``(cap, info)``. ``info`` always carries a ``reason`` string and a
    ``source`` in {"static", "free", "contended", "fallback_unknown",
    "env_override"} so callers can log WHY, and tests can assert it.

    ZERO ENV VARS REQUIRED. ``OVFX_MAX_FULLS_<TRACK>`` exists as an emergency
    operator escape hatch only; the default is correct unaided. The override is
    still bounded by the external-gate invariant.

    UNKNOWN IS NOT FREE: any probe failure resolves to min_concurrent_fulls
    (the conservative branch) and is reported with source="fallback_unknown"
    and a logged WARNING — never silently.

    SAMPLED ONCE, AT SEMAPHORE CONSTRUCTION — deliberately. A
    threading.Semaphore cannot be shrunk after permits are issued: to go 8 -> 5
    you would have to acquire 3 permits, which blocks behind in-flight trials
    (and deadlocks outright if done on the dispatch thread), and to go 5 -> 8
    you release permits, which a plain Semaphore accepts without bound — a
    flapping signal would then over-issue past the external gate, i.e. exactly
    the 2026-07-12 failure. Re-sampling would also be self-referential: our own
    launches consume the headroom that decides the cap. The external gate
    (safe_launch_gate.sh / job_cap.sh) remains the real-time backstop, so the
    internal cap only has to be a correct coarse throttle, and a stable one is
    strictly safer than a responsive one.
    """
    # Read every field defensively: run_rounds is also driven by lightweight
    # stub track objects in tests, and a cap resolution must never be the thing
    # that breaks dispatch.
    track_name = str(getattr(tcfg, "name", "") or "track")
    static_cap = int(getattr(tcfg, "max_concurrent_fulls", 1) or 1)
    contended_cap = getattr(tcfg, "min_concurrent_fulls", None)

    env_key = f"OVFX_MAX_FULLS_{track_name.upper()}"
    raw = (os.environ.get(env_key) or "").strip()
    if raw:
        try:
            val = int(raw)
        except ValueError:
            val = -1
        if val < 1 or val > EXTERNAL_LAUNCH_GATE:
            msg = (f"{env_key}={raw!r} is not an integer in "
                   f"[1, {EXTERNAL_LAUNCH_GATE}]; ignoring the override")
            if logger:
                logger.error("full-slots[%s]: %s", track_name, msg)
        else:
            info = {"source": "env_override", "cap": val,
                    "reason": f"{env_key}={val}"}
            if logger:
                logger.warning("full-slots[%s]: cap=%d via %s (operator "
                               "override; the default needs no env var)",
                               track_name, val, env_key)
            return val, info

    if contended_cap is None:
        # Not cluster-adaptive: a flat cap, and NO cluster probe is paid for.
        info = {"source": "static", "cap": static_cap,
                "reason": f"track {track_name!r} has a flat cap"}
        if logger:
            logger.info("full-slots[%s]: cap=%d (flat, not cluster-adaptive)",
                        track_name, static_cap)
        return static_cap, info

    conservative = int(contended_cap)
    try:
        import cluster_load
        probe = cluster_load.gpu_headroom(TRAIN_FREE_GPU_HEADROOM)
    except Exception as e:  # noqa: BLE001 — a probe must never break dispatch
        probe = {"status": "unknown",
                 "reason": f"cluster probe raised: {e!r}"}

    status = probe.get("status")
    if status == "free":
        cap = static_cap
        if logger:
            logger.info("full-slots[%s]: cap=%d — cluster FREE (%s)",
                        track_name, cap, probe.get("reason"))
        return cap, {"source": "free", "cap": cap,
                     "reason": probe.get("reason", ""), "probe": probe}
    if status == "contended":
        if logger:
            logger.info("full-slots[%s]: cap=%d — cluster CONTENDED (%s)",
                        track_name, conservative, probe.get("reason"))
        return conservative, {"source": "contended", "cap": conservative,
                              "reason": probe.get("reason", ""),
                              "probe": probe}

    # status is "unknown" (or anything unrecognised): NOT free. Say so.
    reason = probe.get("reason") or "cluster state undetermined"
    if logger:
        logger.warning(
            "full-slots[%s]: cap=%d — cluster state UNKNOWN, using the "
            "CONSERVATIVE fallback (this is a FALLBACK, not a reading of a "
            "free cluster): %s", track_name, conservative, reason)
    return conservative, {"source": "fallback_unknown", "cap": conservative,
                          "reason": reason, "probe": probe}


# Dispatch order. This historically defaulted to "preprobe,train" (preprobe
# FIRST, spec §3); with preprobe RETIRED the default is train-only.
#
# OVFX_TRACK_ORDER still overrides the REQUEST, but it cannot resurrect a
# retired track: retired names are filtered out of TRACK_ORDER and recorded in
# TRACK_ORDER_RETIRED, and TRACK_ORDER_EXCLUSION_NOTE says so in words, so the
# plan shrinking is never SILENT. An explicit single-track request for a
# retired track does NOT fall through to an empty plan -- the CLI calls
# assert_track_live and REFUSES (see run_cycle._cycle_tracks).
#
# Unknown names are left in place on purpose so get_track() still raises the
# familiar "Unknown track" error rather than being quietly dropped here.
TRACK_ORDER_REQUESTED = tuple(t.strip() for t in os.environ.get(
    "OVFX_TRACK_ORDER", "train").split(",") if t.strip())
TRACK_ORDER = tuple(t for t in TRACK_ORDER_REQUESTED
                    if not (t in TRACKS and TRACKS[t].is_retired))
TRACK_ORDER_RETIRED = tuple(t for t in TRACK_ORDER_REQUESTED
                            if t in TRACKS and TRACKS[t].is_retired)
#: "" when nothing was dropped; otherwise the line a planner must log.
TRACK_ORDER_EXCLUSION_NOTE = retirement_exclusion_note(TRACK_ORDER_RETIRED)


def get_track(name: str) -> TrackConfig:
    """The RUNNING TASK's track by canonical name; raise on anything else.

    ``TRACKS`` stays authoritative for the names it owns -- it carries the
    operational fields (``runner``, ``worktree_mode``, ``agent_suffix``) that
    no spec declares, so the shipped task resolves through the identical
    entries it always has. Any OTHER task's declared track is built from that
    task's own spec instead of raising: a second task planned its cycle
    correctly and then died here one call later, because this lookup could
    only ever answer for one task (``docs/rebuild_status.md`` §9.3 row 3).
    """
    if name in TRACKS:
        return TRACKS[name]
    spec = load_task_spec()
    declared = spec.tracks.get(name)
    if declared is None:
        raise KeyError(f"Unknown track {name!r}; task {spec.name!r} declares "
                       f"{sorted(spec.tracks)}")
    return TrackConfig(
        name=name, agent_suffix=name,
        budget=spec.budget.trials_per_cycle.get(name, 0),
        lineage_concurrency=declared.lineage_concurrency,
        max_concurrent_fulls=spec.budget.full_slots.cap,
        # The shipped task's two runners are ``trial.py`` and
        # ``preprobe_run.py``; a foreign task's work happens inside its
        # ADAPTER, so neither name is true here and every runner-scoped
        # branch must miss rather than guess one of them.
        runner="", worktree_mode="",
        max_fix_attempts=declared.max_fix_attempts,
        dispatch_mode=declared.dispatch, retired=declared.retired)


def valid_tracks() -> tuple[str, ...]:
    """THE track vocabulary: every track name the ACTIVE TASK declares.

    ``TRACKS`` is deliberately NOT consulted. It carries the shipped task's
    operational fields and answers ``get_track`` for the names it owns, but as
    a VOCABULARY it is one task's, and a second task's cycle died on exactly
    that: ``state.py track init --track main`` was refused with "invalid
    choice: 'main' (choose from train, preprobe)" before any code ran, because
    the CLI's ``choices=`` came from a frozenset literal in the shared tool lib
    (``tools-v2/lib/common.VALID_TRACKS``) rather than from the running task.
    Deriving from ``TRACKS`` instead would keep the same hole open in the other
    direction -- ``--track train`` would be ACCEPTED under a task that declares
    no such track, which is worse than the refusal, because a typo'd or foreign
    track name would then silently open a state file nothing else reads.

    DECLARATION ORDER, not sorted. The order is load-bearing twice: it is what
    a refusal message lists ("choose from train, preprobe" — byte-identical to
    what the deleted literal's set produced for the shipped task), and
    ``[0]`` is the fallback default for the two places that had one
    (``state.py allocate --track``, ``injected_ideas._guess_track``). Sorting
    silently moved both to ``preprobe`` for nanopath, which is a behaviour
    change wearing a tidiness costume. The YAML mapping's order is just as
    stable as sorted order, and it is the order the task's author chose.
    """
    return tuple(load_task_spec().tracks)


def score_key() -> str:
    """THE key a run's ``summary.json`` writes its panel aggregate under.

    Sibling of ``valid_tracks``/``track_trial_budget``, and here for the same
    reason: it is a fact the ACTIVE TASK declares (``slots.score_key``), asked
    of ``config`` so that the library half and every tool subprocess get ONE
    answer. ``kernel/model.Slots.score_key`` has declared it since the panel
    was modelled and ``tasks/*.yaml`` have spelled it since they were written;
    only ``kernel/ingest`` and ``lib/result_table`` ever read it, so the other
    twenty-odd readers went on spelling ``final_probe_score`` themselves. That
    is the declared-but-never-consumed defect: an eval that renames its
    aggregate cannot be adopted by editing YAML, which is the whole point of
    the seam.

    No default and no fallback literal. A task that declares no score key is a
    task nothing can score, and the ValidationError that says so is a better
    answer than one task's key quietly answering for another's.
    """
    return load_task_spec().slots.score_key


def require_track(name: str, *, flag: str = "--track") -> str:
    """Return ``name`` if the active task declares it, else raise ValueError.

    The one-layer-in refusal for parsers that are BUILT before ``--task`` has
    been read (``cli.build_parser``, and ``run_cycle._add_cycle_args`` before
    it): a ``choices=`` list there could only ever be one task's vocabulary.
    Closing the vocabulary here keeps the validation the ``choices=`` provided
    -- a typo'd track is still refused, and the message names the tracks THIS
    task declares -- without freezing the names at import time.
    """
    tracks = valid_tracks()
    if name not in tracks:
        raise ValueError(
            f"{flag}: invalid choice: {name!r} (task "
            f"{load_task_spec().name!r} declares {', '.join(tracks)})")
    return name


# ---------------------------------------------------------------------------
# Budget and thresholds (track-independent)
# ---------------------------------------------------------------------------

# STALL_WINDOW: consecutive SCORED, non-void trials that fail to beat the
# running best by NOISE_FLOOR before the track rolls over. It counts SCORED
# trials only (run_cycle.run_rounds) — infra-voids advance it by 0.
#
# History, both directions of which are regressions:
#   * 8 was FAR too small. It killed live training runs and pre-empted
#     copy-forward's first build-on-top trial, so it was disabled outright.
#   * 999999 disabled the rollover. Cycle 8 logged stall=950/999999 and, with
#     the budget gate also reading a dead counter, the track could never end.
#
# 400 is the bounded default:
#   * The worst observed gap between cycle-bests in c8 was ~250 consecutive
#     non-improving scored trials, so 400 keeps ~1.6x margin over the observed
#     worst case and cannot pre-empt a win of a kind we have actually seen.
#   * MANIFEST's stall epsilon (0.006) is STRICTER than NOISE_FLOOR (0.004),
#     i.e. it would classify MORE trials as non-improving — the margin above
#     absorbs that too.
#   * 400 is >3x the largest per-track budget, so on a healthy cycle the budget
#     gate always fires first; the stall window is purely the backstop for an
#     over-supplied / dead track (exactly the c8 failure mode).
# OVFX_STALL_WINDOW still overrides (set it high to disable again if needed).
STALL_WINDOW = _env_override("OVFX_STALL_WINDOW", 400)


def accepted_unsubmittable_base_ref() -> str:
    """The ref (if any) the operator has knowingly accepted as Labless-unsubmittable.

    The Labless base_ref gate (``labless_surface.assert_ref_submittable``) REFUSES a base
    a cycle's bundles could never be reviewed from. This is its one escape hatch, and it
    is deliberately shaped so it cannot rot: the value must be the ref ITSELF, not a
    boolean, so an override left in an operator's shell cannot silently cover a different
    base three rollovers later. Read at CALL time, not import time, so a rollover run in
    the same process as an earlier check sees the environment it was actually given.
    """
    return (os.environ.get("OVFX_ACCEPT_UNSUBMITTABLE_BASE_REF") or "").strip()

# Preprobe baseline-mint deferral budget.
#   mint_baseline_panel retries a launch-gate INFRA block (pannuke co-location)
#   4x90s (~6 min) and then returns status="deferred". But the condition that
#   produces that block is the TRAIN track's own node packing, which holds
#   node-1/node-2/node-7 at/over the co-location threshold for HOURS -- far longer than
#   6 minutes. Since run_rounds is submitted exactly ONCE per track, returning
#   on that first defer silently kills the preprobe track for the entire cycle
#   (c9 lost its whole preprobe track this way). These bound how long the
#   preprobe thread will wait the environmental block out before giving up.
#   Waiting is free: it blocks only the preprobe thread, and the train track
#   runs in a sibling thread that is entirely unaffected.
PREPROBE_MINT_DEFER_WAIT_S = _env_override("OVFX_PREPROBE_MINT_DEFER_WAIT_S",
                                           21600)  # 6h total
PREPROBE_MINT_DEFER_POLL_S = _env_override("OVFX_PREPROBE_MINT_DEFER_POLL_S",
                                           600)    # re-mint every 10 min

# Recency window for the failure_analysis TRIGGER's void clustering. Voids older
# than this do not count toward VOID_CLUSTER_BLOCKING.
#
# c8 fired the failure_analysis brief 192 times because the trigger counted
# LIFETIME voids per root cause: setup_failed had 1213 voids whose last
# occurrence was 126h earlier, so the counter sat permanently above the
# threshold and every wave paid for another LLM call. Matches the default
# recency window agents.briefs.collect_void_clusters uses to mark a cause
# `stale`, so the trigger and the brief agree on what is "happening now".
VOID_TRIGGER_WINDOW_H = _env_override("OVFX_VOID_TRIGGER_WINDOW_H", 24.0)

# Consecutive dispatch waves in which EVERY lineage RAISED (an orchestration /
# infra fault, never a void) before the track aborts. Without it a wave that
# always raises spins forever: nothing commits, so the budget gate never
# advances, and reconcile re-enqueues the same non-terminal lineages. Voids
# RETURN a result dict rather than raising, so void storms keep retrying
# untouched (void != dead).
WAVE_FAULT_ABORT = _env_override("OVFX_WAVE_FAULT_ABORT", 3)

# --- void != dead re-open: idempotency + bound --------------------------------
# run_cycle._lineage_terminal_reason RE-OPENS a lineage whose every failure was
# infra-class and never got TESTED (void != dead). reconcile re-evaluates that
# predicate on EVERY wave, so with nothing persisted the same lineages were
# resurrected forever: 9,890 RE-OPENING events in 40 minutes on c8-preprobe with
# `committed` frozen at 2/20 — a livelock that starved four never-executed
# operator readout ideas out of the queue.
#
# The grant is now PERSISTED on the lineage record (policy.reopen.spend)
# so a driver bounce cannot reset it, and it is bounded two ways:
#
#  * LINEAGE_REOPEN_CAP — total grants ever. 2, because one grant buys the
#    untested mechanism a full fresh slate, and a second covers the one honest
#    repeat case (the re-opened slate hit a DIFFERENT, live infra fault). Past
#    two full slates with zero valid trials the blocker is environmental and
#    belongs to the watchdog/operator, not to another dispatch round — every
#    further re-open is pure queue bandwidth taken from untried science.
#  * LINEAGE_REOPEN_GRANT_TTL_H — how long ONE grant keeps answering "re-open"
#    for free. Inside the TTL the same question replays the SAME grant (so
#    reconcile and run_rounds agree within a wave without burning a second one,
#    and nothing re-logs). A new grant is only earned by a MATERIAL change — a
#    new completed trial — never by the clock alone.
#  * LINEAGE_REOPEN_RECLAIM_CAP — a grant is SPENT only when the lineage
#    actually DISPATCHED a trial under it. A grant that expired having bought
#    ZERO trials was consumed by an infrastructure fault, not by an attempt, so
#    it is RECLAIMED (re-issued without inflating `count`, since no evidence was
#    gained) instead of closing the lineage. c8 proved why: 44 never-executed
#    preprobe lineages — including three operator-authored ideas — were marked
#    terminal by "grant spent" after the livelock cap-killed them on arrival,
#    turning an infra bug into a permanent blocklist over untested mechanisms.
#    Bounded, because a lineage that cannot run even with a fresh slate is an
#    environmental problem for the watchdog/operator; and each reclaim costs a
#    full TTL, so it can never hot-loop.
LINEAGE_REOPEN_CAP = _env_override("OVFX_LINEAGE_REOPEN_CAP", 2)
LINEAGE_REOPEN_GRANT_TTL_H = _env_override("OVFX_LINEAGE_REOPEN_GRANT_TTL_H",
                                           1.0)
LINEAGE_REOPEN_RECLAIM_CAP = _env_override("OVFX_LINEAGE_REOPEN_RECLAIM_CAP",
                                           3)
# |delta| below this is noise. Was the literal 0.004; now floor_sigmas *
# seed_transfer_sd, which is EXACTLY 0.004 at the shipped defaults. The ratio is
# stored unrounded for that reason: plan section 3.1's rounded "1.18 sigmas"
# would have moved this to 0.004012.
NOISE_FLOOR = _NOISE.floor()

# ---------------------------------------------------------------------------
# Greedy-accumulate-positive exploitation (spec: "carry small positive-direction
# wins forward and build on them"). When ON, after each dispatch wave the L1
# orchestrator picks the best within-cycle POSITIVE parent (exploit_parent.py,
# single-seed carry allowed) and dispatches the advisory exploit fraction of the
# next wave as EXPLOIT lineages that build on it.
#
# SAFETY (baked in, not optional):
#  * STRICT positivity — GREEDY_EXPLOIT_MIN_DELTA defaults to 0.0, i.e. the
#    parent's aggregate paired_delta must be > 0. A within-noise NEGATIVE is
#    NEVER carried, so noise cannot accumulate into a regression.
#  * PROVISIONAL — a carried parent is never graduated here; graduation is still
#    decided at the Phase-6 reset via the existing graduate/reseed path, so a bad
#    single-seed carry is pruned cross-cycle.
#  * The explore fraction is preserved (state.py _PHASE_EXPLOIT_FRAC caps exploit
#    below the batch size), so exploration keeps hedging every wave.
# ON by default (the user wants greedy-accumulate); tunable via env.
# ---------------------------------------------------------------------------
GREEDY_EXPLOIT = _env_override("OVFX_GREEDY_EXPLOIT", True)
# EXPLOIT_SEED_TRANSFER_SD — the MEASURED cross-seed sd of paired_delta (c9: 0.0034).
# This is the canonical noise constant for cross-seed exploit decisions. It is
# ENV-OVERRIDABLE so experimental cycles can probe a different measured sd without
# editing code. CANONICAL SOURCE OF TRUTH — GREEDY_EXPLOIT_MIN_DELTA derives from it.
EXPLOIT_SEED_TRANSFER_SD = _env_override("OVFX_EXPLOIT_SEED_TRANSFER_SD",
                                         _NOISE.seed_transfer_sd)
# Min aggregate paired_delta (at the reseed seed) a parent must clear to be
# stacked on. DEFECT FIX (2026-08-12): raised from 0.002 (0.59 sigma of cross-seed
# noise) to EXPLOIT_SEED_TRANSFER_SD (0.0034 = 1 sigma). At 0.002, ~28% of
# pure-null cross-seed trials clear the bar by chance — the old floor admits more
# noise draws than signal. At 1 sigma, only ~16% do. Tradeoff: the tool selects
# less often, but when it selects, the pick is more likely real. The POOL floor
# (EXPLOIT_POOL_MIN_DELTA) is deliberately left lower (1 sd of same-seed replicate
# noise, 0.0016) so condensation still has candidates even when no aggregate clears.
# DERIVED from EXPLOIT_SEED_TRANSFER_SD; ENV-OVERRIDABLE for experiments.
GREEDY_EXPLOIT_MIN_DELTA = _env_override("OVFX_GREEDY_MIN_DELTA", EXPLOIT_SEED_TRANSFER_SD)
GREEDY_EXPLOIT_ALLOW_SINGLE_SEED = _env_override("OVFX_GREEDY_ALLOW_SINGLE_SEED", True)

# Eligibility floor for the exploit-parent POOL (orchestrator._rank_exploit_parents),
# which is a DIFFERENT question from GREEDY_EXPLOIT_MIN_DELTA above. That one gates
# the tool's single primary pick on an AGGREGATE across a lineage's trials; this one
# gates whether a lineage's BEST single measured trial is allowed to be built on.
#
# The pool used a 0.0 floor, i.e. "any positive delta". That admits a parent whose
# delta is indistinguishable from re-running the SAME config at the SAME seed, and an
# exploit that refines noise refines nothing. EXPLOIT_REPLICATE_SD is the MEASURED
# same-seed replicate sd of paired_delta (c9: 0.0016; best delta in the whole cycle
# was +0.0048, i.e. ~3 sd). One sd, one-sided, is the smallest floor at which a
# single measured delta is more likely signal than replicate noise.
#
# Why not 2 sd (0.0032): in c9 that leaves 2 of 45 lineages eligible and 1 above the
# pre-existing 0.004 greedy floor — which is exactly the starvation this floor is
# meant to end. Why not 0.0: it is not a measurement claim at all. 1 sd keeps 5 of
# c9's 12 positive lineages, enough for DISTINCT exploit parents plus a condensation
# slot, and a wrong pick costs one killable trial (unlike graduation, which is
# permanent) — so the asymmetry justifies the looser of the two defensible bars.
EXPLOIT_REPLICATE_SD = _env_override("OVFX_EXPLOIT_REPLICATE_SD",
                                     _NOISE.replicate_sd)
EXPLOIT_POOL_MIN_DELTA = _env_override("OVFX_EXPLOIT_POOL_MIN_DELTA",
                                       EXPLOIT_REPLICATE_SD)
# ---------------------------------------------------------------------------
# COPY-FORWARD / BUILD-ON-TOP (git-native seed-ref).  DEFAULT OFF.
#
# Within an L2 lineage, trial N+1 can BUILD ON the accumulated code+config of a
# prior scored trial instead of re-deriving from base_ref every time. The
# mechanism is GIT-NATIVE (NOT a filesystem copytree): after each scored,
# non-void trial the worktree's accumulated source is committed as a durable
# "seed head" (tag build-head/<lineage_id>/<n>) and the next trial's worktree is
# `git worktree add`-cut from that commit. CODE is carried via a git ref; the
# checkpoint is NEVER carried (weights still train from scratch off MANIFEST's
# pinned train_seed), so the foreign-resume guard is untouched.
#
# The "scaffold" is simply the special case where trial-1 is a mandatory
# code-build from base whose head seeds the rest — NOT a separate subsystem.
#
# OFF => behavior is BYTE-IDENTICAL to the pre-copy-forward driver (no seed
# commit, no seed cut, no journal field). ON => the copy-forward path activates.
#
# DEFAULT ON since the c8 post-mortem: it was written to bounce dark and be
# enabled at relaunch, and never was — which is one of the two reasons c8
# exploitation was 100% inert (0 of 48 exploit trials inherited their parent;
# effective_base_ref was the constant ad1eef6 on 626/626 dispatches).
# OVFX_COPY_FORWARD=0 still disables it.
#
# HISTORICAL CAVEAT — RESOLVED 2026-07-30, do not read this as current state.
# The flag WAS inert while head_commit was never journaled on trial_completed
# (0 of 1899 c8 events carried one, so journal.head_map() was always empty and
# orchestrator._resolve_seed_ref could only ever return base_ref). The three
# inheritance bugs behind that are fixed: 4ab6dc6 records the trial build head
# regardless of this flag, f5ca7e6 inherits the parent's CODE via its build-head
# (not just its config), d4856fd self-resolves the parent head for the legacy
# dispatch paths, and 234cd16 spends exploit slots on DISTINCT parents plus a
# condensation slot. Compounding is LIVE; a run that still shows a constant
# effective_base_ref is a fresh regression, not this known caveat.
# ---------------------------------------------------------------------------
COPY_FORWARD = _env_override("OVFX_COPY_FORWARD", True)
# Silent-default seed-selection policy when the L2 emits no spec["build_on"].
# "auto" = latest-if-improving-else-best (see orchestrator._resolve_seed_ref).
# The L2 may request "latest" | "best" | "base" | <n>; the ORCHESTRATOR resolves
# the concrete SHA (an L2 never authors an infra ref).
COPY_FORWARD_DEFAULT_POLICY = _env_override("OVFX_COPY_FORWARD_POLICY", "auto")
# Durable tag namespace for seed heads; protected from prune_leaked_branches
# (which only deletes BRANCHES). The cycle-end GC (worktree.py gc-build-heads)
# collects only tags whose source is PROVABLY redundant — reachable from a ref
# that survives the GC, or backed by a worktree still on disk. A tag recording a
# worktree that was archived and reclaimed is the sole source record for that
# trial and is structurally uncollectable; --apply additionally refuses an
# unscoped target set, so the destructive mode cannot default to the namespace.
BUILD_HEAD_TAG_PREFIX = "build-head"

# ---------------------------------------------------------------------------
# TRIAL WORKTREE REAP  (l3.run_trial leaked every worktree it ever cut)
#
# run_trial cuts one `git worktree add` per trial and NEVER removed it: 1048
# branches and 1434 worktree registrations accumulated in the nanopath repo and
# overfitX/ reached 40 GB. The compose path already reaps correctly
# (consolidation/compose.py::_drop_worktree); the dispatch path did not.
#
# The policy below is the DEFAULT-CORRECT one -- a cycle runs with zero
# environment variables set, so anything that has to be switched on by hand is
# not a fix. Each knob has an env override for the operator, never a requirement.
#
#  * REAP_TRIAL_WORKTREES -- master switch. ON: the disk cost is real and
#    recurring, and it is what makes operators reclaim worktrees mid-cycle,
#    which is what destroyed head_commit records in c8.
#
#  * REAP_ON_VOID -- OFF. A failed / void trial's worktree is EVIDENCE: it is
#    the applied edit, the config as it ran, and the slurm logs, and it is
#    exactly the trial an operator has to open. Successes are the ones nobody
#    reads. So the retention policy is "reap what converged, keep what broke".
#
#  * REAP_REQUIRES_RECORDED_HEAD -- ON, and it is the hard gate: a worktree is
#    only ever reaped once its build head is durably recorded as a repo-level
#    tag (build-head/<lineage>/<n>, which outlives the directory -- see 2abe59e).
#    With no recorded head the directory is the ONLY copy of the source a score
#    came from, and reaping it would silently convert "we ran this" into "we
#    have no idea what ran". Unrecorded => RETAIN and say so loudly.
#
#  * REAP_DROP_BRANCH -- OFF, deliberately. The 40 GB is DIRECTORIES; a branch
#    ref is ~100 bytes and is the operator's handle on the exact commit a trial
#    ran. Decoupling them buys back ~all of the disk at ~none of the
#    provenance cost. Branch refs are swept separately, end-of-cycle, opt-in
#    (main.py sweep-branches --apply) -- and that sweep now classifies a branch
#    whose worktree the reclaim ledger records as archived-and-gone as ARCHIVED,
#    not leaked. Without that distinction the post-archival repo reads as "every
#    branch is leaked", which is precisely when a sweep is most destructive.
# ---------------------------------------------------------------------------
REAP_TRIAL_WORKTREES = _env_override("OVFX_REAP_TRIAL_WORKTREES", True)
REAP_ON_VOID = _env_override("OVFX_REAP_ON_VOID", False)
REAP_REQUIRES_RECORDED_HEAD = _env_override("OVFX_REAP_REQUIRES_HEAD", True)
REAP_DROP_BRANCH = _env_override("OVFX_REAP_DROP_BRANCH", False)

SUSPECT_THRESHOLD = 0.01    # |delta| < 0.01 is suspect
GRADUATION_BAR = 0.01       # state.py lineage verdict --graduation-bar default
MAX_TRIALS_PER_LINEAGE = _env_override("OVFX_MAX_TRIALS_PER_LINEAGE", 12)
CLEAN_CYCLES_FOR_GRADUATION = 2   # Phase-6 reset: validated && clean_cycles>=2

# Deterministic verdict guard (orchestrator.apply_verdict_guard): a lineage is
# "void-capped" — i.e. an LLM `kill` with <2 valid trials is a LEGITIMATE kill
# rather than a premature one — once this many trials have voided (crashed /
# smoke-failed / apply-failed). Below the cap the mechanism has not had a fair
# shot at a valid trial, so the kill is downgraded to `continue`.
VOID_CAP = 3

# ---------------------------------------------------------------------------
# PREPROBE ADOPTION: "consistently better", not "better once"
#
# OPERATOR DECISION 2026-08-07 (verbatim): "for preprobe it needs to be
# consistently better for us to do it."
#
# THE MEASUREMENT. Re-minting the panel on an IDENTICAL frozen checkpoint moves
# panel_mean_overall by 0.0092 / 0.0130 / 0.0203 across the three checkpoints in
# overfitX/_preprobe_mint_history/. Pooled sd = PREPROBE_MINT_SD = 0.0053. A
# preprobe paired delta is one trial mint MINUS one baseline mint, so its
# standard error is sd*sqrt(2) = PREPROBE_MINT_SE = 0.0075. Against the 0.01
# bar an observed +0.0100 is 1.33 SE — one-sided p ~ 0.18. ONE MINT CANNOT
# ESTABLISH A READOUT WIN, and until now nothing said so.
#
# WHY n = 2. The SE of the MEAN of n mints is 0.0075/sqrt(n):
#
#     n=1  SE 0.0075   +0.0100 is 1.33 sigma   p~0.18
#     n=2  SE 0.0053   +0.0100 is 1.89 sigma   p~0.03
#     n=3  SE 0.0043   +0.0100 is 2.31 sigma   p~0.010
#     n=5  SE 0.0034   +0.0100 is 2.98 sigma   p~0.0014   (80% power)
#
# n=5 is what 80% power at a TRUE 0.010 effect costs. It is not affordable: a
# pp_eval takes ~1.9 h and the preprobe runner is a hard 1-at-a-time serializer,
# so 5 mints is 9.5 h per candidate and the 20-trial preprobe budget buys 4
# candidates. n=2 is the smallest n at which the EXISTING bar becomes a ~2-sigma
# claim instead of a 1.33-sigma one, and it costs 1.9 h only on the rare
# candidate that is positive at all (c9: 0 of 10). It matches
# DUPLICATE_CONFIG_REPLICATE_CAP on purpose — same argument, same number, so the
# gate that admits the replicate and the guard that requires it cannot drift.
#
# THE RULE (orchestrator.apply_verdict_guard). An `escalate` on the preprobe
# track is downgraded to `continue` unless some ONE readout identity has
# >= PREPROBE_ADOPT_MIN_MINTS scored mints, EVERY one of them positive, whose
# mean clears GRADUATION_BAR. All three clauses matter: n alone lets a
# +0.02/-0.01 pair through, "all positive" alone lets two +0.001 mints through,
# and the mean is the existing bar re-stated at the point of adoption — it is
# not moved, and this guard can only ever REFUSE, never promote.
#
# ASYMMETRY, deliberately. Only `escalate` is touched; `kill` is untouched, so a
# single -0.028 mint still rejects with no replication. Adopting a null readout
# writes it into the standing panel every later trial is measured against and
# corrupts the denominator for the rest of the cycle; discarding a real readout
# costs one idea out of a supply that is not the binding constraint. The error
# costs are not equal, so the evidence bars are not equal.
# ---------------------------------------------------------------------------
PREPROBE_MINT_SD = 0.0053   # measured: _preprobe_mint_history, identical ckpt
PREPROBE_MINT_SE = 0.0075   # sd*sqrt(2): one trial mint vs one baseline mint
PREPROBE_ADOPT_MIN_MINTS = _env_override("OVFX_PREPROBE_ADOPT_MIN_MINTS", 2)

# ---------------------------------------------------------------------------
# Void attrition caps (fix/voids-dont-eat-develop-budget).
#
# MAX_TRIALS_PER_LINEAGE counts only VALID (terminal, scored, non-void) develop
# trials — voids do NOT eat the develop budget. A moonshot lineage (which voids
# most, from bad-anchor pre-checks + a flaky backend) still gets its full 6 real
# analyze-and-build shots. Two bounded caps stop retry/regeneration from looping
# forever:
#
#  * INFRA_RETRY_CAP — TRANSIENT infra voids (qwen backend down, shared-output-dir
#    race, setup/env failure, OOM/disk crash) RE-RUN the same trial slot without
#    advancing the valid count. This is a CUMULATIVE per-lineage ceiling, so a
#    PERMANENT outage cannot loop forever; on exhaustion the lineage is killed
#    with reason `infra_retry_cap`.
#  * SPEC_INVALID_KILL_CAP — PERSISTENT spec voids (the L2/editor keeps emitting
#    an unrunnable spec: unresolvable anchor / apply_failed / inert_mechanism /
#    void_liveness / illegal readout label) get the existing bounded
#    regeneration, but after this many CONSECUTIVE spec-voids on the SAME develop
#    slot (the streak is reset by any valid trial) the lineage is killed with
#    reason `spec_invalid_kill_cap` instead of burning all 6 slots looping.
#
# Neither an infra retry nor a spec regeneration counts as a valid develop trial.
# ---------------------------------------------------------------------------
INFRA_RETRY_CAP = 8
SPEC_INVALID_KILL_CAP = 3

# ---------------------------------------------------------------------------
# FIRST-SCORED-TRIAL EARLY KILL (measured on the live c9 journal, 2026-08-07).
#
# Over c9's 24 lineages with >=2 SCORED trials, 8 had a first scored trial worse
# than -0.0034 (one cross-seed sigma below baseline). ZERO of the 8 ever
# recovered: the best subsequent paired_delta across all 8 lineages was itself
# -0.0034 — still negative. The 9 follow-up trials they bought (~14 GPU-h plus
# ~14 editor-hours) returned no signal, and the rule would have been 8-for-8
# correct.
#
# The counter-fact bounds what this constant may be used for. The symmetric
# 16 lineages whose first scored trial was >= -0.0034 ALSO never produced a
# later delta above +0.0034. So a first-trial delta separates "certainly dead"
# from "probably dead" and NOTHING else: it is a KILL predicate only. It must
# never be read as a positive predictor, never gate exploitation, and never
# promote a lineage.
#
# Threshold = -0.0034, the measured cross-seed sigma — deliberately NOT
# NOISE_FLOOR (0.004). NOISE_FLOOR is the "is this signal?" band used by
# apply_verdict_guard; this is the "is this evidence of harm?" band, and the
# two answer different questions. A within-noise first trial is a HOLD (the
# guard's existing behaviour) and is untouched here.
#
# A lineage killed by this rule is killed ON MEASURED EVIDENCE: it has >=1
# scored trial, so run_cycle._infra_only_untested is False and the void != dead
# re-open path can never resurrect it as "untested". That is the intent — this
# is not a void.
FIRST_TRIAL_KILL_DELTA = _env_override("OVFX_FIRST_TRIAL_KILL_DELTA", -0.0034)

# ---------------------------------------------------------------------------
# FIRST-TRIAL KILL DEFECT FIXES (2026-08-12).
#
# The original threshold (-0.0034, 1 sigma) had a ~16% false-kill rate on
# TRUE-NULL lineages: one noisy draw at 1-sigma fires 1-in-6 times by chance,
# and c10 lost 14/40 lineages that way. Three constants fix this:
#
#  * SEED_TRANSFER_SIGMA — the measured cross-seed sd of paired_delta (c9: 0.0034).
#    Named so every threshold derived from it is traceable to the measurement,
#    not a magic number.
#
#  * FIRST_TRIAL_KILL_DELTA_2SIGMA — the new main bar: 2 sigma below baseline.
#    At 2 sigma a true-null lineage draws this low with p~0.023 (vs 0.16 at 1σ),
#    cutting false kills from ~16% to ~2.3%. Requires FIRST_TRIAL_KILL_MIN_VALID
#    to be met before it fires (so a single trial, however bad, cannot kill unless
#    it is catastrophic — see below).
#
#  * FIRST_TRIAL_KILL_MIN_VALID — minimum valid (scored) trials required before
#    any score-based kill fires. With only 1 valid trial the 2-sigma bar is not
#    reached unless the result is catastrophic (handled by the carve-out below),
#    because one draw cannot separate signal from noise.
#
#  * FIRST_TRIAL_KILL_DELTA_CATASTROPHIC — CARVE-OUT: a single result THIS far
#    below baseline kills immediately without waiting for a second trial.
#    At 5 sigma (p~3e-7) a true-null lineage NEVER draws this low; the only
#    real-world source is a blow-up (c10 had four: -0.1132, -0.0627, -0.0341,
#    -0.0299). Without the carve-out, a confirmed blow-up would waste a second
#    full training run to "confirm" the obvious.
# ---------------------------------------------------------------------------
# Measured cross-seed sd of paired_delta (c9 journal, 2026-08-07). A SIXTH copy
# of 0.0034 that plan section 2 did not list among the five; same number, same
# provenance, so it reads from the same model. FIRST_TRIAL_KILL_DELTA_2SIGMA
# below is unchanged at -0.0068.
SEED_TRANSFER_SIGMA = _NOISE.seed_transfer_sd

# 2-sigma bar: main first-trial-kill threshold (requires >=FIRST_TRIAL_KILL_MIN_VALID).
FIRST_TRIAL_KILL_DELTA_2SIGMA = -(2 * SEED_TRANSFER_SIGMA)   # -0.0068

# Minimum valid (scored) trials before any score-based kill may fire.
FIRST_TRIAL_KILL_MIN_VALID = _env_override("OVFX_FIRST_TRIAL_KILL_MIN_VALID", 2)

# Catastrophic carve-out: a SINGLE trial this far below baseline kills immediately
# (5 sigma; p~3e-7 for a true null). No minimum-valid requirement.
FIRST_TRIAL_KILL_DELTA_CATASTROPHIC = -(5 * SEED_TRANSFER_SIGMA)   # -0.017

# Exploit deadlock fix: when select_exploit_parent declines but the pool is non-empty,
# label the slots as replication instead of returning [].
EXPLOIT_REPLICATE_INSTEAD_WHEN_NO_TOOL_PARENT = _env_override(
    "OVFX_EXPLOIT_REPLICATE_INSTEAD", True)

# Skip pool parents with this many consecutive waves of all-negative children.
# At 2, a parent that twice produced only negative children is considered exhausted
# and is skipped in favour of the next pool candidate.
EXPLOIT_CONSECUTIVE_FAILED_CHILD_CAP = _env_override(
    "OVFX_EXPLOIT_CONSECUTIVE_FAILED_CHILD_CAP", 2)

# ---------------------------------------------------------------------------
# DUPLICATE-CONFIG DISPATCH GATE (measured on the live c9 journal, 2026-08-07).
#
# c9 dispatched 11 distinct effective_diff values 2-4x apiece — 25 dispatches,
# nothing anywhere checking whether a config had already been measured. The
# worst, {"dino.lr": "0.000125->6.25e-05", "model.type": "...vits14_reg->
# ...vitb14_reg"}, ran FOUR times across two independent lineages
# (-0.0124, -0.0119, -0.0126, -0.0145) — three re-measurements of a decided
# negative.
#
# But a duplicate is not automatically waste. Those accidental replicates
# produced the single most useful measurement of the cycle: pooled within-config
# SAME-SEED replicate sd = 0.0016 (df=9), against the assumed cross-seed sigma
# of 0.0034. The harness therefore resolves ~0.004 effects at a FIXED seed, and
# 0.0034 is a SEED-TRANSFER penalty, not the measurement limit. So the gate
# CONVERTS duplicates rather than just rejecting them:
#
#   * prior measurement DECIDED (|paired_delta| >= NOISE_FLOOR) -> REFUSE. The
#     sign is already resolved; a re-run buys nothing. (3 of c9's dispatches.)
#   * prior measurement WITHIN NOISE -> ADMIT as a LABELLED replicate, up to
#     DUPLICATE_CONFIG_REPLICATE_CAP scored runs. This is where the 0.0016
#     number actually lives, and a second point at a fixed seed halves the sd of
#     the mean — this is real power, and labelling it keeps the noise floor a
#     maintained measurement instead of an accident. (3 of c9's dispatches.)
#   * prior dispatch still IN FLIGHT -> REFUSE. Nothing has been measured yet,
#     so a concurrent twin is pure duplicated spend, not replication.
#   * every prior dispatch VOIDED -> ALLOW. void != dead: the config was never
#     measured. (6 of c9's dispatches were exactly this and MUST stay legal.)
#
# PREPROBE ASYMMETRY (2026-08-07). The "decided" test above assumes one
# measurement resolves a sign. On the preprobe track it does not: re-minting the
# panel on an IDENTICAL frozen checkpoint moves panel_mean_overall by
# 0.0092 / 0.0130 / 0.0203 across the three checkpoints in
# overfitX/_preprobe_mint_history/ (pooled sd ~0.0053), so one mint against one
# baseline mint carries SE ~0.0075 and an observed +0.0100 is 1.33 SE, p~0.18.
# So on that track only the NEGATIVE side is decided by one run:
#
#   * prior mint <= -NOISE_FLOOR -> REFUSE, exactly as on train.
#   * prior mint POSITIVE        -> ADMIT as a labelled replicate. A positive is
#     a CANDIDATE, and the readout cannot be adopted until the win holds across
#     mints (orchestrator.apply_verdict_guard).
#
# The asymmetry is a difference in error COST, not a double standard: adopting a
# null readout writes it into the standing panel every later trial is measured
# against, while discarding a real one costs a single idea out of a supply that
# is not the binding constraint (10 scored c9 preprobe trials, 0 positive).
#
# DUPLICATE_CONFIG_REPLICATE_CAP counts SCORED runs of one config, so the
# default 2 admits exactly one deliberate replicate.
DUPLICATE_CONFIG_REPLICATE_CAP = _env_override(
    "OVFX_DUPLICATE_CONFIG_REPLICATE_CAP", 2)
# Fail-open switch, matching OVFX_REQUIRE_LIVE_DIFF / OVFX_REQUIRE_OUTPUT_DIR.
REQUIRE_UNIQUE_CONFIG = _env_override("OVFX_REQUIRE_UNIQUE_CONFIG", True)

# DEFER ATTEMPT CAP — bounds a status the void tables never covered.
#
# A runner may return status="deferred" meaning "I did NOT launch; the condition
# is transient, ask again later" (preprobe: `an existing pp_eval is still
# active`; a probe-timeout defer from 0088018). A defer is NOT a trial: nothing
# ran, nothing was measured, and it must never be typed as a void — VOID != DEAD
# already conflates too much, and a defer is weaker still ("we did not even
# try"). So `deferred` is deliberately absent from _VOID_STATUSES,
# _INFRA_VOID_STATUSES and every kill cap.
#
# LIVE c9 CONSEQUENCE: being in no table at all, it was bounded by nothing. The
# L2 loop re-dispatched, the runner deferred again, and 104 identical
# `preprobe launch deferred: an existing pp_eval is still active` rows were
# journaled as `trial_completed` — inflating the completion count (292 rows for
# ~188 real trials) and polluting every void statistic read off the journal.
# That is the tell recorded in [[policy-join-key-must-match]]: a repeated action
# with identical inputs means the planner cannot see its own results.
#
# The cap bounds CONSECUTIVE defers on one slot. Exceeding it ends the lineage's
# attempts for this wave WITHOUT a void and WITHOUT a verdict, so the lineage
# stays re-openable and the mechanism stays UNTESTED rather than being recorded
# as tried-and-failed.
DEFER_ATTEMPT_CAP = 6

# NO-FORWARD-PROGRESS BACKSTOP (issue #22 — preprobe void-livelock).
#
# INFRA_RETRY_CAP / SPEC_INVALID_KILL_CAP each bound ONE void *class*; a void
# that evades both typing rules (a future/unclassified void status, or a mix
# that stays under each cap) has NO ceiling on TOTAL dispatched trials, so an
# all-void lineage can burn past its intended MAX_TRIALS_PER_LINEAGE slate and
# — worse — a lineage left with verdict "continue"/absent and ZERO valid trials
# is re-enqueued by run_cycle.reconcile on every pass without ever making
# forward progress (the reported livelock: "6/6 void slots re-render continue").
#
# NO_VALID_PROGRESS_CAP is a classification-INDEPENDENT total-dispatch ceiling:
# a lineage that has DISPATCHED this many trials while producing ZERO valid
# (scored, non-void) trials is making no forward progress and is terminated with
# a distinct `void_exhausted` reason (orchestrator.run_lineage forced-kill +
# run_cycle._lineage_terminal_reason drop), instead of looping. Set to the
# theoretical max LEGITIMATE dispatch count (a full valid slate PLUS a full
# round of infra retries) so it can NEVER fire on a lineage still legitimately
# infra-retrying under INFRA_RETRY_CAP or one that already has real evidence.
NO_VALID_PROGRESS_CAP = MAX_TRIALS_PER_LINEAGE + INFRA_RETRY_CAP

# Transient-infra void statuses (retry the SAME spec) that l3.classify_failure
# does NOT already type as "infra". These are pre-dispatch GATE voids that clear
# on a fresh per-trial output_dir / once a racing peer finishes — transient, not
# a bad spec. classify_failure=="infra" (setup_failed / backend_unavailable /
# OOM-noted crash) is treated as infra too; EVERYTHING else void is a persistent
# SPEC void routed through regeneration + the spec-invalid kill cap.
# probe_failed is transient-infra: the checkpoint is VALID (training reached
# budget), only the probe/eval hung or crashed. The same-spec retry re-enters
# trial.step_launch, whose resume-probe self-heal scores the existing checkpoint
# (no retrain). Bounded by INFRA_RETRY_CAP so a deterministically-deadlocking
# probe can't loop forever. (nan_loss is deliberately NOT here — a NaN reproduces
# on the same spec, so it stays a persistent SPEC void routed through regen.)
INFRA_RETRY_STATUSES = ("void_shared_output_dir", "void_foreign_resume", "probe_failed")

# ---------------------------------------------------------------------------
# SHUTDOWN-INTERRUPTED void — its own first-class status, deliberately NOT one of
# the statuses above.
#
# A graceful shutdown (SIGTERM -> orchestrator.request_shutdown) means: finish
# what is already on the GPU, start nothing new. The L2 loop honoured that for
# the NEXT trial, but the L3 fix-and-relaunch path lives INSIDE a trial body and
# was ungated — so a trial that crashed 6h after SIGTERM still ran a 58-minute
# editor session and then submitted a fresh multi-hour training job.
#
# When the gates below refuse to start that work, the trial must terminate with a
# verdict that says WHY, and that verdict must be confusable with nothing else:
#   * NOT a success — nothing was measured, there is no score.
#   * NOT a plain void — a void reads "this mechanism was built/tested and it
#     failed". Nothing here was tested; the operator stopped the machine.
#   * NOT an infra void either, in the accounting sense: an infra void still
#     spends INFRA_RETRY_CAP, and an operator's shutdown must never consume a
#     lineage's retry budget. orchestrator._is_shutdown_void excludes it from
#     BOTH deterministic caps, and run_cycle._infra_only_untested exempts it from
#     the stale-fault recency window (a shutdown is not a fault that can "still
#     be live"), so the lineage is re-openable on the very next cycle.
# It types as `infra` in l3.classify_failure purely so the void != dead re-open
# path accepts it as the ABSENCE of evidence; the cap accounting is separate.
SHUTDOWN_VOID_STATUS = "void_shutdown_interrupted"

# ---------------------------------------------------------------------------
# Operator knobs — module-level cadence/caps (import-time snapshot; no consumer
# flips these at runtime). Each is `CONST = _env_override(...)`; edit the default
# here, the env var is the emergency override.
# ---------------------------------------------------------------------------

# Reaper poll cadence (s) for the freeze-bounded wave join — NOT a per-trial
# wall-clock cap; only how often the driver checks for a genuinely FROZEN job.
REAP_POLL_SEC = _env_override("OVFX_REAP_POLL_SEC", 120)
# Max meta-work (mid-cycle ideation) actions per reconcile pass.
META_WORK_CAP = _env_override("OVFX_META_WORK_CAP", 3)


# ---------------------------------------------------------------------------
# Operator GATE FLAGS — read at CALL time (consumers/tests flip these at
# runtime), so each is a thin function over _env_override. The env var is an
# EMERGENCY override; the default here is the live-driver behavior.
# ---------------------------------------------------------------------------


def require_live_diff() -> bool:
    """Legibility live-diff gate ON. Emergency off: OVFX_REQUIRE_LIVE_DIFF=0
    (a change we cannot MEASURE is never voided)."""
    return _env_override("OVFX_REQUIRE_LIVE_DIFF", True)


def require_seed_identity() -> bool:
    """Seed-identity gate ON. Emergency off: OVFX_REQUIRE_SEED_IDENTITY=0."""
    return _env_override("OVFX_REQUIRE_SEED_IDENTITY", True)


def require_base_identity() -> bool:
    """Base-identity gate ON. Emergency off: OVFX_REQUIRE_BASE_IDENTITY=0.

    Sibling of require_seed_identity, for the other half of what makes a paired
    delta mean anything: the trial and its control must come from the SAME
    architecture, not just the same seed."""
    return _env_override("OVFX_REQUIRE_BASE_IDENTITY", True)



def require_output_dir() -> bool:
    """Per-trial output-dir race gate ON. Emergency off: OVFX_REQUIRE_OUTPUT_DIR=0."""
    return _env_override("OVFX_REQUIRE_OUTPUT_DIR", True)


def pretrial_verify_enabled() -> bool:
    """QWEN pre-trial semantic verifier ON. Emergency off: OVFX_PRETRIAL_VERIFY=0."""
    return _env_override("OVFX_PRETRIAL_VERIFY", True)


def l3_parallel_review_enabled() -> bool:
    """L3 parallel cross-review ON. Emergency off: OVFX_L3_PARALLEL_REVIEW=0."""
    return _env_override("OVFX_L3_PARALLEL_REVIEW", True)


def l3_parallel_review_n() -> int:
    """L3 reviewer count (caller clamps to 2..5). Override: OVFX_L3_PARALLEL_REVIEW_N."""
    return _env_override("OVFX_L3_PARALLEL_REVIEW_N", 3)


def phase_override() -> bool:
    """Operator override of the phase-gate marker. Emergency on:
    OVFX_PHASE_OVERRIDE=1 (does NOT clear MANIFEST.stop)."""
    return _env_override("OVFX_PHASE_OVERRIDE", False)


def reaper_enabled() -> bool:
    """Freeze-triggered probe reaper ON. Emergency off: OVFX_REAPER_ENABLED=0."""
    return _env_override("OVFX_REAPER_ENABLED", True)


def consolidate_enabled() -> bool:
    """Phase-6 composition (consolidation deep module) ON. Emergency off:
    OVFX_CONSOLIDATE=0."""
    return _env_override("OVFX_CONSOLIDATE", True)


def ideator_tools_enabled() -> bool:
    """Ideator read-tool kill-switch ON (further gated to the oauth path in
    agents.ideator). Emergency off: OVFX_IDEATOR_TOOLS=0."""
    return _env_override("OVFX_IDEATOR_TOOLS", True)


# ---------------------------------------------------------------------------
# Env whitelist (spec §9.1): only these OVFX_* vars may reach training
# subprocesses; anything else OVFX_* is banned.
# ---------------------------------------------------------------------------

OVFX_ENV_WHITELIST = frozenset({
    # WHICH TASK the subprocess belongs to. ``--task`` is an argv flag and argv
    # does not cross a subprocess boundary the parent did not spell it into, so
    # without this entry every tool the driver launches resolved the DEFAULT
    # task no matter which one dispatched it -- task selection stopped dead at
    # the process boundary (``docs/rebuild_status.md`` §9.4 item 6).
    "OVFX_TASK_SPEC",
    "OVFX_TRACK",
    "OVFX_CYCLE",
    "OVFX_BASE_REF",
    "OVFX_BASELINE_DIR",
    "OVFX_PROBE_ONLY",
    "OVFX_RESULT_PATH",
})
# NB: OVFX_PHASE_OVERRIDE (BUG 3 hard phase gate) is deliberately NOT whitelisted
# — it is read from the orchestrator/l3 PARENT env in _hard_phase_gate, never
# forwarded to the training subprocess, so the spec-§9.1 whitelist is unchanged.

# Locked paths (spec §9.2) — the SDK must never edit these
LOCKED_PATHS = ("probe.py", "benchmarking/", "labless/submit_to_labless.py")

# WANDB env vars that must NEVER be inherited by tool subprocesses.
# WANDB gating is CONFIG-borne (train.py fires on project.wandb_mode ==
# "disabled" OR a smoke-named output_dir; make_train_smoke.py sets both for
# the smoke leg). launch_full.py runs smoke+full in ONE subprocess and sbatch
# exports the caller env (train_1gpu.sbatch uses --export=ALL on the Labless
# path), so a stray WANDB_MODE=disabled/offline in the ORCHESTRATOR's shell
# would silently silence every FULL run (the historical "fulls stopped
# reaching WANDB" bug — see trial.py:366-375's comment). Smokes are safe
# regardless: train.py passes mode="disabled" as an explicit wandb.init kwarg,
# which env cannot override. So: strip these from the inherited env; a tool
# wrapper may still force one EXPLICITLY via `extra` (run_preprobe does
# WANDB_MODE=disabled — preprobe is eval-only and never logs to wandb).
WANDB_ENV_BLOCKLIST = ("WANDB_MODE", "WANDB_DISABLED")


def get_python_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for tool/training subprocesses.

    Enforces the OVFX_* whitelist: every OVFX_* var not in OVFX_ENV_WHITELIST
    is stripped from the inherited environment, and passing a non-whitelisted
    OVFX_* var in `extra` raises. Also strips WANDB_ENV_BLOCKLIST vars from
    the INHERITED env (wandb on/off is config-borne; an inherited
    WANDB_MODE/WANDB_DISABLED would silence full runs through sbatch's env
    export) — an explicit `extra` entry may still set them deliberately.
    Prepends the nanopath venv bin to PATH.
    """
    env = {
        k: v for k, v in os.environ.items()
        if not (k.startswith("OVFX_") and k not in OVFX_ENV_WHITELIST)
        and k not in WANDB_ENV_BLOCKLIST
    }
    venv_bin = str(NANOPATH_VENV / "bin")
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    # WHICH TASK the child belongs to. ``--task`` is argv, and argv does not
    # cross a process boundary the parent did not spell it into, so every tool
    # the driver launched resolved the DEFAULT task no matter which one
    # dispatched it (``docs/rebuild_status.md`` §9.4 item 6). Set HERE, on the
    # child's env only: writing it into the parent's ``os.environ`` would make
    # one selection outlive the process that made it.
    env["OVFX_TASK_SPEC"] = str(task_spec_path())
    if extra:
        for k in extra:
            if k.startswith("OVFX_") and k not in OVFX_ENV_WHITELIST:
                raise ValueError(
                    f"env var {k!r} is not in the OVFX_* whitelist "
                    f"{sorted(OVFX_ENV_WHITELIST)} (spec §9.1)")
        env.update(extra)
    return env


# ---------------------------------------------------------------------------
# MANIFEST handling
# ---------------------------------------------------------------------------


def task_home() -> Path:
    """The RUNNING task's state home: where its MANIFEST and batches live.

    ``OVERFITX_HOME`` stays the relocation override and still wins outright
    (``tests/test_relocation_config.py`` is entirely about that). Underneath it
    the answer is the task's own declared ``home`` -- which is the difference
    between a second task having its own phase, cycle and live batch and it
    silently inheriting nanopath's. A task that declares no home (``fake``)
    keeps ``OVERFITX_ROOT``, so the offline path is unchanged.
    """
    env = os.environ.get("OVERFITX_HOME")
    if env:
        return Path(env)
    home = load_task_spec().home
    return Path(home).expanduser() if home else OVERFITX_ROOT


def manifest_path() -> Path:
    """``identity.manifest`` inside :func:`task_home` -- declared AND read.

    Both halves used to be literals here, so every task shared nanopath's
    MANIFEST and therefore nanopath's phase gate (measured:
    ``docs/phase5_results.md`` site A).
    """
    return task_home() / load_task_spec().identity.manifest


def load_manifest() -> dict:
    """Load the running task's MANIFEST."""
    mp = manifest_path()
    if not mp.exists():
        raise FileNotFoundError(f"MANIFEST.json not found at {mp}")
    return json.loads(mp.read_text())


#: The base a cycle resolves when its MANIFEST records nothing to resolve FROM.
#:
#: One constant, two readers that MUST agree: ``orchestrator
#: .resolve_train_base_ref`` returns it as its last resort, and
#: :func:`seed_manifest` DECLARES it. They disagreed until 2026-09-01, and the
#: disagreement is not cosmetic -- see the note in ``seed_manifest``.
LAST_RESORT_BASE_REF = "HEAD"


def seed_manifest(path: Path) -> dict:
    """WRITE a first MANIFEST at ``path``. Callers check absence themselves.

    A task's first cycle has no cycle state yet, and refusing it is what made a
    second task unrunnable without a hand-written file: the shipped task's
    MANIFEST has simply always existed, so nothing had ever had to write the
    first one (``docs/rebuild_status.md`` §9.3 row 3).

    Every seeded value is the TASK's own declared identity -- its first
    declared phase, cycle 1, and that cycle's batch under its own declared
    ``output_root`` -- so no literal belonging to one task is written into
    another's state.

    Takes the path rather than re-deriving it so the caller's answer is the one
    written: ``run_cycle`` resolves the MANIFEST through ``manifest_path``, and
    a writer that re-derived it would seed a different file than the one the
    cycle then reads.
    """
    spec = load_task_spec()
    cycle = 1
    doc = {
        "phase": spec.identity.phases[0] if spec.identity.phases else "",
        "cycle": cycle,
        "live_batch": str(Path(spec.identity.output_root)
                          / f"batch{cycle:02d}"),
        # DECLARE the base this cycle will actually resolve.
        #
        # A seeded MANIFEST used to say nothing about any base. That is not the
        # same as saying "one base, the default": ``manifest_bases
        # .is_default_base`` answers True only for the ref the MANIFEST NAMES,
        # so with no ``base_ref`` at all, ``orchestrator
        # .resolve_train_base_ref`` fell through to LAST_RESORT_BASE_REF and
        # every reader then asked ``is_declared_base`` about that same ref and
        # got False. Measured live (2026-09-01, localcv cycle 1): "exploit
        # assignments REFUSED: MANIFEST declares no bases['HEAD'] entry
        # (declared bases: none -- this is a legacy, top-level-only MANIFEST)"
        # -- refusing the ref the core itself had just chosen, and degrading a
        # whole wave to all-explore.
        #
        # The refusal is RIGHT and is not touched here: its job is to stop a
        # trial running under one base's name on another base's panel, seeds
        # and baseline dir. What was wrong is that resolution and declaration
        # named different things. Now they name the same one, so a first cycle
        # has exactly one base and it is DECLARED -- which is the condition the
        # refusal is checking for, not an exemption from it. A task that later
        # stands up a real second base still gets refused until it declares
        # ``bases[<ref>]`` with that base's own artifacts.
        "base_ref": LAST_RESORT_BASE_REF,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(doc, indent=2) + "\n")
    return doc


def get_live_batch() -> str:
    """Current live batch (e.g. 'overfitX/batch06', relative to repo root)."""
    return load_manifest().get("live_batch", "batch01")


def get_baseline_dir(base_ref: str | None = None) -> str | None:
    """The paired-baseline output dir for ``base_ref``, or None when nothing
    resolves.

    OVFX_BASELINE_DIR is the boundary knob every trial/worktree setup already
    honours (worktree.py do_setup exits 1 without it); the MANIFEST keys are the
    durable fallback so a reset plan can check the precondition BEFORE running
    anything destructive rather than discovering it at exit 1.

    ``base_ref=None`` means the cycle's default base -- byte-identical to the
    single-base behaviour. A pinned base resolves through
    ``manifest_bases.baseline_dir``, the ONE per-base resolver.

    The env override deliberately still wins, unchanged: it is a single global
    dir and cannot be base-aware, so making it lose here would change existing
    behaviour for no gain. A hand-set OVFX_BASELINE_DIR pointing at the WRONG
    base is caught downstream by ``orchestrator._base_identity_gate``, which
    reads the resolved dir's own recorded base rather than trusting the path.
    """
    env = _env_override("OVFX_BASELINE_DIR", None)
    if env:
        return str(env)
    try:
        manifest = load_manifest()
    except Exception:  # noqa: BLE001 — planning must stay dry-run-safe
        return None
    return manifest_bases.baseline_dir(manifest, base_ref)


#: The ONE naming convention for a SECOND base's in-batch artifacts.
#:
#: A batch dir has always held exactly one ``panel.json`` and one ``baseline/``,
#: because a cycle had exactly one base. With two bases those two names are
#: contended, and the failure mode is not a crash: base B's provisioning would
#: quietly REPOINT base A's ``<batch>/baseline/summary.json``, and every A trial
#: after that would compute its paired delta against a B checkpoint. (The
#: precedent is on record: four compose trials once trained INTO the baseline's
#: output dir and destroyed its weights.)
#:
#: So the default base keeps the bare names, byte for byte -- a single-base
#: cycle produces the identical paths it always has -- and every OTHER base is
#: suffixed with its ``manifest_bases.base_token``. Two bases can therefore
#: never name the same file, which makes standing up a second base purely
#: ADDITIVE on disk.
def base_suffix(base_ref, manifest: dict | None = None) -> str:
    """``""`` for the default base, ``".<token>"`` for any other."""
    if manifest is None:
        try:
            manifest = load_manifest()
        except Exception:  # noqa: BLE001 — planning must stay dry-run-safe
            manifest = {}
    tok = manifest_bases.base_token(manifest, base_ref)
    return f".{tok}" if tok else ""


def base_panel_path(batch_dir, base_ref=None, manifest: dict | None = None):
    """``<batch>/panel.json`` for the default base, ``<batch>/panel.<tok>.json``
    otherwise -- the per-base measurement grid.

    Deliberately derived from the BATCH DIR, not from ``MANIFEST.panel``: the
    rollover writes the NEXT cycle's panel while ``MANIFEST.panel`` still names
    the CURRENT one, so a MANIFEST-authoritative path here would send the mint
    into the live cycle's grid. The MANIFEST stays authoritative for READERS
    (``manifest_bases.panel``); this is the writer's address.
    """
    from pathlib import Path as _P
    return _P(batch_dir) / f"panel{base_suffix(base_ref, manifest)}.json"


def base_train_baseline_dir(batch_dir, base_ref=None,
                            manifest: dict | None = None):
    """``<batch>/baseline`` for the default base, ``<batch>/baseline.<tok>``
    otherwise -- the dir holding that base's TRAIN paired-delta control."""
    from pathlib import Path as _P
    return _P(batch_dir) / f"baseline{base_suffix(base_ref, manifest)}"


def get_batch_dir() -> Path:
    """Absolute path to the current batch directory.

    Mirrors tools-v2/lib/common.resolve_batch_dir: MANIFEST.live_batch is
    resolved against MANIFEST.repo. The two FALLBACKS are the task's own
    declared layout rather than nanopath's literals: ``identity.output_root``
    when a task's rollover writes no ``live_batch``, and ``repo`` (then the
    task home) when it records no repo. A task with no VCS declares ``repo:
    ""`` and used to have its artifacts resolved against nanopath's checkout.
    """
    manifest = load_manifest()
    spec = load_task_spec()
    cycle = manifest.get("cycle", 1)
    batch = manifest.get("live_batch") or spec.identity.output_root.format(
        cycle=cycle)
    if str(batch).startswith("/"):
        return Path(batch)
    return Path(manifest.get("repo") or spec.repo or task_home()) / batch


def batch_dir_for(cycle: int) -> Path:
    """Absolute path to the batch directory of an ARBITRARY cycle.

    ``get_batch_dir`` answers this for the LIVE cycle. The rollover needs the
    NEXT one, and both ``reset.build_reset_plan`` and ``rollover`` answered it
    with the literal ``OVERFITX_ROOT / f"batch{n:02d}"`` -- nanopath's
    ``~/nanopath/base-models/nanopath-jepa/overfitX``. A localcv consolidation
    therefore planned to mkdir its next batch, init its track state and mint its
    panel inside the NANOPATH checkout, while every other path in the same cycle
    used ``.localcv/runs``.

    Two shapes, told apart by what the task DECLARES rather than by its name:

      * ``identity.output_root`` already carries the cycle (nanopath declares
        ``overfitX/batch{cycle:02d}``). That layout's container is what
        ``OVERFITX_ROOT`` names, so this branch is byte-identical to the literal
        it replaces -- including the ``OVERFITX_HOME`` relocation override, and
        including the ``OVERFITX_ROOT`` seam every rollover fixture redirects
        this resolution through.
      * ``output_root`` names the CONTAINER of the per-cycle batch dirs
        (localcv declares ``runs``, and its MANIFEST records ``runs/batch01``).
        The next batch is then the LIVE batch's sibling, under the ``batchNN``
        convention that container already uses -- resolved through
        ``get_batch_dir``, so it lands inside the running task's own home and
        the two cannot name different trees.

    The second branch is the whole fix: a task that does not declare the cycle
    used to get the first branch's literal anyway.
    """
    if "{cycle" in (load_task_spec().identity.output_root or ""):
        return OVERFITX_ROOT / f"batch{cycle:02d}"
    return get_batch_dir().parent / f"batch{cycle:02d}"


def get_cycle() -> int:
    """Cycle number from the running task's MANIFEST."""
    return load_manifest().get("cycle", 1)


def user_ideas_path() -> Path:
    """Durable user-idea injection store (JSONL).

    Lives at the overfitX HOME, NOT inside a batch dir, so it SURVIVES Phase-6
    cycle resets (batch dirs are archived each cycle; the home is stable).
    Override with OVFX_USER_IDEAS (tests point this at a tmp path).
    """
    return Path(os.environ.get(
        "OVFX_USER_IDEAS", str(OVERFITX_ROOT / "user_ideas.jsonl")))


USER_IDEAS_MD = Path(os.environ.get(
    "OVFX_USER_IDEAS_MD", str(SDK_ROOT / "IDEAS.md")))

# Repo-tracked STRUCTURED direction seed (JSONL, one direction-shaped dict per
# line, full schema incl. mechanism_spec/opportunity_contingency). Unlike the
# free-form IDEAS.md, this carries the exact execution hint L2/L3 needs. Seeded
# into the durable store at cycle start (idempotent, existing ids never
# clobbered) so committed directions auto-propose at the next ideation.
USER_IDEAS_DRAFT = Path(os.environ.get(
    "OVFX_USER_IDEAS_DRAFT", str(SDK_ROOT / "injected_ideas_draft.jsonl")))


# git ref (tag) the preprobe shared worktree's model.py is reset to before each trial, so per-trial readout edits never accumulate.
PREPROBE_CLEAN_BASE_REF = "preprobe-clean-base"


def get_preprobe_worktree() -> Path | None:
    """Authoritative shared preprobe worktree from MANIFEST.tracks.preprobe.workspace.

    The consolidator provisions this once per batch; never scan the batch dir
    for a name containing 'preprobe'.

    A relative workspace is resolved against MANIFEST.repo (mirrors
    get_batch_dir), so the returned path is absolute and .is_dir() no longer
    depends on the process CWD. An absolute workspace is returned as-is.
    """
    try:
        manifest = load_manifest()
    except FileNotFoundError:
        return None
    ws = manifest.get("tracks", {}).get("preprobe", {}).get("workspace")
    if not ws:
        return None
    ws_path = Path(ws)
    if ws_path.is_absolute():
        return ws_path
    repo = Path(manifest.get("repo", str(REPO_ROOT)))
    return repo / ws_path


# ---------------------------------------------------------------------------
# Locating THE task spec
# ---------------------------------------------------------------------------
# THIS is where the task's NAME is allowed to appear. ``kernel/`` is
# task-independent by contract -- ``tests/test_kernel_model.py`` asserts that no
# kernel module names a specific task, and it caught this loader sitting there.
# The spec PATH is layout, and layout is this module's job; the env read belongs
# here for the same reason (``tests/test_env_migration.py``).
#
# Nothing in ``policy/`` calls this: a gate is HANDED its spec and never fetches
# one, which is what keeps that package pure and directly testable with a dict.
_TASK_SPEC_CACHE: dict[str, "_TaskSpec"] = {}
#: ``DEFAULT_TASK_SPEC`` is declared at the top of this module, beside the
#: noise model that also reads it.
#: THE selected task, resolved. ``None`` until something asks.
_SELECTED_TASK_SPEC: "Path | None" = None


def select_task_spec(path: "Path | str | None" = None) -> Path:
    """Record WHICH task this process runs, and return its resolved path.

    ``--task`` reaches here through ``cli.select_task``; ``OVFX_TASK_SPEC`` is
    the same choice spelled for a subprocess that has no argv of ours; the
    literal default is the last resort. There used to be TWO answers -- the CLI
    resolved ``--task`` and this module resolved the env var, and ``--task``
    reached only the CLI's -- so a run could dispatch nanopath's cycle while its
    own log line named the other task (``docs/phase5_results.md`` site B).
    """
    global _SELECTED_TASK_SPEC
    chosen = Path(path or os.environ.get("OVFX_TASK_SPEC") or DEFAULT_TASK_SPEC)
    _SELECTED_TASK_SPEC = chosen if chosen.is_absolute() else SDK_ROOT / chosen
    return _SELECTED_TASK_SPEC


def task_spec_path() -> Path:
    """Where the running task's YAML lives -- THE selection, made once."""
    return _SELECTED_TASK_SPEC or select_task_spec()


#: THE resolved adapter for the selected task, installed by ``cli.select_task``.
#: ``None`` means nobody has selected a task in this process.
_SELECTED_ADAPTER: "object | None" = None


def use_adapter(adapter: "object | None") -> None:
    """Record the resolved adapter beside the task path it was resolved from.

    HERE and not in a new module because this file is already where the
    selection is RECORDED: ``orchestrator``, ``run_cycle``, ``reseed`` and
    ``agents.l3`` are the library half that never saw ``--task`` and all reach
    the task through ``config``. The runner half of the same decision is
    installed the same way, into ``tool``/``state`` (``cli.install_runner``).

    ``config`` does not import ``adapters`` to do this: the object arrives
    already built from ``cli.resolve_adapter``, so there is exactly one resolver
    and no static edge from the core into an adapter package.
    """
    global _SELECTED_ADAPTER
    _SELECTED_ADAPTER = adapter


def active_adapter() -> "object | None":
    """The installed adapter, or ``None`` -- "no task was selected here".

    ``None`` is a real answer, not a failure, and it is the same shape as
    ``tool.active_runner()``: a process that never chose a task has no task to
    ask a task-shaped question of, and the caller must be able to proceed. Every
    production path goes through ``cli.main`` -> ``cli.select_task``, which
    installs one before any command runs.
    """
    return _SELECTED_ADAPTER


def load_task_spec(path: "Path | str | None" = None) -> "_TaskSpec":
    """The loaded, CACHED ``TaskSpec``.

    Cached per resolved path because gates are asked one at a time on a hot
    dispatch path, and re-parsing the YAML per gate would make the stage runner
    the slowest thing in a wave. RAISES rather than defaulting: a caller that
    cannot read its own declared thresholds must not proceed on some other
    module's fallbacks -- that is exactly how a "port" silently becomes a policy
    change.
    """
    resolved = str(Path(path) if path is not None else task_spec_path())
    if resolved not in _TASK_SPEC_CACHE:
        _TASK_SPEC_CACHE[resolved] = _TaskSpec.load(resolved)
    return _TASK_SPEC_CACHE[resolved]
