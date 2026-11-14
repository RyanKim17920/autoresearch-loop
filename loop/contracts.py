"""loop.contracts — the records and vocabulary loop modules share, and the one plug-in
seam (:class:`ExecutionAdapter`). Records are frozen dataclasses; the store derives its
tables from them. Config lives in ``loop.spec``. Stdlib only (DESIGN D0.4)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable


#: every role the loop asks a model for; a spec's ``llm_roles`` may name only these
LLM_ROLES = ("researcher", "review", "scout")


class ForeignTaskError(RuntimeError):
    """A row not tagged with the active task id reached a consumer (D0.3)."""


class SubmissionRejected(RuntimeError):
    """sbatch REJECTED a submission (bad account/qos/partition): deterministic, never retried."""


class UnmeasuredLoopNoise(RuntimeError):
    """``loop run`` refused: a sigma is UNMEASURED; run ``python -m loop calibrate``."""


def require_task(task_id: str, rows: Iterable[Any], *, where: str) -> list[Any]:
    """``rows`` as a list iff every row's ``task_id`` is ``task_id``."""
    for row in (rows := list(rows)):
        if (got := getattr(row, "task_id", None)) != task_id:
            raise ForeignTaskError(f"{where}: row tagged task_id={got!r} reached a consumer of "
                                   f"task {task_id!r} (DESIGN D0.3)")
    return rows


PANEL = "panel"          # the weighted panel target, next to the per-axis targets
ROOT_NODE_ID = "root"    # the baseline node: paired control for every seed
# reject_reason of an identical knob exposure (loop.tune.adopt): a never-run CARRIER commit a dose ladder's points
# build on, not a rejection -- loop.quality leaves it out of every count ('expose_carrier': its pre-2026-10-04 name)
CARRIER = "carrier"
CARRIERS = (CARRIER, "expose_carrier")
#: loop.components: the OPTIONAL components (causally audited, may be auto-disabled) and what is never one
OPTIONAL_COMPONENTS = ("review_probes", "lever_prior", "design_auto",
                       "crossover_choice", "scout", "web", "idea_banks", "novelty")
PROTECTED_COMPONENTS = ("statistics_gates", "fresh_seed_confirm", "held_out_audit", "locks", "budget",
                        "eval_guard", "labless_gate", "init_compat")
K_INFLIGHT = "driver.inflight"   # [{id, strategy, family, parents}] of requests a researcher is working on
# {epoch, ts, reason, handled}: bumped when the world model's estimates are CORRECTED (a measurement fix); reviews
# made before it were reasoned on numbers since withdrawn (loop.meta.on_epoch)
K_EPOCH = "world.measurement_epoch"
K_STALE_META = "meta.stale"   # META hypothesis id -> {"nodes": its fallen evidence, "ts"} (loop.meta.mark_stale)


def outage_requests(store: Any) -> set[str]:
    """Research requests whose LATEST attempt ended ``backend_unavailable`` (an LLM / smoke-slot outage): no verdict
    on anything, so never a used bank idea, a spent review probe, a component-audit decision or a failure stat."""
    if not hasattr(store, "events"):
        return set()
    last: dict[str, bool] = {}
    for e in sorted(store.events("research_started") + store.events("research_failed"), key=lambda e: e["ts"]):
        last[e.get("request_id")] = e["event"] == "research_failed" and e.get("status") == "backend_unavailable"
    return {r for r, out in last.items() if out}


def current_reviews(store: Any) -> list[Any]:
    """The reviews made since the last measurement-epoch bump (all of them when there was none)."""
    ts = float((store.get_state(K_EPOCH) or {}).get("ts", float("-inf")))
    return [rv for rv in store.reviews() if (rv.created_ts or 0.0) >= ts]
DEFAULT_LEVERS: tuple[str, ...] = (
    "training_strength", "regularization", "effective_lr", "augmentation_strength",
    "capacity", "objective_difficulty", "data_curation", "readout")


def _enum(name: str, values: str) -> Any:
    return Enum(name, {v.upper(): v for v in values.split()}, type=str, module=__name__)


# parent = a non-root parent's control run at a child's seed; audit = held out from the world; tiebreak = a challenger's
# extra fresh-seed run against the audited incumbent (the retired loop.tiebreak: history only), held out like audit; remint = a run of the
# incumbent / an audited challenger on a never-used seed block (loop.remint: the audit comparator, the re-mint)
Stage = _enum("Stage", "baseline calibrate screen promote confirm parent audit tiebreak remint")
RunStatus = _enum("RunStatus", "submitting running done void crash")
VoidType = _enum("VoidType", "infra eval too_slow cancelled")   # infra is retried; eval / too_slow are verdicts;
# cancelled: the loop itself killed the run (its node closed), never retried and never an infra / void statistic
# evaluating: in the eval-only readout lane (loop.evalonly); eval_passed: it generalised and was handed to a
# full retrain on the incumbent (a new node), never a confirmed winner itself
NodeStatus = _enum("NodeStatus", "rejected queued screening screened_out promoting confirming "
                   "confirmed not_confirmed audit_failed vetoed void crashed baseline evaluating eval_passed")
Strategy = _enum("Strategy", "exploit explore crossover revisit baseline moonshot design")
Operator = _enum("Operator", "draft improve debug crossover")
HypothesisKind = _enum("HypothesisKind", "trial meta stack")   # stack: "A and B are additive / conflict"
HypothesisStatus = _enum("HypothesisStatus", "untested testing supported refuted inconclusive")

CLOSED_STATUSES = frozenset(NodeStatus(s) for s in (
    "rejected screened_out confirmed not_confirmed audit_failed vetoed void crashed eval_passed").split())
ACTIVE_STATUSES = frozenset({NodeStatus.SCREENING, NodeStatus.PROMOTING, NodeStatus.CONFIRMING})
BANDIT_ARMS = (Strategy.EXPLOIT, Strategy.EXPLORE, Strategy.CROSSOVER, Strategy.REVISIT, Strategy.MOONSHOT)
DESIGN_FAMILY = "design"       # a designed batch's nodes (loop.design): exempt from the one-family rule


@dataclass(frozen=True)
class LeverTags:
    """Lever -> direction in {-1, 0, +1}; an absent lever is 0 (D3.10)."""

    directions: Mapping[str, int] = field(default_factory=dict)

    def vector(self, vocabulary: Sequence[str]) -> tuple[int, ...]:
        unknown = set(self.directions) - set(vocabulary)
        if unknown:
            raise ValueError(f"levers {sorted(unknown)} not in vocabulary {list(vocabulary)}")
        return tuple(int(self.directions.get(l, 0)) for l in vocabulary)


@dataclass(frozen=True)
class AxisPrediction:
    sign: int                  # +1 / -1
    size: float                # score units, > 0


@dataclass(frozen=True)
class HypothesisRecord:
    id: str
    task_id: str
    kind: HypothesisKind
    claim: str
    mechanism_tag: str
    family: str
    predictions: Mapping[str, AxisPrediction]   # target -> prediction; primary is PANEL
    falsification: str
    levers: LeverTags = field(default_factory=LeverTags)
    source: str = "researcher"                  # researcher | review
    review_id: str | None = None
    created_ts: float = 0.0     # evidence is computed (WorldSnapshot.hypotheses), never stored here


@dataclass(frozen=True)
class Node:
    id: str
    task_id: str
    commit: str
    parent_id: str | None
    strategy: Strategy
    operator: Operator | None
    family: str
    mechanism_tag: str
    diff_hash: str
    status: NodeStatus
    island: str = "main"
    second_parent_id: str | None = None         # crossover only
    diff_path: str = ""
    hypothesis_id: str | None = None
    levers: LeverTags = field(default_factory=LeverTags)   # registered before any run
    debug_depth: int = 0
    confirm_alt: float | None = None            # frozen delta_1 (D3.7)
    confirm_e: float | None = None
    prior_bonus: float = 0.0                    # the retired tournament's bonus (stored history; unused)
    embedding: tuple[float, ...] | None = None
    reject_reason: str = ""
    created_ts: float = 0.0
    updated_ts: float = 0.0
    notes: str = ""


def design_point(node: "Node") -> bool:
    """A designed batch's point (its stacked optimum is Operator.DRAFT): screened on the batch's one
    shared block, never promoted, no postmortem, never a parent."""
    return node.strategy is Strategy.DESIGN and node.operator is None


def tune_control(node: "Node", stage: Any) -> bool:
    """A closed structural node under a tuning pass (loop.tune marks it ``tuning=t<b>``): its PARENT
    runs are the tuning trials' controls, LOW importance and preemptible-only like the trials."""
    return stage is Stage.PARENT and node.status in CLOSED_STATUSES and " tuning=t" in f" {node.notes or ''}"


@dataclass(frozen=True)
class Run:
    id: str
    task_id: str
    node_id: str
    seed: int
    stage: Stage
    status: RunStatus
    job_id: str = ""
    void_type: VoidType | None = None
    scores: Mapping[str, float] = field(default_factory=dict)
    aggregate: float | None = None              # the task's own score_key value
    submitted_ts: float | None = None           # first submission (kept across retries)
    gpu_h: float = 0.0
    output_dir: str = ""
    log_tail: str = ""
    attempt: int = 0                            # re-submits of the same row
    pool: str = ""                              # loop.slurm pool of the current attempt
    mitigation: str = ""                        # OOM ladder step this attempt runs with ('' = none)
    resumed: int = 0                            # starts that resumed from its own latest.pt (loop.resume): NOT
                                                # its seed's exact replicate, so its paired deltas carry seed noise

    @staticmethod
    def make_id(node_id: str, stage: Stage, seed: int) -> str:
        return f"{node_id}:{Stage(stage).value}:{int(seed)}"


#: a GPU out-of-memory in a log tail: torch's error, or the caching allocator's warning before a hang
OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError|memory allocation failed with OOM|"
                    r"CUDA error: out of memory", re.I)
_OOM_NOTE = re.compile(r"\s*\boom=\S+")


def is_oom(text: str) -> bool:
    return bool(OOM_RE.search(text or ""))


def oom_note(node: "Node") -> str:
    """The OOM ladder step a node's runs start at (``oom=<step>`` in its notes; '' = none)."""
    m = re.findall(r"\boom=(\S+)", node.notes or "")
    return m[-1] if m else ""


def with_oom_note(notes: str, step: str) -> str:
    return (_OOM_NOTE.sub("", notes or "") + f" oom={step}").strip()


def next_mitigation(ladder: Sequence[str], *applied: str) -> str:
    """The ladder step after the strongest of ``applied`` ('' when the ladder is spent). Steps are
    cumulative: step k runs with every step before it too."""
    rank = max((ladder.index(a) + 1 for a in applied if a in ladder), default=0)
    return ladder[rank] if rank < len(ladder) else ""


def stronger(ladder: Sequence[str], *applied: str) -> str:
    return max((a for a in applied if a in ladder), key=ladder.index, default="")


@dataclass(frozen=True)
class Postmortem:
    id: str
    task_id: str
    node_id: str
    verdict: str                                # loss | null | crash | void
    explanation: str
    mechanism_reasoning: str
    levers: LeverTags
    related_hypothesis_ids: tuple[str, ...] = ()
    suggested_next: str = ""
    created_ts: float = 0.0
    under_tuned: bool = False                   # a structural change judged at the incumbent's knobs (loop.tune)
    tune_knobs: str = ""                        # JSON [{key, low, high}]: the knobs most likely needing a retune


@dataclass(frozen=True)
class HandoffNote:
    id: str
    task_id: str
    node_id: str
    actions: str
    learnings: str
    concerns: str
    created_ts: float = 0.0


@dataclass(frozen=True)
class ReviewReport:
    id: str
    task_id: str
    trigger: str                                # cadence | stall | manual | first_principles
    summary: str
    findings: tuple[str, ...]
    meta_hypothesis_ids: tuple[str, ...]
    probe_briefs: tuple[tuple[str | None, str], ...]   # (meta id | None, brief)
    explore_bias: Mapping[str, float]           # family -> bias (clipped where used)
    n_evals_at: int
    created_ts: float = 0.0
    design: str = ""                            # JSON [{key, low?, high, levers?}]: a designed batch asked for
    selftune: str = ""                          # JSON [{knob, value, reason}] asked for (loop.selftune)
    proposals: tuple[str, ...] = ()             # rule changes for the operator (open until `loop ack`)


@dataclass(frozen=True)
class Posterior:
    mean: float
    sd: float
    n: int = 0

    @property
    def var(self) -> float:
        return self.sd * self.sd


@dataclass(frozen=True)
class NodePosterior:
    node_id: str
    increment: Posterior                        # epsilon_i over the parent (panel)
    utility: Posterior                          # U_i vs root (panel)
    axes: Mapping[str, Posterior]
    p_positive: float                           # P(U_i > 0)
    p_beats_incumbent: float
    vetoed: bool
    veto_axes: tuple[str, ...] = ()
    confirm_estimate: Posterior | None = None   # from confirm seeds only (D3.6)
    gain_p: float | None = None                 # P(own gain > 0) under the selection-aware EB mixture (loop/world/mixture)
    gain_prior: float | None = None             # ... and before its own seeds: the generation's prior P(eps > 0)
    gain_plus: float | None = None              # E[max(own gain, 0)] under the EB posterior: what a confirm can bank

    @property
    def p_gain(self) -> float:
        """P(own increment over the parent > 0): the node's OWN gain, not the carrier it inherits. Under the EB
        spike-and-slab (``gain_p``) the null spike counts as no gain: P(a real effect AND it is positive)."""
        if self.gain_p is not None:
            return self.gain_p
        m, s = self.increment.mean, self.increment.sd
        return (1.0 if m > 0.0 else 0.0) if s <= 0.0 else 0.5 * (1.0 + math.erf(m / (s * 2.0 ** 0.5)))

    @property
    def p_gate(self) -> float:
        """The screen / promote-gate probability: P(U>0) AND P(own gain>0) -- a child of a positive parent
        (the incumbent) passes on its own increment, never on the carrier's lead."""
        return min(self.p_positive, self.p_gain)


@dataclass(frozen=True)
class LeverEffect:
    lever: str
    target: str
    mean: float
    se: float
    p_positive: float
    n_nonzero: int


@dataclass(frozen=True)
class HypothesisEvidence:
    hypothesis_id: str
    e_value: float
    e_against: float
    n_trials: int
    status: HypothesisStatus
    trial_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class VarianceComponents:
    tau_node: float
    tau_tag: float                              # fixed 0: the tag level of the tree was removed
    tau_family: float
    n_scored_at_fit: int
    cold: bool


@dataclass(frozen=True)
class WorldSnapshot:
    task_id: str
    n_scored: int
    sigma: Mapping[str, float]                  # target -> paired-delta sd in force
    variance: Mapping[str, VarianceComponents]
    nodes: Mapping[str, NodePosterior]
    incumbent_id: str
    family_means: Mapping[str, Posterior]
    levers: tuple[LeverEffect, ...]
    hypotheses: Mapping[str, HypothesisEvidence]
    banked: tuple[str, ...] = ()                # confirm-and-condense: non-CONFIRMED winner-cross parents (world.banked)
    # node -> (measure parent, panel U_m - U_parent mean, var): a node measured against a contained ancestor
    measured: Mapping[str, tuple[str, float, float]] = field(default_factory=dict)
    # node -> (mean own increment on its fresh CONFIRM seeds alone -- the confirm e-value's numbers --, n seeds)
    own_confirm: Mapping[str, tuple[float, int]] = field(default_factory=dict)
    # selection-aware EB (loop/world/mixture): generation -> (prior components (w, mean, var), n); node -> (generation, own selection-
    # seed increment, its variance) -- the raw numbers a caller pooling more seeds re-posteriors (loop.select.pooled_gain)
    eb: Mapping[str, tuple[tuple[tuple[float, float, float], ...], int]] = field(default_factory=dict)
    eb_raw: Mapping[str, tuple[str, float, float]] = field(default_factory=dict)
    wm: Any = None                              # the predictive fit (loop.world.predictive.Fit) when in force


@dataclass(frozen=True)
class CandidateRequest:
    id: str
    task_id: str
    strategy: Strategy
    operator: Operator
    island: str
    parent_ids: tuple[str, ...]                 # 2 for crossover; ("root",) for a draft
    brief: str
    brief_kind: str = ""
    family_hint: str = ""
    meta_hypothesis_id: str | None = None       # a review probe
    bank_id: str = ""                           # an idea-bank idea (loop.select.bank)
    ablated: tuple[str, ...] = ()               # optional components bypassed for it (loop.components)
    persona: str = ""                           # ideation persona (loop.novelty)
    cell: str = ""                              # target mechanism type (MAP-Elites cell, loop.novelty)
    branch: str = ""                            # fork-and-flush branch <round>.<k> (loop.core.fnf), '' = shared
    shortlist: tuple[str, ...] = ()             # family|mechanism the brief points at (pre-checked by the researcher)
    claim_floor: float = 0.0                    # v3 target mode: a budget total below this is bounced once (c4i)
    rows: tuple[str, ...] = ()                  # v3: the brief's row ids a per-axis budget may draw on (c4i)
    k_candidates: int = 0                       # v3-c6i: candidate mechanisms the finish returns (0 = no slate)
    ref_gap: float = 0.0                        # v3-c6i: the gain the candidates' p_gap is stated against
    ref_kind: str = ""                          # v3-c6r: what ref_gap is (target | mde): calibration base rates per kind
    category: str = ""                          # v3-c6i: the mechanism-taxonomy category the brief suggested
    arm: str = ""                               # v3-c9 C2: "alt_base:<source>" = starts from a non-incumbent base (measured arm)


@dataclass(frozen=True)
class Candidate:
    request_id: str
    task_id: str
    commit: str
    parent_commit: str
    diff_text: str
    diff_hash: str
    changed_paths: tuple[str, ...]
    config_diff: Mapping[str, Any]              # flattened key -> new value
    hypothesis: HypothesisRecord
    handoff: HandoffNote
    init_compat: InitCompat | None = None


@dataclass(frozen=True)
class ResearchResult:
    status: str        # ok | failed | timeout | smoke_failed | init_incompatible | init_illegal | over_flop_cap | backend_unavailable | max_turns
    candidate: Candidate | None
    error: str = ""
    web: bool = False                           # the session had WebSearch / WebFetch
    web_searches: int = 0
    web_fetches: int = 0
    mechanism_type: str = ""                    # the researcher's declared type (loop.novelty validates it)
    dose_knob: str = ""                         # a code change's declared strength knob (loop.tune's dose ladder)
    stacks_with: tuple[str, ...] = ()           # incumbent legs the researcher expects it to stack with ...
    conflicts_with: tuple[str, ...] = ()        # ... or to conflict with (a swap test, loop.legs)
    identical: bool = False                     # a knob exposure: verified numerically identical at its default
    # first-principles record, idea id, config variants and branched follow-up plan (loop.agents.ideas, loop.variants)
    plan: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmPull:
    task_id: str
    arm: Strategy
    node_id: str
    reward: float
    gpu_h: float
    ts: float


@dataclass(frozen=True)
class Worktree:
    path: str
    branch: str
    base_commit: str


@dataclass(frozen=True)
class InitCompat:
    ok: bool
    reason: str = ""
    checks: Mapping[str, Any] = field(default_factory=dict)

    @property
    def swap(self) -> bool:
        """A new init (loop.adapters.initcompat: another model.type / checkpoint than the parent's)."""
        return "init_swap" in self.checks

    @property
    def rule(self) -> str:
        """The intake rule a failed check fires: init_illegal (a denied init source), over_flop_cap (the run would
        stop on the FLOP cap before the sample budget), else init_incompatible."""
        failed = {n for n, c in self.checks.items() if isinstance(c, Mapping) and c.get("ok") is False}
        return "init_illegal" if "init_legal" in failed else "over_flop_cap" if "flop_budget" in failed \
            else "init_incompatible"


@dataclass(frozen=True)
class SmokeResult:
    ok: bool
    reason: str = ""
    log_tail: str = ""
    unavailable: bool = False   # the smoke never got a slot (queue timeout): no verdict on the change
    infra: bool = False         # the ENVIRONMENT failed it (cache path, weights download, disk; loop.infra): no verdict either


@dataclass(frozen=True)
class JobHandle:
    run_id: str
    job_id: str
    output_dir: str
    submitted_ts: float


@dataclass(frozen=True)
class JobStatus:
    state: RunStatus                            # RUNNING | DONE | VOID | CRASH
    scores: Mapping[str, float] = field(default_factory=dict)
    aggregate: float | None = None
    void_type: VoidType | None = None
    gpu_h: float = 0.0
    log_tail: str = ""
    detail: str = ""
    restarts: int = 0                           # SLURM requeues of the job so far (preemptions it survived)
    requeued: int = 0                           # ... new since the last poll (gpu_h is then the time they lost)
    resumed: int = 0                            # starts that resumed from the run's latest.pt so far


@runtime_checkable
class ExecutionAdapter(Protocol):
    """How the loop touches code, GPUs and scores for one task (D6); ``Cls(ctx)``."""

    def root_commit(self) -> str: ...
    def prepare_worktree(self, *, parent_commit: str, name: str) -> Worktree: ...
    def reap_worktree(self, worktree: Worktree) -> None: ...
    def reap_orphans(self) -> list[str]: ...           # every research worktree; at driver start
    def commit(self, worktree: Worktree, *, message: str) -> str: ...
    def diff(self, worktree: Worktree) -> tuple[str, tuple[str, ...], Mapping[str, Any]]: ...
    def merge(self, worktree: Worktree, *, other_commit: str) -> tuple[bool, str]: ...
    def smoke(self, worktree: Worktree) -> SmokeResult: ...   # optional kw: memory=, mitigation=
    def init_compat(self, worktree: Worktree, *, parent_commit: str) -> InitCompat: ...
    def submit(self, *, run: Run, commit: str) -> JobHandle: ...   # idempotent per (run, attempt)
    def poll(self, handle: JobHandle) -> JobStatus: ...
    def find(self, run: Run) -> JobHandle | None: ...
    def cancel(self, handle: JobHandle) -> None: ...                # no-op on a finished job
    # optional: ``oom_ladder(commit) -> tuple[str, ...]``, the numerics-neutral OOM mitigations a run
    # at ``commit`` can be relaunched with, weakest first; ``Run.mitigation`` names the step to apply


# ---------------------------------------------------------------------------------------------- the adapter contract
#: (11-10 generalization) THE ADAPTER A NEW TASK WRITES: the :class:`ExecutionAdapter` methods above plus the
#: ``config_rel`` attribute (the repo-relative config file every config reader edits) are REQUIRED. Everything below is
#: OPTIONAL: the loop probes it with getattr / hasattr and, when it is absent, does what the value says. Grouped by
#: capability. tests/loop/test_task_conformance.py keeps this catalogue complete (every probe in loop/ is listed).
ADAPTER_REQUIRED: tuple[str, ...] = ("root_commit", "prepare_worktree", "reap_worktree", "reap_orphans", "commit",
                                     "diff", "merge", "smoke", "init_compat", "submit", "poll", "find", "cancel",
                                     "config_rel")
ADAPTER_OPTIONAL: Mapping[str, str] = {
    # -- code / branch mechanics (git) --
    "repo": "no repo path: selfcheck / local runs / labless skip what needs the checkout",
    "CONFIG_REL": "(class default of config_rel)",
    "CONFIG_FILE": "(legacy alias of config_rel)",
    "python": "the project python comes from loop.project.python / <repo>/.venv",
    "ctx": "the researcher reads slurm limits from the loop's own context",
    "_show": "researcher / ideation tools read files at a commit through git directly",
    "_struct_cfg": "(set by the loop) struct-tune's per-adapter config cache",
    "dry_build": "(the dry-run executor's marker) builds are real",
    "base_config": "the root config is read through config_at(root_commit)",
    "config_at": "no config-only analysis (pairs config moves, near-dup detection, struct-tune, design points)",
    "file_at": "local-run confirm nodes cannot be built",
    "commit_files": "local-run confirm nodes cannot be built",
    "changed_files": "re-root ports / grafts cannot check overlapping files",
    "diff_commits": "single-key change detection is off",
    "trees": "no tree-identity dedupe of commits",
    "ancestry": "no lineage walk across commits (driver falls back to the ledger's parents)",
    "graft": "no crossover / backfill port",
    "graft_port": "no re-root ports (test.reroot_port) and no code-leg ports",
    "graft_keys": "no config-key graft onto a new incumbent",
    "warm_worktree": "node worktrees are created at submit time",
    "worktrees": "no worktree tidy listing",
    "tidy": "stale worktrees are not reaped by prune",
    # -- launch / SLURM --
    "command": "calibrate prints no launch command",
    "oom_ladder": "an OOM crash is not relaunched with a mitigation",
    "repool": "a run cannot move pools (preempt backoff / stale-pending moves are off)",
    "pool_capacity": "pool slots are taken as declared (no effective-capacity cut)",
    "free_gpus": "the per-tick new-submit cap is the constant slurm.max_new_per_tick",
    "pending_reason": "stale-pending detection is off",
    "pending_jobs": "the courtesy backoff (slurm.protect_jobs) sees no pending jobs",
    "release_receipt": "nothing to release after a terminal run",
    "nice": "slurm.low_nice / run_nice are not applied",
    "concurrent_submit": "submits run serially",
    "pool_picker": "(set by the loop) the adapter picks no pool itself",
    "postprocess": "no per-run post-processing (run diagnostics)",
    # -- smokes --
    "smokes": "no in-flight smoke registry (smoke caps count nothing)",
    "smokes_pending": "smoke-pending gates see none",
    "smoke_capacity": "research sessions are never paused for smoke capacity",
    "smoke_hold_wait": "infra re-issues do not wait out the smoke hold",
    "cancel_smokes": "orphaned smokes are not cancelled at startup / stop",
    "smoke_gate": "(set by the loop) no smoke gate",
    "smoke_config": "selfcheck train_config is skipped",
    "nan_scan": "selfcheck nan_scan is skipped",
    "KNOWN_GOOD_SMOKE_LOG": "selfcheck nan_scan is skipped",
    "TRAIN_REL": "selfcheck train_config is skipped",
    "DEFAULT_CONFIG_REL": "selfcheck merges onto config_rel itself",
    "SELFCHECK_MODEL": "selfcheck builds no model (asserts + config merge only)",
    "SELFCHECK_ENV": "selfcheck adds no task env vars",
    "SELFCHECK_OVERRIDES": "selfcheck merges no dotlist overrides onto the smoke config",
    "lock_violations": "no locked-surface check beyond surface.locked at intake",
    "config_violations": "no adapter-level config rule (weights allowlist, caps) at intake",
    "LAUNCHER": "diagnose names no launcher",
    # -- resume after preemption (slurm.resume) --
    "resume_state": "a preempted run is treated as restarting from step 0",
    "interrupt": "the resume calibration and loop.resume tick are off",
    "resume_ok": "(set by the loop) resume permission",
    "resume_runs": "(set by the loop) runs allowed to resume",
    # -- eval-only readout lane (evalonly.enabled) --
    "eval_only": "no eval-only candidates",
    "plan_eval": "no eval-only runs",
    "eval_plan": "no eval-only runs",
    "clear_eval": "no eval-only runs",
    # -- scoring --
    "task_score": "local_runs re-scoring needs local_runs.scorer",
}


def adapter_conformance(adapter: Any) -> list[str]:
    """What an adapter (class or instance) lacks of :data:`ADAPTER_REQUIRED` ([] = it conforms). ``config_rel`` may be
    set per instance from the project block: a class declares its default as ``config_rel`` or ``CONFIG_REL``."""
    return [f"missing required {name!r}" for name in ADAPTER_REQUIRED
            if not hasattr(adapter, name) and not (name == "config_rel" and hasattr(adapter, "CONFIG_REL"))]
