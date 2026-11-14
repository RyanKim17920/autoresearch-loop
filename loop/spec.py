"""loop.spec — the config schema (the task spec's ``loop:`` block -> :class:`LoopConfig`)
and :func:`load_context`.

``build`` is a small strict decoder (an unknown key is a ``ValueError`` naming its path);
``loop.store`` reuses it for stored rows, so a record round-trips through one codec. The
noise model comes from the ACTIVE spec, never the root ``config`` (D0.4).
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
import types
import typing
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Union

import yaml

from kernel.model import UNMEASURED, TaskSpec, Unmeasured
from kernel.stats import NoiseModel
from loop import contracts as C
from loop import project as P
from loop.project import DomainCfg, ProjectCfg
from loop.contracts import DEFAULT_LEVERS, PANEL

_NONE = type(None)


@dataclass(frozen=True)
class SeedPlan:
    """Four disjoint seed sets (winner's curse, D3.6); screen blocks rotate above them."""

    screen: tuple[int, ...] = (11, 12)
    promote: tuple[int, ...] = (13, 14)
    confirm_pool: tuple[int, ...] = (101, 102, 103, 104, 105, 106)
    audit: tuple[int, ...] = (201, 202, 203, 204)
    calibration: tuple[int, ...] = ()           # () -> screen + promote + confirm_pool[:4]
    max_seeds_per_node: int = 0                 # > 0: hard cap, checked at spec load (_parse)
    block_children: int = 4                     # children of one parent that share a screen block
    # CONTROL SHARING (2026-10-06: 66 screen controls served no child, 154 served one): a parent's control run at a
    # screen-block seed is launched only once >= 2 live children (queued / screening) share that block, or the
    # requesting child has waited control_wait_h since its first own screen run; an existing same-seed run of the
    # parent (any reuse stage) is reused as before (its luck is the shared-offset term). 0 = launch at once.
    control_wait_h: float = 0.0
    promote_per_node: int = 0                   # > 0: promote seeds rotate per block through `promote`
    # False: no promote stage -- the screen seeds' posterior is the promotion gate (screen gate AND P(U>U_inc)) and a
    # passing node confirms at once; `promote` stays for nodes already PROMOTING (and keeps the block seeds put)
    promote_stage: bool = True
    # SEED ROTATION (loop.remint; operator 2026-10-04: "the original point of overfitx was to prevent winner's curse by
    # switching seeds"): each new held-out audit draws its seeds -- as many as ``audit`` -- from the never-used pool
    # (>= fresh_base, never run by any node at any stage nor drawn before); False: every audit on ``audit``
    audit_rotate: bool = True
    # CONFIRM ROTATION (loop.remint.confirm_block): each parent's confirming children draw fresh never-used confirm
    # blocks (<= block_children children per block; the parent runs a control on each seed a child uses);
    # False: every confirmation on ``confirm_pool``
    confirm_rotate: bool = True
    fresh_base: int = 100_000

    @property
    def calibration_set(self) -> tuple[int, ...]:
        return tuple(self.calibration) or tuple(dict.fromkeys(
            (*self.screen, *self.promote, *self.confirm_pool[:4])))

    def __post_init__(self) -> None:
        seen: dict[int, str] = {}
        for name in ("screen", "promote", "confirm_pool", "audit"):
            seeds = getattr(self, name)
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError(f"seeds.{name} {list(seeds)} is empty or has duplicates")
            for s in seeds:
                if s in seen:
                    raise ValueError(f"seed {s} is in both seeds.{seen[s]} and seeds.{name}")
                seen[s] = name
        if self.block_children < 1 or not 0 <= self.promote_per_node <= len(self.promote):
            raise ValueError("seeds.block_children must be >= 1 and 0 <= promote_per_node <= len(promote)")

    @property
    def n_promote(self) -> int:
        return self.promote_per_node or len(self.promote)


@dataclass(frozen=True)
class NoisePriorCfg:
    """An EXTERNAL noise prior (seed pairs measured elsewhere, e.g. the leaderboard's replicated runs): PER-RUN sds
    worth ``df`` degrees of freedom. ``loop calibrate`` pools its own measurement with it (df-weighted variances,
    written as the measured sds; the calibration df recorded is its own + ``df``), so a 4-df calibration is not the
    only evidence. df 0 = none (nanopath)."""
    df: float = 0.0
    panel_sd: float = 0.0                       # per-run sd of the official panel
    axis_sd: Mapping[str, float] = field(default_factory=dict)    # per-run sd per axis (slots.vector keys)
    source: str = ""


@dataclass(frozen=True)
class CalibrateCfg:
    """``loop calibrate`` placement: ``pools`` "auto" (the driver's routing: a calibration run is HIGH, primary first)
    | "spread" (round-robin over every pool with a free slot, so both pools' nodes / load appear in the replicate
    pairs); ``max_concurrent`` runs in flight at most (0 = the full capacity: production concurrency)."""
    pools: str = "auto"
    max_concurrent: int = 0


@dataclass(frozen=True)
class PromotionCfg:
    p_promote: float = 0.7
    p_confirm: float = 0.7
    alpha_confirm: float = 0.05
    alpha_audit: float = 0.05                   # the held-out audit's own bar (Labless gate)
    futility_e: float = 0.1
    # a screen whose first scored seed(s) leave P(own gain>0) below this is screened out at once: its other screen
    # runs are cancelled (void/cancelled), not spent (0 = always run every screen seed). Design points excepted.
    screen_futility_p: float = 0.1
    # EARLY CONFIRMATION (loop.seeds.early_confirm_bf): a screened node with raw mean > 0, futile <= P(gain>0) <
    # p_promote and a Bayes factor for a gain > 1 goes to CONFIRMING on fresh confirm seeds (it replaced the screen
    # extension, whose 2 selection-biased screen seeds left 1 confirm seed under the 5-seed cap)
    early_confirm: bool = False
    # ... and only when its predicted probability of reaching the confirm bar on those fresh seeds
    # (loop.seeds.confirm_power) is >= this (0 = no power condition)
    early_confirm_power: float = 0.0
    # VALUE OF INFORMATION (loop.seeds.promote_ok): a screen below p_promote still promotes when its EB posterior's
    # E[max(own gain, 0)] (P(gain>0) x the gain given one) >= voi_gain -- the gain its confirm seeds must be able to
    # bank to pay for them -- and P(U>0) >= p_promote (0 = off: the P bar alone)
    voi_gain: float = 0.0
    confirm_max_seeds: int = 6
    # SHRUNK GRADUATION (loop.legs / loop.campaign / loop.variants): a selection result (a leg factorial's optimum, a
    # campaign's / variant pass's best) whose SHRUNK (selection-corrected) estimate >= this x the confirm MDE goes to
    # fresh confirmation even when its raw estimate misses the null-max gate: the fresh seeds, not a second selection
    # gate, guard the winner's curse (live L4: shrunk +0.0042 vs MDE 0.0027 discarded on the null gate). 0 = off
    shrunk_grad_mde: float = 1.0
    delta_min_sigmas: float = 0.5
    # confirm-and-condense (loop.world.banked): a node with >= this many scored seeds (it reached fresh-seed
    # confirmation) and P(own panel increment > 0) >= bank_p joins the winner-cross stack pool even when not
    # CONFIRMED. Stack-building only: the graduation bar is unchanged. 0 = CONFIRMED nodes only.
    bank_min_seeds: int = 5
    bank_min_confirm: int = 2                   # ... of them confirm seeds: bank_p is judged on those alone
    bank_p: float = 0.8
    # the winner-cross stack HEAD is the incumbent unless a confirmed / banked node beats it with
    # P(U>U_inc) >= head_p (a null child on two lucky confirm seeds must not become everyone's partner)
    head_p: float = 0.8
    # a winner / bank cross leg needs P(own panel increment > 0) > leg_p (loop.select.winner_pairs)
    leg_p: float = 0.8
    # RE-MINT (loop.remint, operator 2026-10-04): every remint_hours (and for each new audited incumbent) the incumbent
    # and its top 2 audited challengers run on one never-used seed block; their incumbent comparison then counts only
    # seeds neither was selected on (screen / promote / confirm). 0 = off
    remint_hours: float = 0.0


@dataclass(frozen=True)
class BackfillCfg:
    """AUTOMATIC BACKFILL (loop.backfill): closed nodes a later verdict epoch invalidated are re-evaluated and the
    promising ones get their cheapest corrective action. Off: planned and shown (status / dry-run), never started."""
    enabled: bool = False
    min_gain: float = 0.0003                    # E[max(own gain, 0)] under the current posterior to be worth it
    max_active: int = 2                         # backfills in flight at once
    per_tick: int = 1                           # started per tick
    port_requests: int = 1                      # port_conflict researcher side requests in flight (0: book only)
    # KNOB SHIFT (loop.shift): closed nulls a new incumbent's changed config keys may have decided are ported onto it,
    # at most shift_max per shift; shift_flags maps a diagnostics flag (loop.diagnostics) to the config keys (full or
    # leaf) that relieve it -- a node whose own runs fired it is affected when the shift moves one of them
    shift_max: int = 20
    shift_flags: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class FamilyPrior:
    mean: float
    sd: float


@dataclass(frozen=True)
class PriorCfg:
    tau0: float = 0.005
    tau_floor: float = 0.0005
    tau_max: float = 0.05
    refit_every: int = 10
    refit_min: int = 10
    noise_min_df: int = 10
    # the source of every node's prior (loop/world/predictive.py): "predictive" (one hierarchical regression of a
    # node's own increment on its creation-time features, time-split evaluated: candidate ranking, allocation, the
    # plateau forecast; the gates' nonparametric prior per cell it chose out of time) or "tree" (the tag -> family
    # tree alone)
    model: str = "predictive"
    wm_min_nodes: int = 100       # scored nodes before the predictive model is used (fewer: the tree)
    wm_fresh_weight: float = 2.0  # weight of a node's fresh confirm-seed mean vs its first-screen mean in the fit
    wm_min_rho: float = 0.1       # a feature group whose out-of-time incremental rho is below this drops out
    wm_eval_every: int = 25       # new scored nodes between time-split evaluations (feature-group selection)
    wm_min_fresh: int = 15        # nodes with fresh seeds a generation's screen -> fresh map needs (else pooled)
    # VALUE GATES (waste: screened-out nodes used 61% of v2's GPU; design batches 588 GPU-h and revisits 157 for no
    # confirmation): a queued node / a new design batch / a revisit whose predicted P(fresh gain > MDE) is under
    # wm_value_p is not run, except a wm_explore share (deterministic per decision). 2026-10-06 out of time on 1,107
    # nodes: 0.003 skips 60% of their GPU-h and none of the 6 later confirmed (skipped: screen -0.0022 / fresh
    # -0.0005; kept +0.0027 / +0.0014). 0 = off.
    wm_value_p: float = 0.003
    wm_explore: float = 0.15
    # the screen gate only prunes near-duplicates of WELL-MEASURED mechanisms: a moonshot, an init swap, or a node whose
    # family / mechanism type / init has fewer than this many scored nodes (the fit extrapolates to ~0 there: a node,
    # an init swap, was rejected at P=0.0000) passes it
    wm_gate_min_support: int = 10
    # CARRIER FLOOR (operator: comparisons and building focus on the confirmed best): at least this share of research
    # draws build on the incumbent; the rest follow the predictive allocation (root gets most). A gain found on root is
    # brought onto the incumbent by the backfill's ``port``. 0 = allocation only.
    wm_incumbent_share: float = 0.5
    families: Mapping[str, FamilyPrior] = field(
        default_factory=lambda: {"architecture": FamilyPrior(mean=-0.02, sd=0.005)})


@dataclass(frozen=True)
class VetoCfg:
    harm_sigmas: float = 1.0
    prob: float = 0.8
    # the benchmark is the panel average: a node whose panel mean is positive is vetoed on an axis only
    # when that axis's drop is catastrophic (P(axis < -catastrophic_sigmas * sigma_axis) >= prob)
    catastrophic_sigmas: float = 4.0


@dataclass(frozen=True)
class HypothesisCfg:
    alpha: float = 0.1
    kappa: float = 0.5
    max_trials: int = 24          # a META fold's linked-trial limit (anytime-valid; only a cost bound)


@dataclass(frozen=True)
class LeverCfg:
    extra: tuple[str, ...] = ()

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return DEFAULT_LEVERS + tuple(l for l in self.extra if l not in DEFAULT_LEVERS)


@dataclass(frozen=True)
class FamilyRule:
    keys: tuple[str, ...] = ()                  # globs over flattened config keys
    paths: tuple[str, ...] = ()                 # globs over changed paths; "*" = catch-all
    init_check: bool = False
    combine: bool = False                       # may pair with ONE other family; filed under this one
    forbid: tuple[str, ...] = ()                # regexes an added diff line touching this family must not match
    forbid_config: tuple[str, ...] = ()         # "file:dotpath" in spec.repo: its names/paths join `forbid`


@dataclass(frozen=True)
class InitCompatCfg:
    paths: tuple[str, ...] = ()                 # a change here needs a passing check
    # init swap (loop.adapters.initcompat): regexes (case-insensitive) no init source / model.type, added code line
    # or config value may match (rule init_illegal); the max projected FLOP fraction at the full sample budget
    deny: tuple[str, ...] = ()
    # WEIGHTS ALLOWLIST (operator 2026-10-22: generic Meta DINOv2 / DINOv3 init LEGAL, pathology-pretrained ILLEGAL): when
    # ``allow`` is set, an init swap whose weights SOURCE (the model's pretrained_url / timm id) is non-empty must match
    # one of these regexes (else init_illegal; ``deny`` still wins); a source-less (random-init) architecture is not a
    # weights question. ``allow_types``: model.type values the O5 launch guard accepts without a recorded init-compat
    # pass. Both empty = no allowlist (deny only).
    allow: tuple[str, ...] = ()
    allow_types: tuple[str, ...] = ()
    flop_cap: float = 0.97
    references: tuple[str, ...] = ()            # read-only reference code for researchers (relative to the spec)


@dataclass(frozen=True)
class IslandCfg:
    n: int = 3
    split_at: int = 30
    reset_every: int = 10


@dataclass(frozen=True)
class BanditCfg:
    floor: float = 0.10
    explore_floor: float | None = None          # the explore arm's own floor (None -> floor)
    moonshot_floor: float | None = None         # the moonshot arm's own floor (None -> floor)
    moonshot_max_inflight: int = 2              # moonshot requests + queued (not screening) moonshot nodes at once
    moonshot_floor_min: float = 0.0             # OPERATOR minimum: self-tuning never takes moonshot_floor below it
    moonshot_max_inflight_min: int = 0          # OPERATOR minimum for moonshot_max_inflight
    discount: float = 0.95
    info_weight: float = 1.0
    gpu_h_min: float = 0.1
    crossover_min_p: float = 0.85     # a closed, un-vetoed node with shrunk P(U>0) >= this may be a crossover parent


@dataclass(frozen=True)
class IntakeCfg:
    near_dup_cos: float = 0.95
    queue_low_water: int = 2
    queue_max: int = 6
    compound_min_p: float = 0.8       # a 2-family candidate may stack knobs of nodes with shrunk P(U>0) >= this
    # v3-c8a B3 (a node changed the dataset AND the train resolution at once: confounded): LEVER GROUPS -- group ->
    # config-key globs. A diff whose keys span >= 2 groups is a MIXED change (loop.core.mixed): unless a crossover /
    # a stack (compound) / a claim with a [joint: ...] justification, it is split at intake into one QUEUED node per
    # group (the combined node held until every part has screened) -- or rejected ``mixed_change`` when
    # ``mixed_change: reject`` / when it cannot be split. A key no group names is NEUTRAL (num_workers, logging:
    # never a lever). {} = the rule is off.
    lever_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    mixed_change: str = "split"       # split | reject | off
    split_screen_seeds: int = 2       # a split mixed node waits until each part has this many seeds (or closed)
    # 11-08 NO-OP KNOB GUARD (nanopath lesson: ~7% of commits were numerically inert): flattened config key -> why it
    # does nothing in this recipe (read from the code). A candidate whose config diff touches one is bounced
    # ``noop_knob`` at intake / at the session's finish (typed, before any GPU). {} = off
    noop_keys: Mapping[str, str] = field(default_factory=dict)
    # ... and the adapter's CPU fingerprint of a config-only candidate (loop.adapters.nanobrain.noop_fingerprint):
    # parent vs child numerics identical -> its smoke fails ``noop_knob`` before any GPU. False = off
    noop_fingerprint: bool = False


@dataclass(frozen=True)
class ResearcherCfg:
    max_turns: int = 48                         # v3-c7i: 30 (+2 = the SDK's 32) bound 15 sessions in 24 h
    timeout_min: float = 20.0
    moonshot_max_turns: int = 60                # a structural change needs more reading and edits
    portfolio_max_turns: int = 60               # review fix S3: max_turns of a portfolio-enabled kind (explore / literature)
    moonshot_timeout_min: float = 45.0
    # RESEARCH STAGE (loop.agents.notes) of moonshot / literature / scout sessions: >= min_searches WebSearch calls,
    # >= min_new_papers papers READ (WebFetch) and noted this session, >= min_papers distinct papers built on (new +
    # cited notes); a distant_share of sessions ranges outside the field; notes_path '' = <state_dir>/research_notes
    min_papers: int = 5
    min_new_papers: int = 2
    min_searches: int = 2
    distant_share: float = 0.3
    notes_path: str = ""
    max_debug: int = 3
    max_concurrent: int = 2
    # researcher slots the driver's own request sources (winner crosses, debug fixes, tuning variants) may
    # hold together; the rest are the bandit's (restart re-issues are never blocked by it, but wait for a slot
    # under max_concurrent like every other session)
    max_side: int = 2
    # researcher slots the STACKING sources (winner crosses, the leg-factorial branch, the stack builder) hold
    # together, on top of max_side (never shared with it)
    max_stack: int = 2
    # DRAIN on stop (SIGTERM / deploy): no new LLM job, ticks go on until every research session in flight is
    # harvested or this many minutes passed (systemd's TimeoutStopSec must exceed it; 0 = exit at once)
    drain_min: float = 25.0
    # start no researcher while no pool has a free slot and every pool holds >= this many of our runs PENDING (or
    # runs none of ours while some wait): its smoke could not start soon; resumes once a slot frees (0 = never pause)
    pause_pending: int = 4
    # RESEARCH CIRCUIT BREAKER (loop.breaker, 11-08: nanobrain ran 1207 sessions in 10 h that all produced nothing):
    # this many sessions IN A ROW without a node, or (on a fresh history: no node ever passed intake) this many
    # sessions without a passing smoke, stop new research sessions until a redeploy or <state>/research_breaker.clear
    # (an ALERT names the dominant failure). 0 = that rule off
    breaker_consecutive: int = 6
    breaker_fresh_sessions: int = 30


@dataclass(frozen=True)
class ReviewCfg:
    # only STALL and FIRST-PRINCIPLES reviews run (cadence reviews removed 2026-10-06: 49/75 reviews, 65% of
    # them a near-duplicate -- MiniLM cos >= 0.8 -- of an earlier review's summary)
    stall_evals: int = 20
    stall_min_hours: float = 4.0                # a stall review at most once per this many hours since the last review
    bias_decay_evals: int = 50
    fp_every_reviews: int = 3                   # FIRST-PRINCIPLES review after this many big reviews (0 = off)
    fp_stall_closes: int = 30                   # ... or this many closes without a confirmation
    fp_min_hours: float = 12.0                  # ... but at most once per this many hours


@dataclass(frozen=True)
class DesignCfg:
    """DESIGN mode (loop.design): a resolution-IV two-level batch over 4..10 config knobs."""
    stall_closes: int = 0                       # > 0: start one when this many closes passed without a
                                                # confirmation (or since the last batch); 0 = on review request only
    runs: int = 16                              # 16 or 32 (K > 8 always takes 32)
    z: float = 2.0                              # |effect| >= z*se is significant (stacked, flagged)
    step: float = 1.5                           # an auto-picked numeric knob's high level: x or / by this
    max_knobs: int = 10                         # knobs per batch (4..10)
    lever_p: float = 0.0                        # > 0: a pool / auto knob must be backed by the lever regression:
                                                # one of its levers has P(effect in the knob's direction) >= this
    max_share: float = 1.0                      # of the slots, at most, held by designed-batch points (not tuning)


@dataclass(frozen=True)
class LegsCfg:
    """The LEG FACTORIAL on the stack carrier (loop.legs); on, it retires the top-P stack builder."""
    enabled: bool = False
    max_new: int = 12                           # new legs per batch ...
    max_factors: int = 8                        # ... pre-screened (highest own P first) to this many factors in all
    min_p: float = 0.7                          # a leg's pooled P(own gain > 0) ...
    min_seeds: int = 3                          # ... over at least this many scored seeds
    dose: bool = True                           # also the carrier legs' x0.5 / x2 dose design on the same branch
    min_new: int = 4                            # a further round on the same carrier once this many new legs qualify
    # a leg / winner-cross partner need not be CONFIRMED (loop.legs.promising): besides pooled P >= min_p over
    # min_seeds, a banked node, one with >= fresh_seeds own fresh confirm seeds of positive mean (0 = off), and
    # (port_legs) a backfill port onto the carrier with a positive own increment there qualify
    fresh_seeds: int = 3
    # CADENCE: a further round once >= 2 new legs qualify and this many hours passed since the carrier's last round
    # (0 = only at min_new); a round also runs on a CONFIRMED challenger with P(U>U_inc) >= challenger_p (0 = off)
    round_hours: float = 0.0
    challenger_p: float = 0.0
    port_legs: bool = True
    flip_p: float = 0.8                         # a factor leaves the carrier's level only at EB-shrunk P >= this
    null_q: float = 0.95                        # graduate only when the raw optimum beats this quantile of the
                                                # simulated global-null max (and the shrunk gain >= the confirm MDE)
    swap_max_sigmas: float = 1.0                # > 0: a queued child of the incumbent declared to conflict
                                                # with an incumbent leg whose own gain is under this many sigma also
                                                # runs with that leg OFF (a swap test); 0 = off


@dataclass(frozen=True)
class MoonshotTuneCfg:
    """MOONSHOT TUNING (loop.tune). A closed CODE-changing moonshot / explore / idea-bank node gets ONE DOSE LADDER:
    its own dose knob at 1/2x and 2x on its own screen seeds, a hierarchical dose-response fit picks the dose, and
    only that dose takes promote/confirm/audit on fresh seeds. A numbers-only structural node keeps the older pass
    (a small config design on its commit, then researcher variants), behind the fresh-confirm-seed gate."""
    enabled: bool = True
    ladder_p: float = 0.3                       # a code change ladders with shrunk P(own gain > 0) >= this
    ladder_scarce_p: float = 0.7                # ... in scarce mode only with P >= this (never fully paused)
    ladder_skip_types: tuple[str, ...] = ()     # mechanism types (loop.novelty.TYPES) never laddered (hparam-like)
    max_ladders_inflight: int = 3               # dose ladders running at once
    expose: bool = True                         # no dose knob: a researcher first exposes the hard-coded strength
    # a NUMBERS-ONLY node is tuned only when its OWN increment on its fresh CONFIRM seeds alone reaches this many
    # panel sigmas (loop.tune.gate; 0 = no such gate; a code change's ladder never needs it)
    min_confirm_sigmas: float = 1.0
    p_lo: float = 0.2                           # a screened-out node qualifies with shrunk P(U>0) in [p_lo, p_hi]
    p_hi: float = 0.85                          # ... or when its postmortem says under_tuned
    veto_min_p: float = 0.3                     # a VETOED node qualifies only when borderline: P(U>0) >= this,
    veto_max_sigmas: float = 2.0                # one vetoed axis, its posterior harm within this many axis sigmas
    rough_min_p: float = 0.3                    # MAJOR (code) change with P(U>0) >= this: promising but rough
    major_code_lines: int = 20                  # MAJOR: a new file / def / class, or this many changed code lines
    max_trials_major: int = 5                   # tuned variants for a promising-rough MAJOR node
    max_trials_minor: int = 2                   # ... for any other qualifying node
    max_concurrent: int = 3                     # tuning passes at once (new moonshots are never starved)
    shared: Mapping[str, float] = field(default_factory=dict)   # shared sensitive key -> step factor (high = x it)
    max_research_fails: int = 3                 # failed / rejected variant proposals before the pass ends


@dataclass(frozen=True)
class CampaignCfg:
    """DIFFICULT-TUNE CAMPAIGNS (loop.campaign): a closed init / architecture swap, new objective or loss, or a node
    whose researcher declared ``campaign: true`` gets ONE computed hyperparameter campaign (no LLM call per round):
    round 1 a spread over its search space, 1 seed per point vs its parent on its own screen seeds, then
    successive-halving refinement; the best shrunk config is judged only on fresh confirm seeds."""
    enabled: bool = True
    budget_gpu_h: float = 20.0                  # GPU-h of points per campaign (a round is trimmed to what is left)
    max_concurrent: int = 3                     # campaigns running at once
    round1_points: int = 8                      # round 1's spread (the first knob at up to 4 levels)
    refine_points: int = 4                      # points per refinement round
    rounds: int = 3                             # round 1 + refinement rounds
    max_knobs: int = 4                          # search-space knobs used
    round1_max: int = 16                        # round 1 = max(round1_points, 2 x knobs + 2), at most this
    # INPUT-SIDE changes (loop.campaign.input_side: an init swap, a data-sampling / augmentation mechanism, a node of /
    # a changed key in one of input_families, a changed diff line matching input_code) also search input_space
    # ({key, low?, high, scale} | {key, factor, scale}; no low: half the upward span below the value), past max_knobs
    input_families: tuple[str, ...] = ()
    input_code: str = ""
    input_space: tuple[Mapping[str, Any], ...] = ()
    # once per incumbent: {tag, levels: {key: value}, search_space: [range rows as above]} -> a config-only node on
    # the incumbent at ``levels`` with a declared campaign over the resolved search_space
    incumbent: tuple[Mapping[str, Any], ...] = ()
    retro_hours: float = 72.0                   # the backfill's campaign epoch reopens nodes closed this long before it,
    retro_max_runs: int = 2                     # ... judged on at most this many scored runs,
    retro_max: int = 10                         # ... at most this many (best E[gain+] first)


@dataclass(frozen=True)
class ExtrapolateCfg:
    """FOLLOW THE GRADIENT (loop.extrapolate): the winning-direction ledger over numeric config keys and the
    extrapolation lane continuing each direction on the incumbent; also the hard bounds range-edge extension
    (loop.campaign / loop.variants / loop.tune ladders) clips to."""
    enabled: bool = False
    max_inflight: int = 4                       # extrapolation nodes open (screening .. confirming) at once
    min_z: float = 1.0                          # a direction needs this net evidence (sum of clipped z, sigmas)
    max_item_z: float = 3.0                     # one evidence item counts at most this many sigmas
    mults: tuple[float, ...] = (1.0, 2.0)       # a round's steps, in units of the last winning step
    partner_z: float = 1.5                      # a substitute partner: factorial interaction z <= -this
    edge_max: int = 2                           # range-edge steps per variant pass / dose ladder (0 = off)
    every_min: float = 10.0                     # re-plan at least this often (else on an incumbent / open-node change)
    # knob metadata {key (fnmatch pattern), scale: log | linear, low?, high?}: first match wins; a key none matches
    # is linear in [0, 1] when its value lies there, else log (> 0; an int >= 1)
    knobs: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class DatabankCfg:
    """THE DATA BANK (loop.databank): datasets scripts/retile/build.py writes under ``root``. Ready non-pilot ones
    are registered in the researchers' data bank (the ``adopt`` ones also adopted on the incumbent through a
    campaign); researchers' ``dataset_spec`` requests are built one at a time under a disk cap; a node on a dataset not ready yet waits."""
    enabled: bool = False
    root: str = ""                              # the datasets' directory (nanopath_v2.yaml: /data/nanopath-databank)
    python: str = ""                            # build.py's interpreter (uv can't parse it); '' = the project python
    max_building: int = 1                       # builds in flight at once (any dataset under root, pilots included)
    max_total_tb: float = 2.5                   # datasets under root + building + the request (projected), TB
    min_free_tb: float = 1.5                    # build.py --min-free-tb
    retain_days: float = 14.0                   # unreferenced datasets this old are listed as prune candidates
    smoke_dataset: str = ""                     # a smoke of a node on a not-ready dataset reads this instead
    # datasets ADOPTED when ready (operator 2026-10-07: per dataset, none by default): a config-only node on the
    # incumbent + a campaign. A researcher's requested dataset is consumed by the requesting node itself.
    adopt: tuple[str, ...] = ()
    num_workers: int = 0                        # adoption sets train.num_workers to this (> the incumbent's; 0: no)
    # adoption campaign's declared search space: {key, low, high, scale} or {key, factor, scale} (value / f .. x f);
    # ``key[i]``: element i of a list value (a crop scale's lower bound); keys the config lacks are dropped
    adopt_space: tuple[Mapping[str, Any], ...] = ()
    note: str = ""                              # appended to a registered dataset's line when its tiles exceed 224 px
    # v3 (10-13: 5 banked datasets never ran on the current base): each registered dataset no node of the current
    # base trains on runs once per base epoch as a config-only node on the incumbent; QUEUED config-only dataset
    # nodes stranded on an older base are re-rooted onto the incumbent
    run_unrun: bool = True
    max_concurrent: int = 1                     # run_unrun nodes screening at once (smallest dataset first: I/O)
    screen_runs: int = 3                        # ... a node counts as screening until it has this many done runs
    max_retries: int = 2                        # a dataset whose node ended VOID / CRASHED with 0 done runs reruns


@dataclass(frozen=True)
class MetaCfg:
    """Per-component bias knobs (loop.meta), each in [-1, 1] and self-tunable by reviews: > 0 = be MORE of the
    component's X, < 0 = LESS."""
    researcher_bias: float = 0.0
    review_bias: float = 0.0
    first_principles_bias: float = 0.0
    design_bias: float = 0.0
    crossover_bias: float = 0.0
    scout_bias: float = 0.0
    persona_bias: float = 0.0
    # CONSOLIDATION (loop.meta.consolidate): with every first-principles review, or every this many big
    # reviews, an Opus call condenses the records into a short versioned PLAYBOOK (0 = only with FP reviews)
    consolidate_every: int = 3


@dataclass(frozen=True)
class ComponentsCfg:
    """CAUSAL COMPONENT AUDITS (loop.components): each optional component is bypassed on ``ablation_rate`` of
    its decisions (``rates`` per component); the audit moves a no-value one to ``down_rate`` then off (used on
    ``reprobe_rate`` only). ``enabled: {name: false}`` turns one off for good (operator; never audited back)."""
    ablation_rate: float = 0.15
    down_rate: float = 0.5
    reprobe_rate: float = 0.05
    min_ablated: int = 30
    equivalence_margin: float = 0.05     # no-value needs upper 95% bound < this (screened hit-rate difference)
    seed: int = 0
    enabled: Mapping[str, bool] = field(default_factory=dict)
    rates: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class NoveltyCfg:
    """NOVELTY ENGINE (loop.novelty): sweep redirection, MAP-Elites cells, personas, novelty bonus."""
    enabled: bool = True
    sweep_redirect: bool = True                 # ideation diffs that only move numbers go to the design knob pool
    weight: float = 0.5                         # bonus (reward units, before / GPU-h) at embedding distance 1
    major_ratio: float = 2.0                    # a numeric move LARGER than this (x or /, or from / to 0) is a MAJOR
                                                # hyperparameter change, not a sweep
    mech_dup_cos: float = 0.30                  # near_dup_mechanism: TF-IDF cosine of mechanism texts (0 = off);
                                                # live moonshots: unrelated pairs median 0.07 / p95 0.19, the repeat
                                                # clusters (CLS-decorrelation, hard-patch JEPA, salient) 0.11-0.43
    mech_dup_hours: float = 48.0                # ... against live nodes and those created this recently
    dead_extra: tuple[str, ...] = ()            # operator-declared low-yield mechanisms, appended to the dead list


@dataclass(frozen=True)
class ScoutCfg:
    """RESEARCH SCOUT (loop.scout): a periodic Opus session with web tools that writes cited ideas to an
    append-only idea bank (``path``; list it in ``idea_banks`` to serve them)."""
    enabled: bool = True
    every_hours: float = 6.0                    # self-tunable 2..24 (loop.selftune)
    stall_min_hours: float = 2.0                # a stall (review.stall_evals closes without a confirmation)
                                                # runs one early, at most once per this many hours
    path: str = ""                              # '' = <state_dir>/idea_bank_scout.jsonl
    min_ideas: int = 8
    max_ideas: int = 15
    max_turns: int = 40
    timeout_min: float = 30.0
    # the v3 tick runs the scout too (loop.tick: ``scout.v3``; it was wired only into the v2 driver) and v3 briefs
    # serve its bank (``ideation.bank_v3``). Default off: nanopath unchanged
    v3: bool = False


@dataclass(frozen=True)
class BoardCfg:
    """COMPETITION BOARD INGEST (loop.board): every ``every_hours`` fetch the project's public Labless experiment log
    + reference points (GET only), keep the runs in the loop db (``board_runs``), optionally their pushed source
    (config + code, read as TEXT, never executed), and compute a compact digest (seed-pair noise, per-axis profiles,
    contributor clusters, untested root keys, reference-model per-task gaps) that the v3 briefs / scout / root audit
    read. Default off."""
    enabled: bool = False
    project_url: str = ""                       # GET -> {reference_points: [...], ...}
    log_url: str = ""                           # GET -> {runs: [...]}
    source_url: str = ""                        # '{run_id}' template; '' = no source fetch
    every_hours: float = 6.0
    timeout_s: float = 30.0
    source_max: int = 40                        # run sources fetched per pass (the best unfetched first)
    config_path: str = "config.yaml"            # the run config inside a source tarball
    root_runs: tuple[str, ...] = ()             # board run titles that ARE our root recipe (its profile / untested keys)
    reference_facts: tuple[str, ...] = ()       # operator-stated facts shown verbatim in the digest
    ignore_keys: tuple[str, ...] = ("name", "notes", "output_dir", "wandb", "wandb_entity", "wandb_project", "seed",
                                    "num_workers", "prefetch_factor", "ckpt", "resume", "auto_resume",
                                    "checkpoint_period", "checkpoint_steps", "print_freq", "plot_period")
    top_k: int = 10
    digest_chars: int = 7000
    material_new_runs: int = 10                 # new runs (or a new board best) that make the digest 'materially' new


@dataclass(frozen=True)
class LocalRunsCfg:
    """LOCAL MANUAL-RUN IMPORT (loop.localruns): the operator's hand-launched runs (job tables: name / jobid /
    worktree / branch / sha / overrides / lane, TSV) are read from their output dirs, re-scored from the per-task
    jsons (the adapter's scorer), diffed against the root (effective config + ``.py`` code) and kept as EXTERNAL
    evidence (``local_runs`` table, never ledger nodes / runs): every v3 brief, the root audit and the digest show
    them as 'already measured locally' and a promising one is banked as a priority CONFIRM idea (run under the
    loop's own seeds). Idempotent (a run is re-read only when its files change); default off."""
    enabled: bool = False
    job_tables: tuple[str, ...] = ()            # TSV files with a header row (name + worktree / jobid ...)
    worktrees_root: str = ""                    # '' = loop.project.worktrees_root (a relative worktree column)
    # label -> commit, in order: the first that is an ancestor of a run's commit names its lineage
    lineages: Mapping[str, str] = field(default_factory=dict)
    incomparable: tuple[str, ...] = ()          # lineage labels on another base: absolute scores only, no delta
    reference_runs: Mapping[str, str] = field(default_factory=dict)   # seed -> the root replicate run of that seed
    # a PREVIOUS root's lineage label -> {seed -> that root's replicate run}: its runs become CONTEXT (deltas vs their
    # own root's replicate, never vs the current root, never banked) -- a re-root keeps the old evidence readable
    context_refs: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    scorer: str = ""                            # "module:function"; '' = the adapter's ``task_score``
    full_min_wall_s: float = 3000.0             # summary.wall_seconds below this = a smoke (stored, not briefed)
    every_hours: float = 0.5
    promising_axis_sd: float = 2.0              # an axis delta >= this many paired sd ...
    promising_open_sd: float = 1.0              # ... with the panel delta >= -this, or the panel delta >= this
    harmful_open_sd: float = 2.0
    ignore_keys: tuple[str, ...] = ("name", "notes", "output_dir", "seed", "wandb", "wandb_entity", "wandb_project",
                                    "num_workers", "prefetch_factor", "auto_resume", "resume", "ckpt", "print_freq",
                                    "plot_period", "plot_nrow", "plot_ncol", "plot_num_predict", "device", "debug")
    block_chars: int = 9000
    bank: bool = True                           # bank the promising groups as priority confirm ideas
    # a FULL run whose code (no ``.py`` diff vs the root commit) and effective config (only seed / name / output /
    # logging keys differ: ``ckpt`` / ``resume`` must match) are the ROOT's is a root seed replicate: registered once as
    # a done ``root:baseline:<seed>`` run (no ledger node; skipped when the root already has a run at that seed), so a
    # fresh root gets its seeds / noise fit from replicates already paid for (11-08: a root replicate run / a root replicate run / a root replicate run)
    root_replicates: bool = True
    # 11-09: a promising group is BUILT as a confirm node (no research session): the root's commit + the run commit's
    # changed writable files + its config delta, QUEUED on the root with ``local_confirm=<bank id>`` (screened first like
    # an operator program arm, never aged out), at most confirm_max_open open at once, best panel delta first
    confirm_nodes: bool = False
    confirm_max_open: int = 2


@dataclass(frozen=True)
class TestedCfg:
    """THE MECHANISM MEMORY (loop.core.tested, 11-08): every research session is held to what was already measured
    or is being built. The memory = closed MEASURED ledger nodes, the imported manual runs (loop.localruns) with their
    verdicts, and earlier slate candidates the gate bounced; in flight = the chosen candidates of live sessions and
    the QUEUED / running nodes. A slate candidate (and a ``check_idea`` mechanism) whose title + mechanism matches an
    entry at token similarity >= ``sim`` (``inflight_sim`` for in-flight work) is bounced with the prior result cited,
    unless its ``differs_from_tested`` names that entry and says HOW it differs (>= ``how_min_words`` content words
    the entry does not contain). Every brief lists the memory (one line per mechanism) and, at session start, the
    mechanisms in flight / proposed in the last ``recent_hours``. ``enabled: false`` = the old behaviour."""
    enabled: bool = True
    sim: float = 0.5
    inflight_sim: float = 0.5
    how_min_words: int = 3
    recent_hours: float = 12.0
    brief_max: int = 80                         # memory lines a brief lists (the manual runs, then measured nodes)
    brief_nodes: int = 20                       # ... of which at most this many measured closed nodes (newest first)
    recent_max: int = 30                        # in-flight / recently-proposed lines a session start lists
    keep_rejects: int = 400                     # bounced slate candidates remembered (oldest dropped)
    keep_proposed: int = 300                    # slate candidates remembered as proposals (oldest dropped)


@dataclass(frozen=True)
class KnowledgeCfg:
    """PROJECT KNOWLEDGE (loop.knowledge, 11-08): operator-curated facts every brief carries (the per-task picture, the
    gap to the reference model, what the noise allows), from markdown files -- not hardcoded in prompts. ``files``
    (relative to the repo or absolute); empty = ``<domain prompt_dir>/../knowledge.md`` when it exists. A section
    ``## Tested mechanisms`` is not briefed verbatim: its ``- local:<run>: <what it changed>`` lines describe the
    manual runs for the mechanism memory (loop.core.tested). The bar line (each axis must move >= ``bar_sd`` paired
    sd) is computed from ``loop.axis_sd``."""
    enabled: bool = True
    files: tuple[str, ...] = ()
    bar_sd: float = 2.0
    max_chars: int = 6000


@dataclass(frozen=True)
class RootAuditCfg:
    """ROOT ASSUMPTION LEDGER (loop.root_audit): when the root / base changes (or the board digest changes materially)
    one review-tier LLM call enumerates every baked-in choice of the root recipe (config + hard-coded constants), joins
    it with the board digest, and emits ranked audit arms (config-only ablate / sweep nodes tagged ``root_audit``) that
    enter the queue; the ledger is shown to every v3 brief. Default off."""
    enabled: bool = False
    max_arms: int = 6                           # arms queued per audit
    max_open: int = 6                           # root-audit nodes QUEUED / running at once
    constants_max: int = 120                    # hard-coded constants listed from the root's source
    source_globs: tuple[str, ...] = ()          # root files scanned for constants ('' = the writable surface's .py)
    rerun_on_board: bool = True


@dataclass(frozen=True)
class IdeationCfg:
    """FIRST-PRINCIPLES IDEATION (loop.agents.ideas / dossiers, loop.variants; operator 2026-10-06). Generic machinery,
    no named mechanism: a research session states a mechanistic model, the binding constraint, the derived change,
    the measured quantity it should move and why earlier attempts failed; proposes several DISTINCT ideas as plans
    gated by the intake rules BEFORE any code; the loop ranks them (allocation signal, under-exploration, novelty,
    the share guard), implements the best and banks the rest; each idea carries config variants and a branched
    follow-up plan the loop runs itself. ``base_share`` of the research draws attack ONE base component (discovered
    from the code / config / history) with its auto-built dossier."""
    enabled: bool = True
    first_principles: bool = True               # research-type sessions must fill the first-principles record
    min_ideas: int = 3                          # distinct ideas (plans) a research session proposes ...
    max_ideas: int = 5                          # ... at most
    implement_top: int = 2                      # the best is built in the session; ranks 2..this are banked PRIORITY
    base_share: float = 0.0                     # research draws that become a challenge-the-base brief
    base_ladder_p: float = 0.3                  # a base-challenge code change ladders at this shrunk P(own gain > 0)
    share_window: int = 40                      # recent research nodes the share guard counts
    max_share: float = 0.35                     # a theme above this share of them needs the allocation signal
    max_variants: int = 4                       # config variants per idea, screened beside it on its block
    max_points: int = 8                         # variant points per idea incl. every branch round
    branch_depth: int = 2                       # follow-up rounds of the branched plan
    null_q: float = 0.9                         # best variant's raw delta must beat this quantile of the null max
    # v3 briefs (loop.core.ideation; meta-analysis 10-13: 43/58 sessions claimed exactly the MDE the brief asked for)
    claim_gap_frac: float = 0.5                 # target mode: ask for a claimed gain >= this x the gap (no MDE anchor)
    program_share: float = 0.25                 # target mode: share of explore-kind sessions briefed as a PROGRAM
    program_share_base: float = 0.0             # ... outside target mode
    program_wave_min_z: float = 1.0             # a program node's dose ladder while its belief > -z sd
    program_wave_max_points: int = 5            # trials of that ladder incl. the node itself (moonshot budget <= 5)
    levers_k: int = 12                          # rows per list of the computed "untested levers & deficits" section
    deficit_min_seeds: int = 4                  # deficits vs the best other node: nodes with >= this many seeds
    deficit_z: float = 1.0                      # a deficit / lead is shrunk by this many se toward 0
    external_csv: str = ""                      # external FM table (relative to spec.repo); '' = none
    external_model_col: str = "model"
    external_cols: Mapping[str, str] = field(default_factory=lambda: {
        "cls": "v2_classification", "seg": "v2_segmentation", "panel": "v2_final"})
    external_exclude: tuple[str, ...] = ("genbio",)   # rows matching (case-insensitive) are never shown
    external_peer: str = r"-S/|ViT-S|small"     # regex: a size-matched public peer (shown beside the best)
    # axis -> the base config's (flattened) key listing its probe tasks, named in the deficits section
    # ({} = none named; nanopath's probe.* lists are pinned in tasks/nanopath_v2.yaml)
    axis_tasks: Mapping[str, str] = field(default_factory=dict)
    external_group: str = "baseline"            # external table rows of this group only ('' = every row)
    refute_k: float = 2.0                       # a mechanism term is refuted at pooled own gain <= -k se ...
    refute_min_nodes: int = 3                   # ... over at least this many measured nodes
    refute_max_term_share: float = 0.05         # tag terms in more than this share of nodes are too generic
    # generic tag words that never name a mechanism (a refuted term must also be a whole tag, or the exact tag of
    # >= 2 of its nodes)
    refute_stopwords: tuple[str, ...] = (
        "only", "without", "with", "plus", "and", "from", "into", "over", "under", "weight", "tuned", "tune",
        "stack", "stacked", "strong", "stronger", "mild", "milder", "lower", "higher", "late", "early", "full", "soft",
        "softer", "small", "large", "bigger", "smaller", "extrap", "based", "fixed", "variant", "half", "double",
        "more", "less", "step", "ramp", "warm", "cool", "base", "incumbent", "port", "redo", "retry", "fix")
    retest_min_seeds: int = 4                   # closed nodes with a positive own gain over >= this many seeds
    precheck_recent: int = 8                    # the latest idea mechanisms pre-checked against intake per brief
    fnf_screen_credit: bool = False             # v3-c7i: branches are credited on fresh-seed TEST gains only (True: untested
                                                # branch nodes also earn shrunk screen gain)
    fnf_screen_min_seeds: int = 2               # ... once they have this many screen seeds
    fnf_pause_max_d: float = 3.0                # a fork-and-flush pause grants ONE round after this many days (0 = never):
                                                # branch nodes are never tested, so the test-only pause could not lift
    # GPU-idle scaling (10-13 review): ideas are the GPU-consuming bottleneck, so while more than llm_scale_free_slots
    # GPU slots would otherwise idle (free, or held by idle-fill / relax runs: no decision value; runner.RunBook.idle)
    # the research sessions may run up to llm_slots_max at once (researcher.max_concurrent otherwise). v3-c5: 12 -> 2
    # (live 10-13: free slots > 12 on 2 of 128 ticks -- fill held them -- so ~4.7 sessions ran under a cap of 8)
    llm_slots_max: int = 8
    llm_scale_free_slots: int = 2
    # v3-c4i (09Z deep MA, 10-13): per-axis claim budget, staged programs, the researcher's diagnose tool, computed
    # compute / per-dataset / repair-candidate levers, spread concurrent sessions, bounded web, the session clock
    claim_bounces: int = 1                      # finish bounces for a budget total < claim_gap_frac x gap (then logged)
    program_min_components: int = 2             # a program's finish declares at least this many components
    program_loo: bool = True                    # leave-one-out of a program's components once it screens positive ...
    program_loo_min_z: float = 1.0              # ... its belief above this many sd
    program_loo_max: int = 3                    # ... at most this many LOO children (trials <= wave_max_points + it)
    # v3-c8a B1 (10-14: both operator gate arms only weakened the gate and lost slide 5-7x their cls gain; the
    # stronger direction was never tried): ladders put doses on BOTH sides of the node's value, and an eval-only arm
    # gets its opposite arm (declared ``opposite``, else mirrored numeric literals) beside it by default -- else once
    # the arm is measured harmful; a harmful one-sided ladder dose gets its mirrored dose once (loop.core.program)
    program_ladder_symmetric: bool = True
    evalonly_both_directions: bool = True
    # v3-c8a B4: a program ladder isolates every changed config key it does not dose (one leave-that-key-out child
    # each, within program_loo_max)
    program_iso: bool = True
    # v3-c8a B2 (14/14 databank datasets lost to the incumbent; mpp050_448 ran 3x): a data.dataset_dir (or
    # dataset_spec) already measured on >= this many paired seeds below its parent in some node is DEAD -- a new
    # node on it is rejected ``dead_dataset`` unless its claim's [differs from: ...] names and explains each of those
    # nodes (0 = off)
    dead_dataset_min_seeds: int = 3
    repair_k: float = 2.0                       # repair candidates: a node's axis gain >= this many se ...
    repair_loss_k: float = 1.0                  # ... with another axis lost by >= this many se (or composite <= 0)
    repair_min_seeds: int = 3                   # ... over at least this many seeds
    compute_keys: str = r"(views|_size$|crop_scale|resolution|patch_size)"   # config keys setting per-tile compute
    external_task_csv: str = ""                 # per-dataset / per-head external table (THUNDER results.csv); '' = none
    external_task_models: str = "base-control"  # regex: the table's rows that are the off-the-shelf public models
    external_task_metric: str = "f1"
    external_task_scale: float = 0.01           # table scores are percent
    external_task_heads: Mapping[str, str] = field(default_factory=lambda: {
        "linear": "linear_probing", "knn": "knn", "fewshot": "simple_shot"})
    external_task_names: Mapping[str, str] = field(default_factory=lambda: {"pcam": "patch_camelyon"})
    task_metric_key: str = "final_probe_{dataset}_{head}_val_f1"   # a run summary's per-dataset / per-head score
    diagnose_tiles: int = 32                    # feature diagnostic: tiles per classification dataset
    diagnose_job_min: int = 15                  # its SLURM time limit (minutes, primary pool)
    diagnose_wait_s: float = 600.0              # the tool blocks at most this long for the job, then answers pending
    diagnose_precompute: bool = False           # v3-c5: submit the incumbent's feature job once per incumbent (no wait);
                                                # nanopath's feature_diag job (a ViT probe_features tool): opt-in per task
    databank_list_max: int = 60                 # datasets / files the diagnose tool lists
    web_kinds: tuple[str, ...] = ("repair", "crossover")   # v3 brief kinds that also get (bounded) web research
    web_max_searches: int = 3                   # ... at most this many searches (prompted; counted on the node)
    # EVIDENCE GATE (loop.agents.researcher): a structural session's finish must carry hypothesis.evidence (citation
    # URL / arXiv / doi, a board run id, or a board reference-model fact) and hypothesis.readout_path (how the change
    # moves what the frozen-feature eval reads); bounced evidence_bounces times, then logged and its claim parked in
    # the evidence backlog the next literature / scout brief researches. Those kinds always get web tools
    require_evidence: bool = False
    evidence_kinds: tuple[str, ...] = ("bold", "kuhn", "program")
    evidence_bounces: int = 2
    # v3 briefs serve the operator idea banks (loop.select.bank; loop.idea_banks + the v3 scout's bank) as the head of
    # explore / literature briefs, each idea once (the v2 driver did; v3 never read them). Default off
    bank_v3: bool = False
    # TRAINING-SIDE ONLY (operator 2026-11-08, nanobrain): brief kinds that tune the eval-time readout are not drawn
    # and every brief says eval-side observations may only motivate training changes
    training_only: bool = False
    slow_sessions_n: int = 6                    # with >= this many sessions in flight at start ...
    timeout_scale: float = 1.5                  # ... a session's time limit is x this
    smoke_off_clock: bool = True                # the smoke / diagnose tools' wall time does not count against the limit
    levers_scan_s: float = 20.0                 # levers: the summary.json scan's time box (disk-cached by mtime; 0 = none)
    # R4: research briefs render on this many worker threads off the tick (the tick enqueues; the next tick starts the
    # rendered sessions); 0 = render + start synchronously in the tick (shadow / dry-run always does)
    brief_workers: int = 2
    brief_ttl_s: float = 1800.0                 # v3-c5b S6: a rendered brief older than this (or of an older view) is re-planned
    # A3: an explore-lane slot's brief kind by its measured yield this epoch (mean own posterior of the kind's nodes,
    # + their untagged descendants when kind_yield_descendants), shrunk to the all-kind mean with kind_yield_n0
    # pseudo-nodes; each eligible kind keeps >= kind_weight_floor of the share. False = the fixed rotation
    kind_weight: bool = True
    kind_yield_n0: float = 5.0
    kind_weight_floor: float = 0.08
    kind_yield_descendants: bool = True
    # v3-c5 (research-session tools): a finish whose per-axis budget (official weights x gains) differs from its panel
    # claim by more than this RELATIVE share is bounced claim_mismatch_bounces times with the numbers, then logged
    claim_budget_tol: float = 0.2
    claim_mismatch_bounces: int = 1
    # program / moonshot briefs: structural levers able to move the panel >= structural_gain in one shot, and the
    # measured tail -- the share of this epoch's screened incremental nodes (>= tail_min_seeds) with own gain >= tail_gain
    structural_gain: float = 0.008
    tail_gain: float = 0.004
    tail_min_seeds: int = 1
    # v3-c6i (10-13 literature scan + ideation analysis; loop.core.calibration / mechanisms / prescreen):
    # verbalized-sampling finish (arXiv 2510.01171): k candidate mechanisms, each with a predicted effect
    # distribution {p_neg, median, p_gap}, one chosen; the chosen one's prediction is scored (log score of the 3-bin
    # outcome under the node's own-increment posterior) once its node has calibration_min_seeds screen seeds, and the
    # per-kind skill over the base rates (shrunk with calibration_n0 pseudo-sessions) scales its kind weight by
    # exp(calibration_weight x skill)
    k_candidates: int = 4
    vs_bounces: int = 1
    calibration_min_seeds: int = 3              # v3-c6r: a 2-seed screen mean is too noisy to score a forecast on
    calibration_n0: float = 5.0
    calibration_weight: float = 0.5
    # mechanism-level dedup (DAPS / ShinkaEvolve): the signature family + tag + one-line mechanism (ancestors' own text
    # removed) vs the last mech_dup_window signatures (ancestors excluded); cos > mech_dup_cos rejects near_dup_mechanism
    mech_dup_cos: float = 0.85
    mech_dup_window: int = 200
    # v3-c6r: the top-3 lexical neighbours above this cosine (and below mech_dup_cos) get a qwen yes/no "same
    # mechanism?" judge (paraphrases the hashing embedder misses; a dose variant counts as the same); 0 = off
    mech_judge_cos: float = 0.5
    # mechanism taxonomy: category -> regex over "family mechanism_type tag" (first match wins, else 'other'); briefs
    # suggest the category furthest below its share of this epoch's research nodes (+ in-flight claims)
    taxonomy: Mapping[str, str] = field(default_factory=lambda: {
        "readout_eval": r"eval_readout|readout|probe|pool|gate|typical",
        "data_curation": r"data_curation|data_sampling|dataset|curat|sampl|balanc|filter|stain",
        "compute_resolution": r"resolution|hires|_size\b|patch_size|global_crop|local_crop|n_local|views|schedule_clock",
        "augmentation": r"augment|crop|jitter|flip|blur|mix|solari|color",
        "representation_geometry": r"whiten|center|isotrop|uniform|spectr|rank|geometr|collapse|orthog|norm",
        "objective": r"objective|loss|jepa|ibot|dino|koleo|distill|contrast|target|teacher",
        "regularization": r"regulari|dropout|drop_path|weight_decay|smooth|noise",
        "optimization": r"optimi|\blr\b|lr_|warmup|schedul|decay|momentum|ema|batch|clip|training",
        "architecture": r"architect|register|adapter|head|attention|block|layer|token|embed"})
    # per-session discipline lens (round-robin; diversity-collapse lit 2604.18005 / 2608.02618); these kinds see no
    # recent / in-flight idea pool (a terse 'taken' list only)
    personas: tuple[str, ...] = ("signal processing", "optimization theory", "data curation",
                                 "representation geometry", "pathology domain", "systems / compute")
    hidden_pool_kinds: tuple[str, ...] = ("explore", "program", "kuhn")
    # clade credit (HGM 2510.21614): a brief-tagged node's kind credit = its best descendant (>= clade_min_seeds)
    clade_credit: bool = True
    clade_min_seeds: int = 2
    # triggered diagnostic summary at the top of every brief (AIDE^2): datasets whose per-tile gate mean / head-cls
    # cosine is below diag_ratio x the reference (readout_reference, else the cross-dataset median), and the
    # diag_summary_k largest per-dataset public gaps
    diag_summary: bool = True
    diag_ratio: float = 0.5
    diag_summary_k: int = 3
    readout_reference: str = ""
    # the readout kind: eligible while the summary flags a distorted dataset; readout_max_sessions per incumbent
    readout_max_sessions: int = 2
    # pairwise pre-screen (REHEARSE / FOREAGENT): one short LLM call per new node vs its pairwise_k nearest screened
    # nodes; the predicted rank nudges its screen value by pairwise_weight x the prior sd (0 = off) while the measured
    # pairwise accuracy is >= pairwise_min_acc over >= pairwise_min_scored nodes (below: weight 0, logged)
    pairwise_weight: float = 0.1
    pairwise_k: int = 5
    pairwise_min_scored: int = 8
    pairwise_min_acc: float = 0.55
    # 11-09: centre the predicted rank s on the book's median (>= prescreen.CENTER_MIN predictions) before the nudge, so
    # an unscored node does not outrank every scored one when s is biased. GENERIC DEFAULT ON (a bias fix, 11-10)
    pairwise_center: bool = True
    # operator-seeded programs: [{id, brief, components, stages, budget_gpu_h, harm_guards}], one program session each
    operator_programs: tuple[Mapping[str, Any], ...] = ()
    # OPERATOR FINDINGS (10-21): the operator's MANUAL experiments as measured facts, one line each (what was run, the
    # paired result, DEAD / NULL / OPEN). Rendered at the top of every research brief and big review as facts that
    # outrank any past session's prose (loop.agents.briefs.findings_block).
    operator_findings: tuple[str, ...] = ()
    # v3-c8b BOLD STRUCTURAL LANE (loop.core.bold; operator 2026-10-14 "ideas too boring / not unique"): the ``bold``
    # brief kind asks for ONE structural change (pretext / data / architecture / schedule / context / pathology), no
    # dose knob, no per-axis budget rows, no lever / diagnostic anchoring; step-0 identity relaxed (never the banned
    # inits / caps / eval identity / locked files). Its share of the explore-lane sessions: bold_share blended with
    # the kind bandit (bold_n0 pseudo-sessions), never below bold_floor while it has < bold_floor_n nodes.
    bold_share: float = 0.3
    bold_floor: float = 0.2
    bold_floor_n: int = 20
    bold_n0: float = 5.0
    bold_tune_max: int = 5                      # tuning trials over a bold node's whole tuned family (loop.tune)
    # STRUCTURAL TUNING LADDER (loop.core.program.struct_ladders; operator 2026-10-22: backbone swaps a node / a node were
    # each judged on ~one untuned shot): a node that changes the backbone (model.type / pretrained init), adds a loss
    # term, or is a bold / moonshot structural change gets <= struct_tune_max tuning trials (excl. the node itself)
    # BEFORE it can count as negative. Knob design per kind: ``struct_tune_knobs`` (yaml). A child's test entry is
    # max-of-K shrunk with at least struct_tune_beta (selection correction; the fresh-seed TEST is the judge).
    struct_tune: bool = True
    struct_tune_max: int = 5
    struct_tune_min_z: float = 2.5              # a non-backbone node is laddered while its belief > -z sd (backbone: always)
    # 11-09 operator moonshot-tuning rule: when set, a non-backbone structural node with no engagement diagnostic is
    # laddered while its belief is ABOVE this absolute floor (e.g. -.005: flat-but-not-dead), whatever its sd; at or
    # below it, it is closed untuned. A node skipped earlier by the z rule is re-checked. None = the z rule (nanopath)
    struct_tune_flat_floor: float | None = None
    struct_tune_beta: float = 1.0
    struct_retro: bool = True                   # old-epoch backbone swaps are ported onto the base for a ladder (retro)
    struct_retro_max: int = 4                   # retro ports in flight (ported, ladder not finished) at once
    # 10-22 ops review: tick 0 booked ~19 ladders (73 children) at program-child priority at once. Only this many
    # laddered parents may have an UNFINISHED ladder at once; the rest wait (arch arms, then retro ports, then the
    # others by posterior) without their hold clock running and are not closed on their untuned shot meanwhile
    struct_ladder_max_open: int = 6
    # OPERATOR list: closed nodes to port onto the base and ladder (retro). An entry is a node id, or a per-node ladder
    # spec {node, knobs: {config key: [values]}} (one config-only child per listed value, round-robin over the keys)
    struct_retro_nodes: tuple[Any, ...] = ()
    struct_min_seeds: int = 3                   # a tuned point counts toward the close rule at this many finished seeds
    struct_hold_h: float = 72.0                 # a stuck ladder releases its node this many hours after it was built
    struct_min_points: int = 3                  # a structural node is not closed negative before this many tuned points
    struct_cata_z: float = 3.0                  # ... unless its first point is worse than -z sd ...
    struct_cata_seeds: int = 3                  # ... on this many seeds with the mechanism engaged
    struct_tune_knobs: Mapping[str, Any] = field(default_factory=dict)
    # MOONSHOT RESCUE (loop.core.rescue; operator 11-10: bold moonshots were judged on 1 untuned seed, no bug check).
    # A bold / structural node with <= moonshot_rescue_max_seeds paired seeds and no run in flight: a CATASTROPHIC
    # first screen (paired mean < -moonshot_cata_k x panel_sd, or < -moonshot_cata_abs) or an all-crashed one gets ONE
    # researcher DEBUG session (a fixed child, or "no bug found"); a not-clearly-negative one (mean >
    # -moonshot_rescue_neg_se x se) is ported onto the CURRENT incumbent and gets <= moonshot_rescue_rungs tuning
    # configs over its declared knobs, screened moonshot_rescue_seeds seeds each (port + rungs <= 5 trials); the best
    # then takes the normal path. At most moonshot_rescue_max rescues / debugs at once (by prior plausibility, never the
    # 1-seed delta), and no new one while rescue runs hold >= moonshot_rescue_gpu_share of the capacity
    moonshot_rescue: bool = True
    moonshot_rescue_max: int = 3
    moonshot_rescue_max_seeds: int = 2
    moonshot_rescue_neg_se: float = 2.0
    moonshot_rescue_rungs: int = 4
    moonshot_rescue_seeds: int = 2
    moonshot_rescue_gpu_share: float = 0.15
    moonshot_rescue_hold_h: float = 96.0        # a rescue undecided this long after its port EXISTS is decided on what it
    #                                             has (a port session pending this long after booking is closed)
    moonshot_rescue_stall_h: float = 12.0       # a candidate's last run was submitted >= this long ago (a STALLED screen)
    moonshot_rescue_llm_slots: int = 1          # researcher sessions (rescue ports + debugs) in flight at once ...
    moonshot_rescue_llm_reserve: int = 2        # ... started only while more than this many researcher slots are free
    moonshot_debug: bool = True
    moonshot_debug_max_open: int = 2            # debug sessions pending / in flight (researcher slots, no GPU)
    moonshot_cata_k: float = 4.0
    moonshot_cata_abs: float = 0.02
    # a knob-judge "knob only" verdict is overridden when the diff adds CODE (a new def / class / module / loss term,
    # or >= bold_knob_code_min_other other new non-constant items the PARENT's source never had): a node added 3 defs
    bold_knob_code_override: bool = True
    bold_knob_code_min_other: int = 3
    bold_knob_judge: bool = True                # the qwen yes/no knob-only judge at intake (besides the diff rule)
    bold_category_rx: Mapping[str, str] = field(default_factory=dict)   # {} = loop.core.bold.CATEGORY_RX
    # DIVERGE-THEN-JUDGE: a bold session (and explore / literature while diverge_explore) must call propose_slate
    # with slate_min..slate_max candidates over >= slate_min_categories categories (<= slate_per_category each)
    # before any edit; scored originality x plausibility, saturated-negative clusters rejected, the top one built
    diverge_explore: bool = True
    slate_min: int = 8
    slate_max: int = 12
    slate_min_categories: int = 4
    slate_per_category: int = 3
    # v3-c9 C3 (RSI-top "Escaping Exploration Collapse": verbalized-sampling portfolio + grounding in distinct history
    # branches): explore / literature (and repair) sessions propose their directions as ONE joint portfolio of
    # portfolio_min..portfolio_max (a repair: portfolio_repair_min..max) candidates, each with a verbalized
    # `probability` (that a typical researcher would propose it: the portfolio's sum is within portfolio_sum_tol of 1,
    # >= portfolio_tail_min of them at or under portfolio_tail_p) and a `grounded_in` history cluster / node id; the
    # candidates cover >= portfolio_min_clusters distinct history clusters (a repair: portfolio_repair_clusters); the
    # chosen direction cites >= 1 history node / cluster it differs from (hypothesis.differs_from: ids that exist in
    # the history table + >= differs_min_chars content characters of explanation). portfolio=False: the old slate.
    portfolio: bool = True
    portfolio_repair: bool = False              # review fix S3: explore / literature first; the repair portfolio is opt-in
    portfolio_min: int = 6
    portfolio_max: int = 8
    portfolio_repair_min: int = 3
    portfolio_repair_max: int = 5
    portfolio_min_clusters: int = 3
    portfolio_repair_clusters: int = 2
    portfolio_sum_tol: float = 0.35
    portfolio_cite_min: int = 1
    portfolio_typical_w: float = 0.5
    portfolio_tail_p: float = 0.15
    portfolio_tail_min: int = 2
    differs_min_chars: int = 12
    differs_sim_min: float = 0.3                # a cited node must share the chosen candidate's cluster / family / tag, or
                                                # have text similarity above this with it (review fix M3)
    history_calls_max: int = 6                  # review fix S3: `history` calls per session (stated in the prompt)
    # v3-c9 C1 (RSI-top trajectory dashboard): the researcher's read-only `history` tool (a per-node table of every
    # trajectory: own-increment posterior, per-axis deltas on the VISIBLE ledger, verdict, failure signature, novelty,
    # cluster) -- `history_view_frac` of the non-key columns are shown per session (seeded; key columns always),
    # `history_rows_max` rows per table call, `history_open_chars` characters per `open`
    history: bool = True
    history_view_frac: float = 0.6
    history_rows_max: int = 40
    history_open_chars: int = 6000
    # v3-c9 C2 (non-incumbent starting points): `alt_base_share` of the explore / bold sessions start from an ALTERNATE
    # base instead of the incumbent -- the best node (by shrunk own gain) of a non-saturated cluster outside the
    # incumbent's lineage, the previous epoch's best (ported onto the current base first), or the top posterior node
    # outside the incumbent's ancestry -- with >= alt_base_min_seeds paired seeds and an own gain not clearly negative
    # (m >= -alt_base_z x sd). alt_base_fnf: of a fork-and-flush round's K branches this many start from alt bases
    alt_base_share: float = 0.2
    alt_base_kinds: tuple[str, ...] = ("explore", "bold")
    alt_base_max_uses: int = 3
    alt_base_min_seeds: int = 3
    alt_base_z: float = 1.0
    alt_base_fnf: int = 1
    # review fixes M1 / M2: a candidate must be a DIFFERENT code branch -- its parent is neither the incumbent nor a
    # non-root ancestor of it (second parents and ports included), the diff-line Jaccard distance of its cumulative diff
    # from the incumbent's is >= alt_base_min_dist (similarity < 0.9), and no more than alt_base_max_contain of its own
    # added diff lines already appear in the incumbent's cumulative diff; a gain vs the incumbent below
    # -alt_base_gain_z x se (paired composite) is skipped. alt_base_require_ckpt: its commit object and a checkpoint exist
    alt_base_min_dist: float = 0.1
    alt_base_max_contain: float = 0.8
    alt_base_gain_z: float = 1.28
    alt_base_require_ckpt: bool = True
    # the fourth source, `root`: the current epoch's root base node (the plain base, without the incumbent's stack). Exempt
    # from the ancestry / containment rules (far by construction: alt_dist=far, alt_src=root), still sound + buildable;
    # its own runs are the paired control. alt_base_root_share: the share of alt draws that use root while other sources
    # have a pick (root gets ALL alt draws when they are empty). A root child with a positive own increment (m > 0,
    # m > 1 se, >= alt_base_min_seeds paired seeds) is auto-ported onto the incumbent (<= 1 open, port_src=alt_root)
    alt_base_root: bool = True
    alt_base_root_share: float = 0.5


@dataclass(frozen=True)
class HeartbeatCfg:
    poll_s: float = 30.0
    freeze_min: float = 20.0                    # a started job with no progress this long is frozen
    log_every_min: float = 10.0                 # one `hb` line in the log this often (0 = off)
    # time-box: past this many seconds into a tick, wave builds pause (journalled, resumed next tick) and no more runs
    # are submitted (re-ranked next tick); 0 = off (live 10-13: 457 / 950 / 1177 s ticks, mostly serial wave builds)
    tick_budget_s: float = 300.0
    # ... but never starved: every tick still builds >= min_wave_points wave points (a new wave may start for them)
    # and makes >= min_submits submits; a paused wave build with 0 new points this many ticks running is quarantined
    min_wave_points: int = 1
    min_submits: int = 8
    wave_pause_quarantine: int = 6


@dataclass(frozen=True)
class DashboardCfg:
    every_ticks: int = 1                        # write <state_dir>/dashboard/index.html this often (0 = off)
    # ... but never more often than this many seconds: the page takes ~50 s of CPU (live 10-22: 26 MB), so the tick
    # renders it in a NICED child process (``loop dashboard``), one at a time
    min_interval_s: float = 900.0


@dataclass(frozen=True)
class PoolCfg:
    """One SLURM association (env SBATCH_ACCOUNT/QOS/PARTITION beats the script's #SBATCH lines;
    empty = the script's own). Slots fill in declared order; a preemptible pool's jobs get
    SBATCH_REQUEUE=1."""
    name: str
    account: str = ""
    qos: str = ""
    partition: str = ""
    slots: int = 16
    preemptible: bool = False


@dataclass(frozen=True)
class SlurmCfg:
    pools: tuple[PoolCfg, ...] = ()             # () -> one pool of loop.slots, the script's association
    time_limit: str = ""                        # SBATCH_TIMELIMIT; a TIMEOUT is a too_slow verdict
    preemptible_slots: int = -1                 # >= 0 (self-tune): slots of every preemptible pool; -1 = as declared
    # the operator's shared jobs (squeue names): while one is PENDING the loop submits no NEW preemptible job --
    # only while it could plausibly start soon: pending for one of protect_reasons (not ReqNodeNotAvail, a
    # reservation, ...) and requeued fewer than protect_max_restarts times (0 = no limit); else backing off only
    # idles our slots (2026-09-30: ~70 slot-h / 24 h for a qwen job at ReqNodeNotAvail with 55 restarts)
    protect_jobs: tuple[str, ...] = ()
    protect_reasons: tuple[str, ...] = ("Priority", "Resources")
    protect_max_restarts: int = 20
    # SLURM nice for a LOW run on a preemptible pool / for every other run; 0 = off (the default). Nice lowers a
    # job's priority against EVERY user's jobs, not only ours (live 2026-10-02: run_nice 10 put our normal-QoS jobs
    # at 390 behind ~57 other users' jobs at 400; 0 jobs ran for hours). Smokes go first among our own jobs by
    # max_pending_per_pool instead
    low_nice: int = 0
    run_nice: int = 0
    # 10-22: a smoke queued 1 h behind 28 of our own pending screens (same QoS = submit order). NOT fixed with Nice (10-02
    # incident above): while one of our smokes is PENDING on a pool no new run is submitted there (RunBook.limits), and
    # max_pending_per_pool bounds the screens ahead of a new smoke
    # a smoke pending this long (s) closes new research sessions (``smoke_capacity``): a session started now would wait
    # behind it. After a smoke queue timeout with no smoke started since, ONE session at a time probes for ``smoke_probe_s``
    smoke_starve_s: int = 1500
    smoke_probe_s: int = 10800
    # once this many of our training runs are PENDING on a pool (at the last poll), submit there at most this many
    # beyond what SLURM runs; the rest wait unsubmitted in the loop's own queue, so a new smoke has at most ~this many
    # of our jobs ahead of it (live 2026-10-02: 15 smokes PENDING behind 23 of our own screens). A pool with fewer
    # PENDING fills to its slots (live 2026-10-05: running + 2 on every pool left 29 GPUs idle for hours).
    # Already-pending jobs are never cancelled; 0 = off
    max_pending_per_pool: int = 2
    # NEW submits per pool per tick are capped to max(2, the GPUs the pool's partition can start now) (adapter.free_gpus:
    # sinfo, read-only); a probe failure falls back to this constant (live 10-22 10:39Z: 50 screens in one tick to an empty
    # idle pool declared 56 with ~3 GPUs free, 43 sat PENDING). free_gpu_probe False = no probe (the gate stands as before)
    max_new_per_tick: int = 4
    free_gpu_probe: bool = True
    # a HIGH run SLURM requeued this many times while pending on a preemptible pool moves to primary even when
    # primary is full (it queues there next)
    repool_restarts: int = 2
    # ANY run (LOW too: screens, design points, tuning controls) SLURM requeued this many times moves to primary
    # the same way (live: a node:screen:377, a LOW design point, 37 restarts / 6.9 GPU-h lost); 0 = never
    repool_restarts_any: int = 5
    # resume-on-requeue (loop.resume): a preempted run continues from its own latest.pt, not step 0 (live: 39 GPU-h
    # lost vs 71 finished in 6 h). "on" (operator 2026-10-01: known NOT exact -- the stats give a resumed run's
    # paired deltas seed-level noise, loop.world.noise.resume_var); "off"; "calibrate" = on only once a one-time
    # interrupted-and-resumed root run lands within resume_tol of the uninterrupted one (same seed)
    resume: str = "on"
    resume_save_every: int = 500                # train.save_every of resumable runs (~2 min; a 1.35 GB save ~3 s)
    resume_tol: float = 0.001                   # |final_score delta| the calibration accepts (~2x replicate sd)
    # slots of the first preemptible pool held for smoke jobs (researchers' and design smokes): runs fill to
    # slots - max(reserve, smokes in flight), so a queued smoke takes the next freed slot, never waits behind our runs
    smoke_reserve: int = 1
    # SCARCE MODE (loop.driver.RunBook.scarce): while our running GPUs averaged over the last 2 h are below this, or a
    # preemptible pool is STARVED, design batches and tuning passes pause (no new batch / pass / trial; in-flight
    # trials finish) and new screens go to at most 2 carriers (live 2026-10-02: ~2.6 GPUs effective, 20% of them on
    # tune / design, 0 on confirm); 0 = off (the default; nanopath_v2 sets 10)
    scarce_gpus: float = 0.0
    # a TEST-stage run (loop.tick.TEST_STAGES: CONFIRM / REMINT on fresh test seeds) waits for a primary slot instead
    # of spilling onto a preemptible pool: it is never resumed from latest.pt, so each preemption restarts it from
    # step 0 (live 10-13: 20 preemptions / 10.2 GPU-h lost in 4 h). Ignored when no non-preemptible pool exists
    test_primary_only: bool = True
    # ... but a TEST-stage run that has waited this long for a primary slot may take a preemptible one (resumed from
    # its latest.pt after a preemption); 0 = never
    test_primary_wait_s: float = 3 * 3600.0
    # ... and a TEST-stage run with no free primary slot while the primary pool is CUT by the user's other jobs (effective
    # capacity: the operator's sweeps hold the association's GrpTRES budget and refill any freed slot) spills to a
    # preemptible pool at once instead of waiting test_primary_wait_s (10-21); False = always wait
    test_spill_capped: bool = True
    # an idle-fill / relax-pass run (fill_only) takes a free primary slot (never one reserved for an owed TEST, never while a
    # HIGH run waits) when no preemptible pool has room: live 10-22 12:13-13:23Z the idle pool held 41 of our runs PENDING
    # (max_pending_per_pool gate) so 2 effective primary slots sat empty for 70 min; False = fill is preemptible-only
    fill_on_primary: bool = True
    # PROBE SMOKE (loop.adapters.probe_smoke): after the training smoke, the same SLURM job drives the smoke checkpoint
    # through the locked probe.py's readout entry points on 64 real tiles per probe dataset (~1-2 min, +10 min time
    # limit), so a readout that crashes in probe.py fails the smoke, not a full run (live a node: linalg.eigh in
    # probe_features after 2.4 GPU-h). "readout" = only when the diff touches model.py / an eval_readout span / a
    # probe.|model.|data.mean|data.std key; "always"; "off"
    smoke_probe: str = "off"                    # nanopath_v2.yaml: "readout" (the probe leg drives nanopath's probe.py)
    # the probe-facing model.py functions whose edit triggers the probe leg under "readout" (empty = the adapter's
    # PROBE_FUNCS default: encode_image, probe_features, ...)
    smoke_probe_funcs: tuple[str, ...] = ()
    # a preemptible pool holding more than this many of our runs PENDING is skipped by smokes (and the smoke reserve
    # moves to primary): a smoke queued there waits behind our own runs (live 10-13); scontrol top is not user-enabled.
    # v3-c5: 8 -> 0 (ANY pending run of ours blocks the pool; 8 never triggered at the live 6-8 pending), primary
    # still capped by smoke_primary_max
    smoke_skip_pending: int = 0
    # at most this many smokes in flight on the primary pool; past it a smoke queues on a preemptible one; 0 = no cap
    smoke_primary_max: int = 2
    # where smokes queue: "auto" (the rules above: a free slot, preemptible first, ...) or "primary": every smoke
    # queues on the primary (non-preemptible) pool up to smoke_primary_max (past it the "auto" preemptible fallback),
    # and while one is PENDING there no new run is submitted to primary (RunBook.limits), so the next primary slot a
    # run frees is the smoke's -- screens fill to capacity - smokes. 11-08 nanobrain: with primary full of 2 h screens
    # the "auto" rule sent every smoke to the contended idle pool, where it sat PENDING behind other users' jobs for
    # the whole SMOKE_WAIT_S (queue timeout -> no verdict, a re-issued Opus session) while research idled on smokecap
    smoke_pool: str = "auto"
    # with smoke_pool "primary": the smoke_reserve slot(s) stay held on primary even while no smoke is in flight (a smoke
    # then starts at once instead of waiting for a 2 h screen to end; costs that idle GPU between smokes). False = the
    # reserve moves to the preemptible pool whenever no smoke is in flight (the c4j rule)
    smoke_primary_hold: bool = False
    # queue wait (s) of a smoke before it is scancelled as no-slot (``smoke_unavailable``, no verdict, never a failure or
    # an infra strike); 0 = the adapter's SMOKE_WAIT_S (3600). smoke_session_wait_s: the same for the smoke an LLM
    # session runs as a TOOL (the Opus session is held open the whole wait); past it the tool returns "not run, finish"
    # and the loop re-smokes after the session ends, with no LLM held; 0 = smoke_wait_s
    smoke_wait_s: float = 0.0
    smoke_session_wait_s: float = 0.0
    # adapter submits (node worktree + sbatch) of one tick run on up to this many threads (the store writes stay on
    # the tick thread; only adapters declaring concurrent_submit); 1 = serial. v3-c8a B5: 6 (node worktree adds of
    # different nodes now share the adapter's worktree lock, so the threads overlap the ~25 s virtiofs adds too)
    submit_workers: int = 6
    # v3-c6a: while TEST-stage runs are owed (an open TEST block's run not yet launched), that many primary slots are
    # RESERVED for them: no other run (repool, idle-churn move, retry, screen, smoke, overflow queue) takes a primary
    # slot a test waits for (live 10-13: a node's block waited 3 h while repooled screens took every freed slot)
    test_reserve: bool = True
    test_reserve_max_h: float = 3.0             # v3-c6r: an owed count unchanged this long releases the reserve (0 = never)
    # EFFECTIVE CAPACITY (10-21): a pool's slots are cut to what its association's ``GrpTRES gres/gpu`` limit leaves after
    # the user's OTHER (non-loop) running jobs in the pool's account (``adapter.pool_capacity``: sacctmgr + squeue, read
    # only, once per tick). The loop then submits no more than it can run and the heartbeat shows the real number.
    effective_capacity: bool = True
    # STALE PENDING (10-21): a loop run PENDING this long (minutes; 0 = off) for one of ``stale_pending_reasons`` (a
    # prefix of SLURM's reason: AssocGrpGRES, QOSMax..., i.e. a pool that cannot schedule it) is moved to a pool with a
    # free slot (a free retry: same attempt) or, with none, its job is scancelled (by id, the loop's own) and the run
    # returns to the queue as an unspent INFRA void the tick re-submits when a slot is free
    stale_pending_min: float = 90.0
    stale_pending_reasons: tuple[str, ...] = ("AssocGrp", "AssocMax", "QOSGrp", "QOSMax")


@dataclass(frozen=True)
class LablessCfg:
    """Auto-submit to Labless (loop.labless); the submitter is <repo>/labless/. Tiers: 'audited' (passed the
    held-out audit) and 'exploratory' (every CONFIRMED node at once, plus one representative -- its best member,
    even a poor one -- per CLOSED tested group: ladder / tuning pass / leg round / design batch / mechanism cluster)."""
    enabled: bool = False
    min_hours_between: float = 4.0             # at most one submission attempt per this many hours (0 = no spacing)
    max_per_day: int = 20                       # rolling 24 h cap on submission attempts, all tiers (0 = no cap)
    exploratory: bool = True                    # the exploratory tier's kill switch
    group_max_age_h: float = 48.0               # a group closed longer ago than this is never sent (0 = any age)
    min_group: int = 2                          # a group needs at least this many tested (scored) members
    sibling_gain: float = 0.0081                # an exploratory near-duplicate of a submitted / earlier-queued node
                                                # goes only with a shrunk panel gain >= this over it (graduation bar)
    python: str = ""                            # "" -> <repo>/.venv/bin/python
    token_file: str = "~/.labless_autosubmit.json"
    timeout_min: float = 30.0
    # node -> {submission_id, run_id} submitted outside the loop: recorded at startup, never resubmitted
    recorded: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    # the EXTERNAL bar (loop.core.bar; operator 2026-10-12): the node whose submission Labless accepted. A new
    # incumbent is submitted only if its mean beats the bar's by more than bar_margin (DIRECT, no root terms:
    # P(mean_cand - bar_score > bar_margin) >= 1 - bar_alpha, one-sided); "" = no bar (every new incumbent is
    # submitted). bar_score: the bar's official Labless mean instead of its local runs'; bar_se its standard error.
    bar_node: str = ""
    bar_score: float | None = None
    bar_se: float = 0.0
    bar_margin: float = 0.0
    bar_alpha: float = 0.1
    # THE BAR = Labless's acceptance rule (operator 2026-10-13; loop.core.bar; supersedes bar_score / bar_se /
    # bar_margin / bar_alpha, which stay readable: without leader_median, leader = bar_score and margin = bar_margin;
    # bar_se / bar_alpha are unused): the maintainer reruns a candidate on 3 fresh seeds and accepts it iff the median
    # beats leader_median by >= promote_margin. A node beats the bar iff P(median of 3 fresh seeds >= leader_median +
    # promote_margin) >= promote_p, from its own seed distribution on >= min_test_seeds fresh TEST seeds; offset (the
    # Labless - local level offset, added to our means) is 0 = no credit; sigma_prior_df: the pooled per-run sigma's
    # pseudo-df in the blend with the candidate's own sd.
    leader_median: float | None = None
    promote_margin: float = 0.004
    promote_p: float = 0.7
    offset: float = 0.0
    min_test_seeds: int = 4
    sigma_prior_df: float = 8.0
    # OUR LEADER (loop.leader): when the board (loop.board's stored rows) marks one of OUR submissions the Labless
    # leader (matched by submission id or exact code commit), leader_median is re-based every tick to max(configured,
    # that submission's estimate): a Labless rerun median from one of leader_median_keys when the row carries one,
    # else its node's local mean (fresh TEST seeds when >= min_test_seeds, else all its seeds). Another contributor
    # leading drops the re-base; no rows / a failed fetch keep the last known bar. Only leader_median is re-based
    # (never the legacy bar_score / bar_node). The incumbent is never resubmitted when it (or its exact code commit)
    # already has a submission in a book, labless.recorded or the board (dedupe_submitted).
    leader_rebase: bool = True
    leader_median_keys: tuple[str, ...] = ("rerun_median", "replicated_median", "replication_median",
                                           "validated_median", "leader_median")
    dedupe_submitted: bool = True
    # ... and the moment our VALIDATED submission is the Labless leader its node becomes the incumbent (loop.leader
    # reroot_plan / tick._leader_reroot): a frozen TEST pass over the incumbent (tier 'leader') through the usual
    # replay, so remint, stale-control re-pairing, carriers and re-root ports follow; once per leader run, and never
    # when it is already the incumbent or one of its ancestors
    leader_reroot: bool = True
    # S1 (10-13): mu's empirical-Bayes Student-t prior (median / MAD of the current epoch's node means, as
    # core.model) needs >= this many nodes; 0 = the flat prior (passed ~19% of true-P=.5 candidates at n=4)
    eb_min_nodes: int = 5
    # MULTI-PROJECT (pkg C): submit_cmd empty = nanopath's labless/submit_to_labless.py argv, byte-identical.
    # Else the argv with {python} {out_dir} {repo} substituted (nanobrain: ["{python}","scripts/submit.py","{out_dir}"]).
    submit_cmd: tuple[str, ...] = ()
    # {remote: "", ref: "loop/{node}"}: before submitting, `git push <remote> <run sha>:refs/heads/<ref>` ("" remote = no push)
    push: Mapping[str, str] = field(default_factory=dict)
    # write the (<=20 char) submission name + notes into <out_dir>/config.yaml for the submitter, restored after
    name_into_config: bool = False


@dataclass(frozen=True)
class EvalOnlyCfg:
    """EVAL-ONLY READOUT RUNS (loop.evalonly): a readout-only candidate is scored by resuming trained models'
    checkpoints instead of retraining; it must generalise across >= min_models distinct models."""
    enabled: bool = False
    family: str = "eval_readout"                # whose ``file::`` spans a change must lie in
    max_share: float = 0.25                     # of the slots, at most, held by eval-only runs
    screen_ckpts: int = 3
    confirm_ckpts: int = 2                      # fresh checkpoints (screen + confirm <= seeds.max_seeds_per_node)
    min_models: int = 2
    sigma_floor: float = 0.0005                 # floor on the across-checkpoint delta sd
    null_models: int = 2                        # the null control: each model's own code on its own checkpoint
    null_tol: float = 0.0001                    # ... must reproduce final_score within this
    # v3-c5 I5 (the v3 tick's run-level lane, loop.evalonly.Lane): spans of the family that TRAINING executes; a
    # change in them is eval-only only inside an ``if not self.training ...`` block (else: training path, full run)
    train_spans: tuple[str, ...] = ()
    readout_keys: bool = False                  # a config key / literal constant read ONLY inside the spans is eligible
    harm_tol: Mapping[str, float] = field(default_factory=dict)   # axis -> the paired-delta loss it may not exceed
    harm_prob: float = 0.8                      # ... with this posterior probability (after screen_ckpts pairs)
    source_wait_s: float = 7200.0               # a run waits this long for its parent's same-seed checkpoint
    noise_min_df: int = 4                       # pooled eval-only pair dfs before the eval-only rho replaces far/near
    null_at_start: bool = True                  # v3-c6a: the null control starts once per epoch on its own trigger (on
                                                # the incumbent's checkpoints), before any eval-only candidate arrives


@dataclass(frozen=True)
class PruneCfg:
    """CHECKPOINT PRUNING (loop.prune): latest.pt files of closed, unreferenced nodes and eval-only runs."""
    enabled: bool = False
    dry_run: bool = True                        # log what would go, delete nothing
    every_min: float = 60.0
    grace_h: float = 24.0                       # a node closed less than this ago keeps everything
    reserve_per_model: int = 5                  # done runs kept per model (root, incumbent, confirmed, ...)
    git: bool = True                            # + research branches / node worktrees of long-closed work (node commits tagged)
    smoke_max_age_h: float = 6.0                # smoke checkpoints (state/smoke/<wt>/out latest.pt) older than this go
    # v3-c8a B6: WHOLE smoke dirs of research requests no longer in flight, newest entry older than this (232 dirs /
    # 3.2 GB on 10-14, mostly out/labless_source copies)
    smoke_dir_max_age_h: float = 24.0
    # alt-base protection: the best ``alt_top`` sound standing current-epoch nodes by posterior keep their checkpoints,
    # plus every such node within one posterior sd of the Nth's value (hysteresis against ranking jitter)
    alt_top: int = 60
    # challenger protection (10-22 ops review: a node, the best challenger by level, settled as a screening node and its
    # checkpoints were deleted): the best ``challenger_top`` live (SCREENING / PASSED / TESTING) current-epoch nodes by
    # ABSOLUTE composite level, every such node within one sd of the best non-incumbent one, and every node in TEST keep
    # their checkpoints (a test retrains from the commit, but the challenger's own checkpoint is its evidence)
    challenger_top: int = 8


@dataclass(frozen=True)
class FactorialCfg:
    """THE HP / STACKING ENGINE (loop.core.factorial): a diagnostics flag -> the knobs a wave tries for it
    ({key, lo, hi, scale lin|log} each, the researcher finish's ``knobs`` schema)."""
    flag_knobs: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)
    stack_min_seeds: int = 2                    # a stacking leg: own posterior > 0 on at least this many seeds
    stack_priority: bool = True                 # a stacking wave takes a relax wave's place (rank) and runs primary
    rearm_max: int = 1                          # a wave that built 0 points re-offers its valid knobs this often
    prereq_ports: bool = True                   # a knob / port needing a config key the carrier lacks: the key's
    #                                             introducer is ported onto the carrier first (<= 1 open), then retried
    churn_resubmit_max: int = 1                 # resubmits of a wave point the idle churn cancelled (0: abandon)
    resume_points_per_tick: int = 4             # v3-c5b M2: open waves' never-ran points launched per tick (after tests + primary)
    stack_leg_z: float = 1.0                    # v3-c5b S1: a stacking leg's own posterior mean >= this x its sd
    stack_leg_z_min: float = 0.75               # v3-c6a: < 2 legs at stack_leg_z -> relaxed to this; still < 2: no wave


@dataclass(frozen=True)
class PairsCfg:
    """v3-c6a CODE-PICKED CROSSOVERS (loop.core.pairs; GEAR arXiv 2605.13874: code-chosen complementary pairs 71%
    elite vs 14% LLM-chosen): partner pairs picked from the posterior, built as config-only merges on the incumbent
    or handed to the ideation side as a crossover brief naming the exact pair (``pairs.next_pair``; claimed at the
    session's start). v3-c6r (adversarial review 10-13): ranked by the additive shrunk prediction corrected for the
    winner's curse, axis legs Bonferroni-gated, old-epoch config moves transferred only as the same move."""
    enabled: bool = True
    min_seeds: int = 2                          # a leg: an own-increment estimate on at least this many seeds
    leg_z: float = 1.0                          # a composite leg: shrunk own posterior mean >= this x its sd (and > 0)
    old_epoch: bool = True                      # legs from older base epochs too (config-only: merged directly;
                                                # code: ported first, then paired)
    old_min_seeds: int = 3                      # ... an old-epoch original needs this many seeds (n=2: selection noise)
    old_epoch_min_z: float = 0.0                # ... an old-epoch composite leg's shrunk own mean >= this x its sd
    # v3 review K5: an old-epoch leg is discounted ONCE, by the posterior's V / OLD_EPOCH_W (``post.belief``); this
    # multiplies its shrunk mean on top only when < 1 is set explicitly (default 1: no second discount)
    old_w: float = 1.0
    combo_legs: bool = False                    # crossover / stack / merged-pair / factorial nodes as legs (off: their
                                                # own increment carries their legs' gains)
    axis_leg_z: float = 2.5                     # an AXIS leg: one axis's own gain at z >= max(this, the Bonferroni z)
    axis_alpha: float = 0.05                    # ... Bonferroni over candidate nodes x axes (one-sided): ~z 4 live
    axis_min_seeds: int = 3                     # ... on at least this many seeds
    # ... never on these axes; None = the project's (loop.project.axis_exclude; nanopath: surv, ~pure noise)
    axis_exclude: tuple[str, ...] | None = None
    axis_leg_tol: float = 0.002                 # ... and a shrunk composite >= -this (it must repair its partner)
    parent_tol: float = 0.004                   # a leg's parent: its own CUPED composite increment >= -this
    min_pred: float = 0.0                       # a pair: its stack-prior prediction (acquire.stack_mean) above this
    # winner's curse: rank on pred - curse x sqrt(2 ln N_legs) x sd(pred). v3 review K5: 0 -- the legs are already
    # posterior means (shrunk), so a curse on top double-shrinks (simulated: a pure -.0044 shift of every pair)
    curse: float = 0.0
    leg_cap: int = 2                            # a leg appears in at most this many of the ranked pairs (0 = no cap)
    coverage_w: float = 0.25                    # ... + this x the coverage (what the union buys beyond the better leg)
    dedup_jaccard: float = 0.5                  # config legs whose moves overlap this much are one mechanism (best kept)
    builds_per_tick: int = 1                    # config-only merges built per tick
    max_open: int = 2                           # merged-pair nodes still screening at once
    ports_per_tick: int = 1                     # old-epoch code legs of the top_k pairs ported per tick
    max_open_ports: int = 2                     # ... with at most this many planner ports still screening
    top_k: int = 5                              # pairs reported per tick (shadow / log); ports come only from these
    config_reads_per_tick: int = 60             # new legs' config diffs (git reads) per tick, best legs first; cached
    backoff_s: float = 600.0                    # a failed build / port / git read: retried after this x 2^(k-1) s ...
    max_failures: int = 3                       # ... and given up (pair marked failed) after this many failures


@dataclass(frozen=True)
class HoldoutCfg:
    """v3-c6a BLIND HOLDOUT (loop.core.holdout; DAPS 2609.00077 / REUSE 2609.33180: loops overfit a reused benchmark):
    probe TASKS held out of every selection-facing number (posterior, acquisition, knob banks, briefs, levers); used
    only in the fresh-seed TEST audit (visible vs holdout gain of each passed node). The OFFICIAL panel (TEST verdict,
    Labless bar) is untouched: the full locked panel. Default OFF: the deterministic pick reads nanopath's per-task
    summary keys (loop.core.holdout); a task opts in (nanopath_v2.yaml)."""
    enabled: bool = False
    # "axis:task" (axis cls / seg / surv / rob -- slide and auc have one task each); () = the deterministic pick: per
    # axis with >= 2 tasks, the task minimising sha1("holdout:<axis>:<task>") (loop.core.holdout.default_tasks)
    tasks: tuple[str, ...] = ()
    alert_ratio: float = 2.0                    # ALERT: mean visible gain > this x the mean holdout gain ...
    min_nodes: int = 5                          # ... over at least this many passed nodes
    cache_workers: int = 16                     # threads reading summary.json files into the per-task cache
    noise_every_h: float = 3.0                  # the visible-vs-full composite noise note is refitted this often


@dataclass(frozen=True)
class DiagnosticsCfg:
    """RUN DIAGNOSTICS (loop.diagnostics): a detached low-priority checkpoint + metrics feature extraction per DONE run
    (loop/diag_ckpt.py under a torch python); its features feed the bottleneck flags and mechanism checks. Default OFF:
    the extractor reads nanopath's latest.pt / dino.* config schema; a task opts in (nanopath_v2.yaml)."""
    enabled: bool = False
    python: str = ""                            # a torch-capable python; '' = the run's worktree venv, else the repo's
    init: str = ""                              # the init weights file the drift features are measured from
    threads: int = 4                            # OMP / MKL / torch threads of one extraction
    pending_h: float = 2.0                      # a launched extraction with no output this long is abandoned


@dataclass(frozen=True)
class AcquireCfg:
    """TARGET-AWARE ACQUISITION (loop.core.acquire, docs/loop-core-v3.md §8 "Target-aware acquisition"). OPERATOR
    keys (not meta-tunable). The target: Labless's acceptance rule (loop.core.bar: labless.leader_median +
    promote_margin, P(median of 3) >= promote_p); the gap = the local mean needed for it - the incumbent on our infra."""
    target_mode: str = "auto"                   # incumbent | target | auto (= target when a labless target is set)
    # 11-08 SHARED-CONTROL BAR (nanopath lesson: siblings screened on one control inherit its run offset): when >=
    # control_bar_share of a control's >= control_bar_min_siblings screened children look positive while that control's
    # mean on their screen seeds sits >= control_bar_z run-sds below its mean on its other seeds, a child enters the
    # fresh-seed test only if its belief still clears 0 after subtracting that shortfall. False = off (nanopath)
    control_bar: bool = False
    control_bar_z: float = 1.0
    control_bar_share: float = 0.7
    control_bar_min_siblings: int = 3
    # target mode: the gap is capped at this many per-run panel sigmas (core.bar.safe_sigma) so a far bar (a gap of
    # 10+ sigma) cannot prune every action / rank by variance alone; 0 = off (nanopath: uncapped)
    gap_cap_sigmas: float = 0.0
    # 11-09 FUTILITY STOP (loop.core.acquire.futile): a SCREENING node with >= futility_min_seeds done seeds whose
    # posterior P(gain over the incumbent > bar) < futility_p gets no more seed extensions / idle-fill seeds (its GPUs
    # go to unscreened ideas). bar = futility_gap_frac x the gap to the external target when measured, else
    # futility_delta. futility_p 0 = off. GENERIC DEFAULT .02 (11-10 generalization; nanobrain's measured choice)
    futility_p: float = 0.02
    futility_min_seeds: int = 3
    futility_gap_frac: float = 0.33
    futility_delta: float = 0.0
    fresh_shrink: float = 0.55                  # fresh / selected effect ratio (ledger 10-12)
    gap_floor_frac: float = 0.5                 # drop wave / port whose upper-90% effect < this x gap (seeds exempt)
    seed_floor_frac: float = 0.25               # ... a seed / idle-fill seed needs upper-90% >= this x gap (0 = off)
    stack_best_leg_frac: float = 0.74           # stack prior: >= this x the best shrunk leg (median stack/best-leg)
    stack_sum_frac: float = 0.24                # ... or this x the sum of the shrunk legs + stack_intercept
    stack_intercept: float = 0.0013             # (regression stack ~ .24 sum + .0013; 0% of stacks beat the sum)
    untested_var_mult: float = 2.0              # gap > best shrunk stack: prior variance x this for untested lanes
    idle_max_restarts: int = 2                  # an idle-fill run preempted this often -> free primary, else dropped
    idle_keep_frac: float = 0.5                 # ... unless resumable (latest.pt armed) and this far through training
    preempt_max_restarts: int = 3               # any other run preempted this often on a preemptible pool -> free
    #                                             primary; else dropped only if it cannot resume (or < keep_frac at 2x)
    idle_max_run_frac: float = 1.5              # idle fill skips nodes whose runs take > this x the median run
    gap_level_min_test: int = 4                 # incumbent level from its TEST-pool seeds when it has this many
    relax_fill: bool = True                     # free slots -> below-floor positive-value seed/wave/screen (preemptible)
    relax_max_waves: int = 0                    # open relax (below-floor, preemptible-only) waves at most (v3-c5: 1->0)
    fill_drop_below: bool = True                # target mode: idle fill (b) / relax DROP seeds under the seed floor
    max_open_waves: int = 3                     # open waves at most, ranked + relax (hard cap)
    # OPERATOR-PROGRAM PRIORITY (10-21: 6 program arms sat 1-4 h with no run behind a 320-node backlog): QUEUED nodes
    # whose notes carry ``program_arm=1`` are ranked FIRST for up to this share of the free slots (at least one
    # screen); 0 = off. They are run on purpose, so their screens are taken whatever the value floors say
    program_arm_share: float = 0.3
    # QUEUED nodes that never ran and are older than this many hours (created / last touched) are AGED OUT: stored
    # status REJECTED / reason ``dropped_backlog: aged_out_queued`` (what ``alloc.drop`` writes) -- recoverable (``acquire.restore_aged``), never deleted;
    # program-arm, wave and carrier nodes are exempt; 0 = off
    queued_max_age_h: float = 96.0
    # idea options: a lane whose sessions get another lane's brief is valued on THAT lane's prior ('' = no idea
    # option; 10-13: the factorial lane bought 45/65 explore sessions with its design prior)
    idea_lane_prior: Mapping[str, str] = field(default_factory=lambda: {"factorial": "explore"})
    # families too broad to mean "one mechanism" (ledger 10-13: training 338 nodes, augmentation 236, objective 415)
    generic_families: tuple[str, ...] = ("", "design", "baseline", "training", "augmentation", "objective",
                                         "optimization", "architecture")


@dataclass(frozen=True)
class StatusGateCfg:
    """The seed-artifact gate before a fresh test (loop.core.status.seed_artifact)."""
    max_seed_share: float = 0.4                 # the replaced share cap (v3 review K6): kept for old configs, unused
    min_seeds: int = 3                          # paired seeds the leave-one-seed-out check needs
    # v3 review K6: from this many paired seeds no seed may be a one-seed OUTLIER of the others: (d_s - mean of the
    # rest) / (sd_d sqrt(1 + 1/(n-1))) > outlier_z blocks (power ~.8 flat over n 3..5 at a true +.004)
    share_min_n: int = 3
    outlier_z: float = 1.8
    outlier_sd_d: float = 0.0034                # the outlier test's sd_d when the caller has no fit (nanopath's pooled
                                                # paired sd; a project with another noise level scales it)
    share_k: float = 2.4                        # the replaced n-scaled cap's slope: kept for old configs, unused
    trim_min_n: int = 4                         # v3-c7a A3: from this many paired seeds the trimmed mean (max and min
                                                # seed out) must keep >= trim_frac of the mean (0 = off)
    trim_frac: float = 0.25


@dataclass(frozen=True)
class TestCfg:
    """THE FRESH-SEED TEST (loop.core.test; LOCKED: ``test.`` keys are never meta-tunable). alpha / K / K_MAX stay
    module constants (.1 / 5 / 8; costs.k_test sets K's seeds)."""
    method: str = "gs2"                         # gs2 (two-look group-sequential, looks at K and K_MAX) | msprt (10-12)
    spending: str = "obf"                       # gs2 alpha spending: obf (Lan-DeMets O'Brien-Fleming type) | pocock
    futility_z: float = 0.0                     # gs2 look 1: Z below this FAILs (binding; priced into c2)
    ucb_df_max: float = 50.0                    # gs2: sigma_d as fitted past this many dfs, else at its 80% UCB
    # gs2: the pooled sigma_d is a prior worth nu0 dfs, blended with the test's OWN paired variance at each look
    # ((nu0 s2_pool + (n-1) s2_own) / (nu0 + n - 1), t dfs nu0 + n - 1): a noisier-than-pooled pair widens its own
    # sigma_d (ledger 10-13: pair variances heterogeneous, Bartlett p .03 epoch). 0 = the pooled sigma_d alone
    sd_prior_df: float = 4.0
    sd_pool: str = "pooled"                     # sigma_d from every clean pair of the epoch | "class" (near / far)
    # an open test whose control is no longer the incumbent (an incumbent switch) can switch nothing: it is re-paired
    # against the new incumbent when that one ran every seed of the block, else closed as stale_control; its PENDING
    # runs are cancelled, its RUNNING ones too when this is on (live 10-13: a node's test held 4 primary GPUs against
    # a node after a node became incumbent)
    stale_cancel_running: bool = True
    # meta guard: simulated type-I with sigma_d 1.25x its fit (meta.type1_sim) must stay <= this. gs2 spends alpha
    # exactly at the fit, so the stress sits at ~.155 (live pooled noise); 2 x alpha is the alarm level
    type1_guard_max: float = 0.2
    # an open test's member-side deaths (crash / eval / too_slow void; infra voids and loop cancels are re-drawn
    # freely): the candidate's this many close it FAIL "unscorable"; the control's this many close it with no verdict
    max_candidate_crashes: int = 2
    # v3-c5b S3: a test closed for an unscorable CONTROL leaves its candidate re-testable only after this many hours
    retest_backoff_h: float = 12.0
    # v3-c7a A4: a candidate whose test closed on ill-conditioned probe crashes (a repair booked) is re-testable only
    # after this many hours (the repair goes first)
    ill_backoff_h: float = 72.0
    # THE LOCAL-INCUMBENT TIER (operator 11-09 "reroot on free wins"; status.local_reroot, tick._local_reroot): at a
    # TEST look the gs2 test leaves undecided (look 1 EXTEND / waiting for a slot) or FAILs at its final look without
    # harm, the candidate still switches the incumbent (a PASS frozen with tier "local") when its >= min_fresh fresh
    # pairs alone have mean >= min_gain and z >= fresh_z, its Gaussian EB-shrunk paired gain over the incumbent on every
    # paired seed is >= min_gain with P(gain > 0) >= p, and no seed artifact. One decision per look. It submits nothing
    # itself (the incumbent-only Labless submitter still needs the unchanged bar). Off = gs2 only. GENERIC DEFAULT ON
    # (11-10 generalization); local_reroot_min_gain unset in the yaml = 0.6 x loop.panel_sd (SIGMA_SCALED).
    local_reroot: bool = True
    local_reroot_p: float = 0.95
    local_reroot_min_fresh: int = 4   # a floor only: a look already needs every planned seed (>= test.K = 5) in
    local_reroot_min_gain: float = 0.002
    local_reroot_fresh_z: float = 1.0
    # RE-ROOT PORTS (operator 11-10; tick._reroot_ports, altbase.reroot_plan): after ANY incumbent switch (gs2 or local),
    # the old incumbent's descendants with an EB-shrunk paired gain over the OLD incumbent > 0 on >= reroot_port_min_seeds
    # paired seeds with P(gain > 0) >= reroot_port_p -- the best per lineage, at most reroot_port_k per switch -- are
    # grafted (their whole change over the old incumbent) onto the new one (port_src=reroot) and screened ahead of the
    # ranked queue. Same-family changes sharing a source file with the new incumbent's own change are skipped (redundant
    # mechanism). Evaluated every tick while the incumbent holds (retroactive for a switch made before deploy). Off = none.
    # GENERIC DEFAULT ON (11-10 generalization).
    reroot_port: bool = True
    reroot_port_k: int = 4
    reroot_port_p: float = 0.8
    reroot_port_min_seeds: int = 3
    # RE-ROOT CROSSOVERS (operator 11-10: a node token-shuffle detection +.0049 was skipped as 'same mechanism' as the
    # new incumbent a node, replaced-token detection -- both objective, both in model.py, but different aux heads): a
    # candidate skipped ONLY for family + shared-file overlap whose own evidence clears the port bar above is not
    # dropped -- a CROSSOVER new incumbent x candidate is made instead (a clean code graft -> a code-made pair node; a
    # conflict -> one researcher crossover session that merges both), screened ahead of the queue like a port. A true
    # same mechanism (identical mechanism-tag stem, or an ancestor / descendant of the new incumbent) is still skipped.
    # At most reroot_cross_k per re-root, best EB gain first, one per lineage (a ported lineage included). Off = skip.
    reroot_cross: bool = True
    reroot_cross_k: int = 2


@dataclass(frozen=True)
class PruneStaleCfg:
    """``alloc.prune_stale_family`` (11-09 nanobrain; absent = off, nanopath unchanged): a QUEUED node older than ``age_h``
    whose mechanism cluster already has >= ``min_measured`` members measured on >= ``min_seeds`` seeds, none of them (nor
    their measured descendants: tuning ladders, doses) above ``max_best_delta`` by posterior or raw paired gain, is
    REJECTED ``pruned_stale_family`` (recorded in the tested-mechanism memory); at most ``max_per_tick`` per tick.
    Confirm / root-replicate / program / in-flight / priority-category nodes are never pruned (loop.core.alloc.stale_family)."""
    enabled: bool = True
    age_h: float = 6.0
    min_measured: int = 3
    max_best_delta: float = 0.004
    min_seeds: int = 2
    max_per_tick: int = 10


@dataclass(frozen=True)
class AllocCfg:
    """v3-c7i BUDGET ALLOCATION (loop.core.alloc; 24-h measurement 10-14: LLM-idea lanes took ~75% of the GPU-h for 1
    of 49 measured nodes >= +.002, waves + code crossovers 17% for 13 of 15). Every value is computed from the ledger:
    nothing names a lane or an idea."""
    enabled: bool = True
    # I1 kind bandit: units (a brief-tagged node's clade, a wave, a code-made pair) ranked on yield per GPU-h -- the
    # sum of max(0, the unit's best value) over its non-test GPU-h, shrunk to the pooled rate with rate_n0 pseudo-units
    # of the pooled mean cost -- weighted by softmax((rate - best) / T), T = 1.4826 x MAD of the kinds' rates
    kind_rate: bool = True
    rate_n0: float = 3.0
    temp_floor_frac: float = 0.25               # T >= this x the pooled rate (a degenerate MAD never gives one-hot)
    # program_share (ideation.program_share) is a CEILING: the effective share decays toward this floor while the
    # program kind's rate is below the pooled rate (floor + (ceiling - floor) x min(1, rate / pooled))
    program_share_floor: float = 0.05
    # I2 GPU split: LLM-idea units' runs hold at most share x the GPU slots (soft: lifted for slots nothing else wants),
    # share = rate_llm / (rate_llm + rate_code) clamped to [llm_gpu_floor, llm_gpu_cap]
    llm_gpu_cap: float = 0.4
    llm_gpu_floor: float = 0.1
    # v3 review K2: the share stays at llm_gpu_cap until the bootstrap 90% CI of rl / (rl + rc) is wholly below it;
    # and this share of the GPU slots is HARD-reserved for seed 1-2 screens of new LLM-idea nodes on the incumbent
    discovery_reserve: float = 0.25
    # v3 review S5: a unit's best member is the max of k noisy increments: subtract the expected max of k noise draws
    # (sd = unit_noise_sd / sqrt(seeds) per member) so units with many members (ladder doses, opposite arms, iso children,
    # split parts) do not inflate the kind bandit; 0 = off
    unit_noise_sd: float = 0.0025
    # brief issuance throttle: no new research brief while the current-base QUEUED backlog exceeds backlog_max nodes
    # or its median age exceeds backlog_age_h
    backlog_max: int = 40
    backlog_age_h: float = 6.0
    # 11-09 trickle: while throttled, up to backlog_trickle research sessions may still be in flight (0 = the hard
    # stop), explore-lane slots only, briefed as backlog_trickle_kinds in turn (quality over volume: papers / bold)
    backlog_trickle: int = 0
    backlog_trickle_kinds: tuple[str, ...] = ("literature", "bold")
    # 11-09: operator-intended QUEUED nodes (acquire.program_child: program arms / ladder doses / local confirms) are
    # left out of the backlog_max count (nanobrain on; nanopath off = every QUEUED node counts, unchanged)
    backlog_exclude_program: bool = False
    # age-out: a QUEUED node older than drop_age_h whose screen value / GPU-h is outside the best backlog_keep is
    # dropped (REJECTED 'dropped_backlog', logged; never deleted), at most drop_max per tick. A zero-run QUEUED node older
    # than drop_age_h is dropped even under backlog_keep (reason 'starved_zero_run'; not program arm / wave / port / open block)
    drop_age_h: float = 24.0
    backlog_keep: int = 40
    drop_max: int = 25
    # 11-09: an operator-bank idea's node (notes ``bank=<id>``, not a scout bank) and a root-audit arm (``root_audit=1``)
    # are exempt from both drops like a program arm (nanobrain on; nanopath off = unchanged)
    keep_bank_nodes: bool = False
    # I3 mechanism clusters (TF-IDF of family + tag + signature, leader clustering at cluster_tau): a cluster with >=
    # cluster_min_n measured nodes and a shrunk mean <= 0 is SATURATED-NEGATIVE (its share of the last cluster_window
    # new research nodes capped at cluster_cap_share: check_idea / intake reject past it); a positive shrunk mean on
    # fewer measured nodes is boosted in the brief
    clusters: bool = True
    cluster_tau: float = 0.3
    cluster_min_n: int = 8
    cluster_n0: float = 4.0
    cluster_cap_share: float = 0.05
    cluster_window: int = 60
    cluster_rows: int = 10
    # FAMILY-LEVEL NEGATIVE BLOCK (10-21: opus proposals kept landing in mechanism families that measured negative --
    # position / geometry pretext heads: 9 nodes, mean -.0066 -- which the cosine dedup and the saturated-cluster SHARE cap
    # miss): a proposal whose mechanism cluster has >= family_block_min_n measured nodes with mean own gain <=
    # family_block_mean and NONE above +sigma (family_block_sigma; 0 = loop.panel_sd) is rejected ``negative_family`` at
    # check_idea / check / intake unless its ``differs_from_cluster`` names >= 2 of the cluster's measured nodes, says how
    # each differs (family_block_explain_min characters). LLM-free. A TEST-pass /
    # incumbent-ancestor cluster, a repair / port session and an operator program arm are exempt
    family_block: bool = True
    family_block_min_n: int = 5
    family_block_mean: float = -0.004
    family_block_sigma: float = 0.0
    family_block_cite: int = 2
    family_block_explain_min: int = 30          # characters of 'how it differs' per cited node (a bare id list never passes)
    # a tag stem shared by >= family_block_min_n measured research nodes is a FAMILY too (the lexical clusters scatter e.g.
    # position-prediction heads), unless more than this share of all research nodes has it (a word every idea has)
    family_block_max_share: float = 0.15
    # ... and never by a GENERIC / polysemous stem (a prefix match on the 7-character stem): 'rate' is coding rate, learning
    # rate and sampling rate, 'project' is projection head and project-level sampling, 'norm', 'gate', 'student', 'entropy'
    # ... name no mechanism (replay 10-21 over the 46 research nodes a node+: 11 blocked on stems like these, 9 of them a
    # different mechanism than the family's nodes -- sampling_rate_jitter vs coding_rate_*, project_level_flatten_sampler
    # vs *_projection); the clusters (lexical centroids) are unaffected
    family_block_stop: tuple[str, ...] = (
        "rate", "norm", "gate", "student", "teacher", "spatial", "project", "entropy", "noise", "loss", "weight", "temp",
        "channel", "objective", "backbone", "clean", "tail", "level", "head", "scale", "token", "block", "layer", "patch",
        "batch", "tile", "image", "stage", "input", "output", "global", "local", "crop", "train", "model", "decay",
        "schedule", "feature", "embed", "dropout", "drop")
    # PROGRAM SHADOWS EXPLORE (10-21: four sessions on project / site sampler flattening while the program's arms covered
    # it): while an operator-program arm is in flight, an explore / literature proposal near its mechanism (a config key
    # the arm owns, the same cluster as its node, or >= shadow_min_terms distinctive tag stems shared with the arm's
    # description) is rejected ``shadowed_by_program``; an arm open longer than shadow_max_h (0 = no limit) stops shadowing
    shadow: bool = True
    shadow_max_h: float = 96.0
    shadow_min_terms: int = 2
    # I4 the dead / in-flight mechanisms nearest a brief's focus, listed in the brief (check_idea rejects them)
    dedup_k: int = 8
    # I5 the LLM crossover brief only for a code-picked pair whose code graft CONFLICTS; a pair that grafts cleanly is
    # a code-made pair node (pair_of=, graft): at most xover_grafts_per_tick per tick, xover_max_open still unscreened
    xover_conflict_only: bool = True
    xover_grafts_per_tick: int = 2
    xover_max_open: int = 4
    # a grafted pair queued longer than this never screens under the backlog's ordering (live 10-21: 5 grafts queued 2-7 days
    # held all 4 slots, so no graft was made for 2 days): it no longer counts as open (0 = count every one)
    xover_open_max_age_h: float = 48.0
    # I6 fork-and-flush: no new round once >= fnf_pause_rounds closed rounds show no positive fresh-seed test effect
    fnf_pause_rounds: int = 2
    # I9 a node whose declared diagnostic REVERSED after its first seed and whose paired first-seed delta is negative
    # gets no more screen seeds (its running screen runs are cancelled)
    early_kill: bool = True
    # 11-09 BOLD FIRST: QUEUED nodes whose brief kind / taxonomy category / slate category / strategy is one of these
    # (e.g. [bold]) have their screens placed before the other screens in acquire.rank (program arms still first;
    # tests / seeds / waves keep their places). () = off (nanopath unchanged)
    priority_categories: tuple[str, ...] = ()
    # 11-09 STALE-TWEAK PRUNE (PruneStaleCfg; None = off)
    prune_stale_family: PruneStaleCfg | None = None


@dataclass(frozen=True)
class CoreMetaCfg:
    """THE 3-HOUR META-CYCLE (loop.core.meta, docs/loop-core-v3.md §7). OPERATOR keys: LOCKED, never meta-tunable."""
    every_h: float = 3.0                        # a meta report every this many hours of loop wall time
    review: bool = True                         # an Opus meta-review of each report (no tools) -> proposals
    locks: bool = True                          # each tick: git-blob hashes of LOCKED_FILES vs the deployed HEAD
    split_h: float = 72.0                       # a config backtest evaluates on the last this many hours of outcomes
    ab_share: float = 0.3                       # sessions on a prompt A/B arm
    ab_min_sessions: int = 10                   # sessions per arm before the arm is judged (fresh-seed credit)
    ab_min_screens: int = 6                     # v3-c6a: ... and nodes with a screening posterior per arm (no retire /
                                                # adopt on 0-vs-0 credit: screen posterior credit decides when no test)
    ab_max_sessions_mult: int = 10              # still uninformative after this x ab_min_sessions: ended inconclusive


@dataclass(frozen=True)
class LoopConfig:
    execution_adapter: str                      # "module:Class"
    axes: tuple[str, ...]                       # from slots.vector / slots.weights
    weights: Mapping[str, float]                # normalised over axes
    panel_sd: float                             # measured paired-delta sd
    axis_sd: Mapping[str, float]
    state_dir: str = ""
    slots: int = 8
    seeds: SeedPlan = field(default_factory=SeedPlan)
    promotion: PromotionCfg = field(default_factory=PromotionCfg)
    prior: PriorCfg = field(default_factory=PriorCfg)
    backfill: BackfillCfg = field(default_factory=BackfillCfg)
    veto: VetoCfg = field(default_factory=VetoCfg)
    noise_prior: NoisePriorCfg = field(default_factory=NoisePriorCfg)
    calibrate: CalibrateCfg = field(default_factory=CalibrateCfg)
    hypotheses: HypothesisCfg = field(default_factory=HypothesisCfg)
    levers: LeverCfg = field(default_factory=LeverCfg)
    families: Mapping[str, FamilyRule] = field(default_factory=dict)
    init_compat: InitCompatCfg = field(default_factory=InitCompatCfg)
    islands: IslandCfg = field(default_factory=IslandCfg)
    bandit: BanditCfg = field(default_factory=BanditCfg)
    intake: IntakeCfg = field(default_factory=IntakeCfg)
    researcher: ResearcherCfg = field(default_factory=ResearcherCfg)
    review: ReviewCfg = field(default_factory=ReviewCfg)
    design: DesignCfg = field(default_factory=DesignCfg)
    moonshot_tune: MoonshotTuneCfg = field(default_factory=MoonshotTuneCfg)
    campaign: CampaignCfg = field(default_factory=CampaignCfg)
    extrapolate: ExtrapolateCfg = field(default_factory=ExtrapolateCfg)
    databank: DatabankCfg = field(default_factory=DatabankCfg)
    legs: LegsCfg = field(default_factory=LegsCfg)
    meta: MetaCfg = field(default_factory=MetaCfg)
    scout: ScoutCfg = field(default_factory=ScoutCfg)
    board: BoardCfg = field(default_factory=BoardCfg)
    local_runs: LocalRunsCfg = field(default_factory=LocalRunsCfg)
    root_audit: RootAuditCfg = field(default_factory=RootAuditCfg)
    tested: TestedCfg = field(default_factory=TestedCfg)
    knowledge: KnowledgeCfg = field(default_factory=KnowledgeCfg)
    ideation: IdeationCfg = field(default_factory=IdeationCfg)
    components: ComponentsCfg = field(default_factory=ComponentsCfg)
    novelty: NoveltyCfg = field(default_factory=NoveltyCfg)
    heartbeat: HeartbeatCfg = field(default_factory=HeartbeatCfg)
    dashboard: DashboardCfg = field(default_factory=DashboardCfg)
    slurm: SlurmCfg = field(default_factory=SlurmCfg)
    labless: LablessCfg = field(default_factory=LablessCfg)
    evalonly: EvalOnlyCfg = field(default_factory=EvalOnlyCfg)
    prune: PruneCfg = field(default_factory=PruneCfg)
    diagnostics: DiagnosticsCfg = field(default_factory=DiagnosticsCfg)
    factorial: FactorialCfg = field(default_factory=FactorialCfg)
    pairs: PairsCfg = field(default_factory=PairsCfg)
    holdout: HoldoutCfg = field(default_factory=HoldoutCfg)
    core_meta: CoreMetaCfg = field(default_factory=CoreMetaCfg)
    acquire: AcquireCfg = field(default_factory=AcquireCfg)
    status: StatusGateCfg = field(default_factory=StatusGateCfg)
    test: TestCfg = field(default_factory=TestCfg)
    alloc: AllocCfg = field(default_factory=AllocCfg)
    operator_focus: str = ""                 # injected verbatim into ideation briefs and review prompts
    # the tick's disk guard: below this many GB free on the state / worktree filesystem no new research session starts
    # (0 = $LOOP_DISK_FLOOR_GB, default 30). nanobrain 11-08: a 5.9 TB FS shared with other users sits near 10-40 GB
    disk_floor_gb: float = 0.0
    idea_banks: tuple[str, ...] = ()            # JSONL idea banks (loop.select.bank), each idea served once
    # "dir: what it is" curated datasets a data.dataset_dir change may use. The data pipeline is NOT fixed: a future
    # option is offline re-tiling of the TCGA WSIs into new Arrow datasets (larger tiles, multi-magnification pairs);
    # the compute-per-tile family (more / larger views of the same 224 px tiles) was an established null, 2026-10-06
    data_bank: tuple[str, ...] = ()
    # SURFACE-level guard, applied to EVERY candidate diff (all families, design points, crossover,
    # moonshot): regexes no added line may match; forbid_config "file:dotpath" adds its names/paths.
    forbid: tuple[str, ...] = ()
    forbid_config: tuple[str, ...] = ()
    forbid_roots: tuple[str, ...] = ()          # derived: forbid_config's paths (no config value may reach them)
    # multi-project (loop.project): axes / panel weights / target / guard of a non-nanopath project; None = the
    # nanopath defaults of loop.project. loop.domain: prompt overlay dir + domain text rules. Installed into the
    # per-process registry by load_context.
    project: ProjectCfg | None = None
    domain: DomainCfg | None = None

    def sigma(self, target: str) -> float:
        return self.panel_sd if target == PANEL else self.axis_sd[target]

    @property
    def pools(self) -> tuple[PoolCfg, ...]:
        pools, n = self.slurm.pools or (PoolCfg("default", slots=self.slots),), self.slurm.preemptible_slots
        return pools if n < 0 else tuple(dataclasses.replace(p, slots=n) if p.preemptible else p for p in pools)

    @property
    def capacity(self) -> int:
        return sum(p.slots for p in self.pools)


@dataclass
class LoopContext:                              # cfg: the EFFECTIVE config (Driver installs self-tune overrides)
    spec: TaskSpec
    cfg: LoopConfig
    noise: NoiseModel                           # from THIS spec, never config.TASK_NOISE
    task_id: str
    spec_path: str
    state_dir: str
    spec_cfg: Any = None                        # the SPEC's config (before self-tune overrides), when known



def encode(obj: Any) -> Any:
    """Dataclass / Enum / Mapping / sequence -> plain JSON-able structure."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Mapping):
        return {k: encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    return obj


_OK = {bool: lambda v: True, str: lambda v: isinstance(v, str),
       int: lambda v: isinstance(v, int) and not isinstance(v, bool),
       float: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)}
_hints = functools.cache(typing.get_type_hints)     # the store decodes every row through build


def _scalar(tp: Any, value: Any, path: str) -> Any:
    if tp not in _OK:
        raise TypeError(f"{path}: unsupported type {tp}")
    if not _OK[tp](value):
        raise ValueError(f"{path}: expected {tp.__name__}, got {value!r}")
    return tp(value)


def build(tp: Any, value: Any, path: str = "") -> Any:
    """Decode ``value`` into type ``tp`` strictly; errors name ``path``."""
    if tp in _OK:
        return _scalar(tp, value, path)
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if tp is Any:
        return value
    if origin in (Union, types.UnionType):
        arms = [a for a in args if a is not _NONE]
        if value is None and len(arms) < len(args):
            return None
        return build(arms[0], value, path)
    if isinstance(tp, type) and issubclass(tp, Enum):
        try:
            return tp(value)
        except ValueError:
            raise ValueError(f"{path}: {value!r} is not one of {[m.value for m in tp]}") from None
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}: expected a mapping, got {type(value).__name__}")
        hints, fields = _hints(tp), {f.name: f for f in dataclasses.fields(tp)}
        for key in value:
            if key not in fields:
                raise ValueError(f"{path}.{key}: unknown key (known: {sorted(fields)})")
        missing = [n for n, f in fields.items() if n not in value and f.default is dataclasses.MISSING
                   and f.default_factory is dataclasses.MISSING]
        if missing:
            raise ValueError(f"{path}.{missing[0]}: required key is missing")
        try:
            return tp(**{n: build(hints[n], value[n], f"{path}.{n}" if path else n)
                         for n in fields if n in value})
        except ValueError as exc:
            raise ValueError(f"{path or tp.__name__}: {exc}") from exc
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path}: expected a list, got {type(value).__name__}")
        items = [args[0]] * len(value) if len(args) == 2 and args[1] is Ellipsis else args
        if len(items) != len(value):
            raise ValueError(f"{path}: expected {len(items)} items, got {len(value)}")
        return tuple(build(t, v, f"{path}[{i}]") for i, (t, v) in enumerate(zip(items, value)))
    if origin in (dict, Mapping, typing.get_origin(Mapping[str, Any])) or tp in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}: expected a mapping, got {type(value).__name__}")
        vt = args[1] if len(args) == 2 else Any
        return {str(k): build(vt, v, f"{path}.{k}") for k, v in value.items()}
    return _scalar(tp, value, path)


def _sd(value: Any, path: str, unmeasured: list[str]) -> float:
    if value == UNMEASURED:
        unmeasured.append(path)
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path}: expected a positive finite number or {UNMEASURED}, got {value!r}")
    return float(value)


#: ABSOLUTE SCORE-UNIT constants as multiples of the task's measured PAIRED panel sd (``loop.panel_sd``, written by
#: ``loop calibrate`` from root replicates). A key the task yaml leaves out resolves to ``mult x panel_sd`` (11-10
#: generalization: the dataclass literals were nanopath's, tuned at panel_sd .00326, and a task with another noise
#: level silently inherited them). The multipliers reproduce nanopath's literals to ~5%; tasks/nanopath_v2.yaml and
#: tasks/nanobrain.yaml pin their tuned values explicitly. An unmeasured panel_sd (calibrate) keeps the literals.
SIGMA_SCALED: Mapping[str, float] = {
    "status.outlier_sd_d": 1.0,                 # the outlier test's sd_d IS a paired sd (nanopath .0034)
    "pairs.parent_tol": 1.2,                    # nanopath .004
    "pairs.axis_leg_tol": 0.6,                  # nanopath .002
    "acquire.stack_intercept": 0.4,             # nanopath .0013
    "test.local_reroot_min_gain": 0.6,          # nanopath-scale .002
    "labless.promote_margin": 1.2,              # nanopath .004
    "labless.sibling_gain": 2.5,                # nanopath .0081 (the graduation bar)
    "slurm.resume_tol": 0.3,                    # nanopath .001
    "prior.tau0": 1.5,                          # nanopath .005
    "prior.tau_floor": 0.15,                    # nanopath .0005
    "prior.tau_max": 15.0,                      # nanopath .05
    "evalonly.sigma_floor": 0.15,               # nanopath .0005
    "evalonly.null_tol": 0.03,                  # nanopath .0001
    "alloc.prune_stale_family.max_best_delta": 1.2,   # nanopath-scale .004 (only when the block is declared)
}


def _given(block: Mapping[str, Any], dotted: str) -> bool:
    node: Any = block
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False
        node = node[part]
    return True


def _replace_path(obj: Any, parts: list[str], value: Any) -> Any:
    head, *rest = parts
    sub = getattr(obj, head)
    if rest and sub is None:
        return obj
    return dataclasses.replace(obj, **{head: _replace_path(sub, rest, value) if rest else value})


def raw_loop_block(spec: TaskSpec) -> Mapping[str, Any]:
    """The spec's raw ``loop:`` mapping (what the yaml declares; :func:`sigma_scaled` scales only what it leaves out)."""
    return dict(spec.loop or {})


def sigma_scaled(cfg: "LoopConfig", block: Mapping[str, Any]) -> "LoopConfig":
    """Every :data:`SIGMA_SCALED` key ``block`` (the raw ``loop:`` yaml) leaves out, set to mult x ``cfg.panel_sd``."""
    sd = cfg.panel_sd
    if not (isinstance(sd, float) and math.isfinite(sd) and sd > 0):
        return cfg
    for dotted, mult in SIGMA_SCALED.items():
        if not _given(block, dotted):
            cfg = _replace_path(cfg, dotted.split("."), mult * sd)
    return cfg


def derived_project(spec: TaskSpec, axes: tuple[str, ...]) -> ProjectCfg | None:
    """The ``loop.project`` of a spec that declares none (11-10 generalization): its axes are its ``slots.vector`` keys
    (short name = the key, weights = slots.weights), no guard / harm / axis exclusions. A spec whose axes ARE
    nanopath's probe keys keeps the registry's nanopath defaults (None: nothing installed, as before)."""
    if set(axes) == set(P._PROBE_KEYS.values()):
        return None
    return P.parse_project({"name": spec.name, "axes": [{"name": a, "key": a, "weight": float(spec.slots.weights[a])}
                                                         for a in axes]})


def _parse(spec: TaskSpec) -> tuple[LoopConfig, list[str]]:
    block = dict(spec.loop)
    if not block:
        raise ValueError(f"{spec.name}: the spec has no `loop:` block")
    for key in ("axes", "weights"):
        if key in block:
            raise ValueError(f"loop.{key}: derived from slots.*, do not declare it")
    for key in ("execution_adapter", "panel_sd", "axis_sd"):
        if key not in block:
            raise ValueError(f"loop.{key}: required key is missing")
    # ``project: null`` (explicit) keeps the registry's built-in defaults -- nanopath's axes; an ABSENT block derives the
    # project from slots (derived_project). fake_loop opts out: the core tests' fixture ledger is nanopath data.
    keep_default = "project" in block and block["project"] is None
    project, domain = P.parse_project(block.pop("project", None)), P.parse_domain(block.pop("domain", None))
    axes = tuple(spec.slots.vector) or tuple(spec.slots.weights)
    if project is not None and {a.key for a in project.axes} != set(axes):
        raise ValueError(f"loop.project.axes: keys {sorted(a.key for a in project.axes)} must equal slots.vector "
                         f"{sorted(axes)}")
    if not axes or set(axes) != set(spec.slots.weights):
        raise ValueError(f"{spec.name}: slots.weights must cover exactly the axes {list(axes)}")
    raw = {a: float(spec.slots.weights[a]) for a in axes}
    if any(w < 0 for w in raw.values()) or not sum(raw.values()) > 0:
        raise ValueError(f"{spec.name}: slots.weights must be >= 0 with a positive sum")
    if project is not None:
        pw = P._weights(project)                 # the registry's official weights (normalised as install does)
        for a in project.axes:
            if abs(pw[a.name] - raw[a.key] / sum(raw.values())) > 1e-9:
                raise ValueError(f"loop.project.axes.{a.name}: normalised weight {pw[a.name]:.6g} != the normalised "
                                 f"slots.weights.{a.key} {raw[a.key] / sum(raw.values()):.6g}")
        yamls = [w for w in spec.surface.writable if w.endswith((".yaml", ".yml"))]
        if project.config_rel and yamls and yamls[0] != project.config_rel:
            raise ValueError(f"loop.project.config_rel {project.config_rel!r} must be the FIRST yaml of "
                             f"surface.writable (the config every reader edits), got {yamls[0]!r}")
    raw_axis = block.pop("axis_sd")
    if not isinstance(raw_axis, Mapping) or set(raw_axis) != set(axes):
        raise ValueError(f"loop.axis_sd must map exactly the axes {list(axes)}")
    unmeasured: list[str] = []
    npr = block.get("noise_prior")
    if isinstance(npr, Mapping) and npr.get("axis_sd") and set(npr["axis_sd"]) != set(axes):
        raise ValueError(f"loop.noise_prior.axis_sd must map exactly the axes {list(axes)}")
    block.update(axes=list(axes), weights={a: w / sum(raw.values()) for a, w in raw.items()},
                 panel_sd=_sd(block.pop("panel_sd"), "loop.panel_sd", unmeasured),
                 axis_sd={a: _sd(raw_axis[a], f"loop.axis_sd.{a}", unmeasured) for a in axes})
    if spec.llm_default_backend not in ("qwen", "opus"):
        raise ValueError(f"llm_default_backend: {spec.llm_default_backend!r} must be 'qwen' or 'opus'")
    if unknown := sorted(set(spec.llm_roles) - set(C.LLM_ROLES)):
        raise ValueError(f"llm_roles.{unknown[0]}: unknown role (known: {list(C.LLM_ROLES)})")
    raw_block = dict(block)
    cfg = sigma_scaled(build(LoopConfig, block, "loop"), raw_block)
    if project is None and not keep_default:
        project = derived_project(spec, axes)
    if cfg.databank.enabled and not cfg.databank.root:
        raise ValueError("loop.databank.root: required when the data bank is enabled")
    if cfg.local_runs.enabled:                  # the scorer must resolve now, not at the first import tick
        mod, _, fn = (cfg.local_runs.scorer or f"{cfg.execution_adapter.partition(':')[0]}:task_score").partition(":")
        try:
            import importlib
            if not callable(getattr(importlib.import_module(mod), fn, None)):
                raise ValueError(f"loop.local_runs.scorer: {mod}:{fn} is not a function (set local_runs.scorer)")
        except ImportError as exc:
            raise ValueError(f"loop.local_runs.scorer: cannot import {mod}: {exc}") from None
    if ":" not in cfg.execution_adapter or cfg.slots < 1:
        raise ValueError("loop.execution_adapter must be 'module:Class' and loop.slots >= 1")
    if cfg.design.runs not in (16, 32) or cfg.design.step <= 1 or cfg.design.z <= 0 \
            or not 4 <= cfg.design.max_knobs <= 10 or not 0 < cfg.design.max_share <= 1:
        raise ValueError("loop.design: runs is 16 or 32, step > 1, z > 0, 4 <= max_knobs <= 10, 0 < max_share <= 1")
    for pat in (*cfg.init_compat.deny, *cfg.init_compat.allow, *cfg.init_compat.allow_types):
        try:
            re.compile(pat)
        except re.error as exc:
            raise ValueError(f"loop.init_compat: bad regex {pat!r}: {exc}") from exc
    ic = cfg.ideation
    if ic.struct_tune_max < 1 or ic.struct_retro_max < 0 or ic.struct_ladder_max_open < 1 or ic.struct_tune_beta < 0 \
            or not isinstance(
            ic.struct_tune_knobs, Mapping):
        raise ValueError("loop.ideation: struct_tune_max >= 1, struct_retro_max >= 0, struct_ladder_max_open >= 1, struct_tune_beta >= 0, "
                         "struct_tune_knobs a mapping {backbone | loss | other: ...}")
    if ic.struct_tune_flat_floor is not None and not (-1.0 < ic.struct_tune_flat_floor <= 0.0):
        raise ValueError("loop.ideation.struct_tune_flat_floor: an absolute composite gain in (-1, 0]")
    ps = cfg.alloc.prune_stale_family
    if ps is not None and (ps.age_h < 0 or ps.min_measured < 1 or ps.min_seeds < 1 or ps.max_per_tick < 0
                           or not math.isfinite(ps.max_best_delta)):
        raise ValueError("loop.alloc.prune_stale_family: age_h >= 0, min_measured >= 1, min_seeds >= 1, "
                         "max_per_tick >= 0, max_best_delta finite")
    mt = cfg.moonshot_tune
    if not (0 <= mt.p_lo <= mt.p_hi <= 1) or min(mt.max_trials_major, mt.max_trials_minor) < 1 \
            or mt.max_concurrent < 0 or any(f <= 0 or f == 1 for f in mt.shared.values()):
        raise ValueError("loop.moonshot_tune: 0 <= p_lo <= p_hi <= 1, max_trials_* >= 1, max_concurrent >= 0, "
                         "shared factors > 0 and != 1")
    cp = cfg.campaign
    if cp.budget_gpu_h <= 0 or cp.max_concurrent < 0 or cp.round1_points < 2 or min(cp.refine_points, cp.rounds,
                                                                                     cp.max_knobs) < 1 \
            or cp.round1_max < cp.round1_points or any(not isinstance(r, Mapping) or not r.get("key")
                                                       for r in (*cp.input_space, *(x for e in cp.incumbent for x in
                                                                                    e.get("search_space") or ()))) \
            or any(not e.get("tag") or not isinstance(e.get("levels"), Mapping) or not e["levels"]
                   for e in cp.incumbent):
        raise ValueError("loop.campaign: budget_gpu_h > 0, max_concurrent >= 0, round1_points >= 2, refine_points, "
                         "rounds and max_knobs >= 1, round1_max >= round1_points, input_space / search_space rows "
                         "{key, ...}, incumbent entries {tag, levels, search_space}")
    ex = cfg.extrapolate
    if min(ex.max_inflight, ex.edge_max, ex.min_z, ex.every_min) < 0 or ex.max_item_z <= 0 or not ex.mults \
            or min(ex.mults) <= 0 \
            or any(not isinstance(r, Mapping) or not r.get("key") or r.get("scale", "log") not in ("log", "linear")
                   for r in ex.knobs):
        raise ValueError("loop.extrapolate: max_inflight / edge_max / min_z / every_min >= 0, max_item_z > 0, mults > 0, "
                         "knobs rows {key, scale: log | linear, low?, high?}")
    db = cfg.databank
    if db.max_building < 0 or db.max_total_tb <= 0 or db.min_free_tb < 0 or db.retain_days <= 0 or db.num_workers < 0 \
            or any(not isinstance(r, Mapping) or not r.get("key") for r in db.adopt_space):
        raise ValueError("loop.databank: max_building >= 0, max_total_tb > 0, min_free_tb >= 0, retain_days > 0, "
                         "num_workers >= 0, adopt_space entries {key, low, high | factor, scale}")
    if any(not -1 <= getattr(cfg.meta, f.name) <= 1 for f in dataclasses.fields(cfg.meta)
           if f.name.endswith("_bias")) or cfg.meta.consolidate_every < 0:
        raise ValueError("loop.meta: every *_bias in [-1, 1], consolidate_every >= 0")
    cc = cfg.components
    if bad := sorted((set(cc.enabled) | set(cc.rates)) - set(C.OPTIONAL_COMPONENTS)):
        raise ValueError(f"loop.components: {bad[0]!r} is not an optional component {list(C.OPTIONAL_COMPONENTS)}"
                         + (" (protected: never ablated or disabled)" if bad[0] in C.PROTECTED_COMPONENTS else ""))
    if not all(0 <= r < 1 for r in (cc.ablation_rate, cc.down_rate, cc.reprobe_rate, *cc.rates.values())) \
            or cc.min_ablated < 1:
        raise ValueError("loop.components: every rate in [0, 1), min_ablated >= 1")
    aq = cfg.acquire
    if aq.target_mode not in ("auto", "incumbent", "target") or not 0 < aq.fresh_shrink <= 1 \
            or min(aq.gap_floor_frac, aq.stack_best_leg_frac, aq.stack_sum_frac, aq.idle_max_restarts) < 0 \
            or aq.untested_var_mult < 1 or aq.idle_max_run_frac <= 0 or aq.gap_level_min_test < 2 \
            or not 0 < cfg.status.max_seed_share <= 1 or cfg.status.min_seeds < 2 or cfg.status.share_min_n < 2 \
            or cfg.status.trim_min_n < 0 or cfg.status.share_k < 0:
        raise ValueError("loop.acquire / loop.status: target_mode auto | incumbent | target, fresh_shrink in (0, 1], "
                         "fracs and idle_max_restarts >= 0, untested_var_mult >= 1, idle_max_run_frac > 0, "
                         "gap_level_min_test >= 2, max_seed_share in (0, 1], min_seeds and share_min_n >= 2")
    if aq.gap_cap_sigmas < 0 or cfg.status.outlier_sd_d <= 0:
        raise ValueError("loop.acquire.gap_cap_sigmas >= 0 and loop.status.outlier_sd_d > 0")
    npc = cfg.noise_prior
    if npc.df < 0 or (npc.df > 0 and not (npc.panel_sd > 0 and all(v > 0 for v in npc.axis_sd.values()))):
        raise ValueError("loop.noise_prior: df >= 0; with df > 0 a positive panel_sd and axis_sd")
    if cfg.calibrate.pools not in ("auto", "spread") or cfg.calibrate.max_concurrent < 0:
        raise ValueError("loop.calibrate: pools auto | spread, max_concurrent >= 0")
    if cfg.novelty.weight < 0 or cfg.novelty.major_ratio <= 1:
        raise ValueError("loop.novelty: weight >= 0, major_ratio > 1")
    sc = cfg.scout
    if not (0 < sc.every_hours and 1 <= sc.min_ideas <= sc.max_ideas and sc.max_turns >= 1 and sc.timeout_min > 0):
        raise ValueError("loop.scout: every_hours > 0, 1 <= min_ideas <= max_ideas, max_turns >= 1, timeout_min > 0")
    eo, cap = cfg.evalonly, cfg.seeds.max_seeds_per_node
    if not (0 < eo.max_share <= 1 and min(eo.screen_ckpts, eo.confirm_ckpts, eo.min_models, eo.null_models) >= 1
            and eo.sigma_floor > 0 and eo.null_tol > 0 and not 0 < cap < eo.screen_ckpts + eo.confirm_ckpts
            and 0 < eo.harm_prob < 1 and min(eo.harm_tol.values(), default=0) >= 0 and eo.source_wait_s >= 0
            and eo.noise_min_df >= 1) \
            or cfg.prune.every_min <= 0 or cfg.prune.grace_h < 0 or cfg.prune.reserve_per_model < 0:
        raise ValueError("loop.evalonly: 0 < max_share <= 1, counts >= 1, floors > 0, screen + confirm checkpoints "
                         "<= seeds.max_seeds_per_node; loop.prune: every_min > 0, grace_h >= 0, reserve >= 0")
    if cfg.slurm.resume not in ("off", "calibrate", "on"):
        raise ValueError("loop.slurm.resume must be off, calibrate or on")
    names = [p.name for p in cfg.slurm.pools]
    if cfg.slurm.pools and ("slots" in block or len(set(names)) < len(names)
                            or min(p.slots for p in cfg.slurm.pools) < 1):
        raise ValueError("loop.slurm.pools: unique names, slots >= 1 each, and no loop.slots beside them")
    sp, cap = cfg.seeds, cfg.seeds.max_seeds_per_node
    for what, n in (("screen + promote + promotion.confirm_max_seeds",
                     len(sp.screen) + sp.n_promote * sp.promote_stage + cfg.promotion.confirm_max_seeds),
                    ("audit", len(sp.audit)), ("calibration", len(sp.calibration_set))):
        if cap > 0 and n > cap:
            raise ValueError(f"loop.seeds: {what} = {n} seeds exceeds max_seeds_per_node {cap}")
    fams = {f: dataclasses.replace(r, forbid=r.forbid + tuple(_word(t) for ref in r.forbid_config
                                                              for t in _config_terms(spec.repo, ref, f)))
            for f, r in cfg.families.items()}
    terms = [t for ref in cfg.forbid_config for t in _config_terms(spec.repo, ref, "surface")]
    cfg = dataclasses.replace(cfg, forbid=cfg.forbid + tuple(map(_word, terms)),
                              forbid_roots=tuple(t for t in terms if t.startswith("/")))
    for f, pats in [*((f, r.forbid) for f, r in fams.items()), ("surface", cfg.forbid)]:
        for pat in pats:
            try:
                re.compile(pat)
            except re.error as exc:
                raise ValueError(f"loop.families.{f}.forbid: bad regex {pat!r}: {exc}") from None
    return dataclasses.replace(cfg, families=fams, project=project, domain=domain), unmeasured


def _word(term: str) -> str:
    return rf"(?<![\w-]){re.escape(term)}(?![\w-])"


def _config_terms(repo: str, ref: str, fam: str) -> tuple[str, ...]:
    """Every list item, nested key and path under ``file:dotpath`` of the
    base clone's config (e.g. the probe block: dataset names and roots). Unreadable = a load error."""
    rel, _, dot = ref.partition(":")
    try:
        node = yaml.safe_load((Path(repo) / rel).read_text())
        for part in dot.split("."):
            node = node[part]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise ValueError(f"loop.families.{fam}.forbid_config: cannot read {ref!r} under {repo!r}: {exc}") from None
    out: set[str] = set()

    def walk(v: Any, depth: int) -> None:
        if isinstance(v, Mapping):
            for k, x in v.items():
                out.update([str(k)] if depth else [])
                walk(x, depth + 1)
        elif isinstance(v, list):
            out.update(str(x) for x in v if isinstance(x, (str, int)) and not isinstance(x, bool))
        elif isinstance(v, str) and "/" in v:
            out.add(v)
    walk(node, 0)
    return tuple(sorted(out))


def install_registry(cfg: LoopConfig) -> None:
    """The per-process project registry of ``cfg`` (no block = the nanopath defaults) and its root-seed noise floor:
    the measured PAIRED sds (loop.panel_sd / axis_sd, calibrate) as per-run sigma floors (/ sqrt 2). load_context and
    the CLIs' data.load_checked both install through here (one registry state per task)."""
    P.install(cfg.project, cfg.domain)
    if cfg.project is not None and cfg.project.noise_floor_seeds > 0:
        floor = {"panel": cfg.panel_sd / math.sqrt(2.0), "composite": cfg.panel_sd / math.sqrt(2.0),
                 **{a.name: float(cfg.axis_sd[a.key]) / math.sqrt(2.0) for a in cfg.project.axes}}
        P.set_noise_floor({k: v for k, v in floor.items() if math.isfinite(v) and v > 0})


def load_context(spec_path: str | Path, *, state_dir: str | Path | None = None,
                 require_measured: bool = True) -> LoopContext:
    """TaskSpec + parsed ``loop:`` block + a NoiseModel of THIS spec.

    ``require_measured=False`` is for ``calibrate`` only: UNMEASURED sigmas come back NaN.
    """
    spec = TaskSpec.load(spec_path)
    cfg, unmeasured = _parse(spec)
    install_registry(cfg)
    noise = NoiseModel.from_spec(spec)
    if isinstance(noise.seed_transfer_sd, Unmeasured):
        unmeasured.insert(0, "noise.seed_transfer_sd")
    if unmeasured and require_measured:
        raise C.UnmeasuredLoopNoise(
            f"{spec.name}: {', '.join(unmeasured)} {'is' if len(unmeasured) == 1 else 'are'} "
            f"{UNMEASURED}; the loop will not run on unmeasured noise. Run "
            f"`uv run python -m loop calibrate --task {spec_path}` first.")
    home = state_dir or cfg.state_dir or (Path(spec.home) / "loop" if spec.home else None)
    if not home:
        raise ValueError(f"{spec.name}: no state dir; pass one or set loop.state_dir")
    return LoopContext(spec=spec, cfg=cfg, noise=noise, task_id=spec.name,
                         spec_path=str(spec_path), state_dir=str(Path(home).expanduser()))


# ---------------------------------------------------------------------------------------------- v3: LOCKED / TUNABLE
# docs/loop-core-v3.md §7. The 3-hour meta-cycle (loop.core.meta) may change ONLY the TUNABLE keys, within their
# bounds, by writing state/loop_config/v<N>.yaml (and prompts under state/prompts/v<N>/). Everything else is LOCKED:
# the statistics (model / seeds / test / data), the store and ledger, this file, the acceptance judge (backtest) and
# the meta-cycle itself, the adapter's eval / probe surface locks, and every key of the task spec.

@dataclass(frozen=True)
class Tunable:
    default: float
    lo: float
    hi: float
    #: where the value takes effect: ``rank:<kwarg>[.<key>]`` (an acquire.rank keyword; ``.key``: a dict entry),
    #: ``costs:<field>`` (acquire.Costs) or ``<module>:<ATTR>`` (a loop.core constant read at call time)
    target: str
    integer: bool = False
    doc: str = ""


#: Only keys a held-out replay can MEASURE are tunable: a change nobody can measure is never accepted.
TUNABLE: Mapping[str, Tunable] = {
    "acquire.harm_q": Tunable(0.2, 0.05, 0.4, "rank:harm_q",
                              doc="a test is allowed only if P(robustness loss > harm_h_rob) <= harm_q"),
    "acquire.harm_h_rob": Tunable(0.008, 0.004, 0.016, "rank:harm_h.rob",
                                  doc="tolerated robustness loss (axis units) in the harm limit: catastrophic only "
                                      "(the composite already prices robustness at its weight)"),
    "acquire.port_p": Tunable(0.5, 0.05, 0.95, "loop.core.acquire:PORT_P",
                              doc="prior P(a port pays: grafts and screens positive); acquire.port_p updates it with "
                                  "the epoch's measured ports (PORT_P_N pseudo-ports)"),
    "acquire.max_of_k_beta": Tunable(0.0, 0.0, 2.0, "loop.core.acquire:MAX_OF_K_BETA",
                                     doc="max-of-K shrinkage of a test belief: m_sel = m - beta sqrt(v) E_K"),
    "acquire.test_p_min": Tunable(0.7, 0.6, 0.9, "loop.core.acquire:TEST_P_MIN",
                                  doc="test entry: m_sel >= MDE/2 or P(gain > 0) >= test_p_min (test VOLUME; the "
                                      "test's verdict stays LOCKED under test.)"),
    # fork-and-flush (loop.core.fnf): the A/B arm. Its share is adopted by the meta-cycle's A/B judgement (fresh-seed
    # credit per GPU-h vs the shared-context control, >= 2 bootstrap SE after >= 2 rounds), not by the replay
    "ideation.fnf_share": Tunable(0.3, 0.0, 0.7, "loop.core.fnf:SHARE",
                                  doc="share of the explore-lane GPU-h that goes to flushed fork-and-flush branches "
                                      "(0 = off)"),
    "ideation.fnf_every_gpu_h": Tunable(350.0, 100.0, 2000.0, "loop.core.fnf:EVERY_GPU_H",
                                        doc="GPU-h between fork-and-flush rounds (each branch's cap is 6% of it)"),
    "ideation.fnf_nn_sim": Tunable(0.55, 0.3, 0.9, "loop.core.fnf:NN_SIM_MAX",
                                   doc="a round starts early when the median nearest-neighbour similarity of the last "
                                       "48 h of new ideas exceeds this"),
}

#: files the meta process can never change; each tick compares their git-blob hashes with the deployed HEAD
LOCKED_FILES: tuple[str, ...] = (
    "loop/core/model.py", "loop/core/seeds.py", "loop/core/test.py", "loop/core/data.py", "loop/core/backtest.py",
    "loop/core/meta.py", "loop/store.py", "loop/spec.py", "loop/core/fnf.py",   # fnf: the arm's A/B judge
    "loop/project.py")                          # the project registry: axes, official weights, guard, composite
#: config prefixes that are LOCKED whatever a proposal says (the test's alpha / k, seed pools, official weights, ...)
LOCKED_KEYS: tuple[str, ...] = (
    "test.", "seeds.", "model.", "data.", "store.", "backtest.", "meta.", "costs.k_test", "costs.max_seeds",
    "loop.", "spec.", "slurm.", "labless.", "ledger")
#: the only places the meta process writes (relative to the state dir)
META_WRITABLE: tuple[str, ...] = ("loop_config/", "prompts/")
#: prompt files a meta proposal may patch (rendered through loop.agents.briefs, so an A/B arm can serve them)
META_PROMPTS: tuple[str, ...] = ("researcher.md", "ideation.md")


def tunable_values(values: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Every TUNABLE key at its default, overridden by ``values`` (an unknown key raises)."""
    out = {k: t.default for k, t in TUNABLE.items()}
    for k, v in dict(values or {}).items():
        if k not in TUNABLE:
            raise ValueError(f"{k}: not a TUNABLE key")
        out[k] = v
    return out


#: TUNABLE keys that exist only for a project with a guard axis (nanopath: rob)
GUARD_TUNABLES: tuple[str, ...] = ("acquire.harm_h_rob",)


def visible_tunables() -> dict[str, Tunable]:
    """The TUNABLE keys the meta-cycle shows and may change for the installed project: without a guard axis
    (loop.project.guard []) the guard's harm limit does not exist, so ``acquire.harm_h_rob`` is hidden."""
    return {k: t for k, t in TUNABLE.items() if P.guard() is not None or k not in GUARD_TUNABLES}


def check_tunable(key: str, value: Any) -> str:
    """'' if ``key = value`` is an allowed meta change, else the reason it is refused."""
    if key in GUARD_TUNABLES and P.guard() is None:
        return f"{key}: the project has no guard axis (loop.project.guard is empty)"
    if key in TUNABLE:
        t = TUNABLE[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return f"{key}: value {value!r} is not a number"
        if t.integer and value != int(value):
            return f"{key}: must be an integer"
        if not t.lo <= value <= t.hi:
            return f"{key}: {value} outside the schema bounds [{t.lo}, {t.hi}]"
        return ""
    if any(key == p.rstrip(".") or key.startswith(p) for p in LOCKED_KEYS):
        return f"{key}: LOCKED"
    return f"{key}: not a TUNABLE key"
