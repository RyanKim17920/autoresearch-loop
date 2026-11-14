"""loop.core.acquire — one ranking rule for every GPU-h (docs/loop-core-v3.md §3).

Every possible action is an :class:`Option` -- a Gaussian belief (m, v) about an effect in composite units
relative to the incumbent, the sampling variance V of what the action would observe, and its cost -- and is
valued in one unit: expected gain in the true composite of the final incumbent per GPU-h.

- ``seed`` (more data on an existing candidate): the knowledge gradient ``σ̃ f(-|m - h|/σ̃)``, the expected
  improvement of the decision "which candidate goes to test" over ``h`` = the best OTHER candidate (or 0).
- ``test`` / ``idea`` / ``wave`` (create a new contender or realise a switch): the expected improvement
  ``σ̃ f((m - h)/σ̃)`` over h, ``× p``: h = 0 for a test (the incumbent); for an idea / wave h = the best
  selection-shrunk test belief ``m_sel`` (>= 0) -- what it must beat to win the next test slot. (Against the
  best RAW candidate mean, as first designed, ideas were worth ~0 on the 10-08 ledger: λ ≈ 4e-6/GPU-h.)
``σ̃ = v/√(v+V)``, ``f(z) = zΦ(z) + φ(z)``.

Harm limit (catastrophic only): a ``test`` is allowed only if ``P(δ_rob < -h_rob) <= q`` (h_rob .008: twice the
robustness loss a 5-seed test resolves). The composite already prices every axis at its weight, so nothing else is
guarded and no other action kind is checked (a seed is how a harm uncertainty resolves; a wave's x* is tested later).

Value floors (fix batch B, 10-12): a GPU action is launched only if its value/GPU-h is at least ``VALUE_FLOOR`` x the
best ``screen`` / ``idea`` value/GPU-h this tick (the 1,898-port tail and tests of negative candidates are worth ~0 but
were still launched whenever a slot was free); a ``wave`` at least ``WAVE_FLOOR`` x the best screen's, and waves hold at
most ``WAVE_SHARE`` of the GPU slots unless nothing else wants them (soft cap). At most ``PORT_OPEN_MAX`` ports are
being screened at once. A test also needs ``status.test_entry_ok`` (fix A: m_sel >= MDE/2 or P(gain > 0) >= .7
against the change class σ_d; PASSED nodes that never switched are re-tested: ``status.retest_candidates``). Two capacities, ranked apart: GPU slots go to runs only (every
positive-value GPU action, best value/GPU-h first, while slots last); LLM slots go to ideas only -- whenever an LLM
slot is free (``llm_free``, the researcher's ``max_concurrent``: ``llm_parallel``) the best-valued lane's brief starts,
round-robin over the lanes in value order. An idea is valued from the UPPER tail of its lane prior (Student-t, ν = 3,
the prior the posterior uses: :func:`ei_t`), not its mean -- every real gain came from an LLM idea, rare and large.

Base epochs (docs §8): after a base change the ``rebase`` action (new root + the incumbent's port, k seeds each) is
the only option until it is launched; an old-epoch candidate is a ``port`` option (graft onto the new base, then
screen: valued like an idea, its old posterior as the belief, ``p`` = the graft success rate), never a seed / test.

``wave`` (one per carrier with a knob bank and no open wave): a factorial over the carrier's best knobs
(``core.factorial.choose``), valued by EI over the bar of its x*'s expected gain (``core.factorial.forecast``), costing
its N runs; the tick designs, materialises and launches it.

``screen``: the first screening runs of a QUEUED node of the current base (an intake idea, a researcher port, an x*):
valued like an idea whose LLM cost is already paid -- belief = an x*'s own prior, a port's original's posterior, else
its prior cell -- so a made node is screened before a new session is asked for.

Idle GPUs (v3-c5, operator 10-13 "never idle while USEFUL work exists"): an idle slot goes, in order, to (1) the ranked
GPU actions above (stacking waves included: the tick launches its own stacking waves ahead of relax / fill), (2) the
relax pass (:func:`rank`: below-floor positive-value seeds / screens; NO relax wave by default, ``RELAX_MAX_WAVES`` 0),
(3) :func:`idle_fill`, then (4) more LLM research sessions (``runner.Research.limit``: up to ``ideation.llm_slots_max``
while slots would otherwise idle). In target mode a seed under the seed floor is DROPPED from relax and fill
(``FILL_DROP_BELOW``), not merely ordered last: such a node cannot reach a test against the gap, so its runs have no
decision value (live epoch 065b87f: 251 fill GPU-h, 24% lost to preemption, 110 of 127 filled nodes ended at a
posterior <= 0 over the incumbent, none above +.0018 vs a +.0074 gap). Fill without decision value is not useful
work: when nothing clears the floors a few GPU slots are LEFT IDLE on purpose (the research sessions take the
slack instead)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Collection, Iterable, Literal, Mapping, Sequence

from loop import project as P
from loop.core import model as M
from loop.core.backtest import fit_at
from loop.core.data import LANES, ROOT, TEST, Ledger
from loop.core.model import _Phi, _phi
from loop.core.seeds import SeedBlock, screening_obs
from loop.core.status import Status

if False:  # pragma: no cover - typing only
    from loop.core.factorial import BankKnob

Kind = Literal["idea", "seed", "screen", "wave", "test", "port", "rebase", "fill"]
#: axis -> h_a (tolerated loss, axis units); q = the allowed probability of exceeding it. Catastrophic only: .008 is
#: about 2x the robustness noise sd (.0039-.0047) -- a loss the panel weight alone would not price as a disaster
HARM_H = P.HARM_H                  # live (loop.project; nanopath: {"rob": 0.008})
HARM_Q = 0.2
#: a GPU action's value/GPU-h must reach this fraction of the best screen / idea value/GPU-h of the tick
VALUE_FLOOR = 0.25
#: a wave's value/GPU-h must reach this fraction of the best screen's (the best idea's when no screen is offered)
WAVE_FLOOR = 0.5
#: soft cap: wave runs in flight + launched <= this share of the GPU slots, unless slots would otherwise stay idle
WAVE_SHARE = 0.25
#: ports being screened (a port node of the current epoch without its screening runs finished) at once
PORT_OPEN_MAX = 6
#: an old-epoch node is ported only with at least this many own seeds and a positive selection-shrunk belief
PORT_MIN_SEEDS = 2
#: fill the GPU slots no ranked action takes with information runs (:func:`idle_fill`; preemptible-only, launched last)
IDLE_FILL = True
#: the relax pass (10-13 review: never leave GPU slots idle while useful work exists): after the ranked selection,
#: free slots go to positive-value seed / wave / screen actions under the value / gap / seed floors, best value per
#: GPU-h first, preemptible pools only (``acquire.relax_fill``)
RELAX_FILL = True
RELAX_KINDS = frozenset({"seed", "wave", "screen"})
#: ``rank(prefer=)``: the action kinds placed AHEAD of the preferred screens (owed tests, in-flight seed extensions, a
#: re-baseline)
PREFER_AHEAD = frozenset({"test", "seed", "rebase"})
#: at most this many OPEN relax (below-floor, preemptible) waves at once (``acquire.relax_max_waves``; 10-13: three
#: relax waves held 24 slots and leaked onto primary; v3-c5: 1 -> 0 -- a below-gap-floor wave has no decision value
#: and its preempted / churn-cancelled points left zombie waves)
RELAX_MAX_WAVES = 0
#: target mode: a seed under the seed floor (``Option.below``) is DROPPED from the relax pass and from idle fill (b)
#: (``acquire.fill_drop_below``; False = the old rule: relax-only / ordered last). Idle slots are left idle instead
FILL_DROP_BELOW = True
#: waves whose result is recorded (``wave:<id>:result`` state): closed even with an unfinished point (the tick sets
#: it each tick, matching its ``_open_waves``; ``factorial.open_waves(closed=...)``)
CLOSED_WAVES: frozenset[str] = frozenset()
#: at most this many open waves in total, ranked or relax (``acquire.max_open_waves``): a hard cap on top of the
#: one-open-wave-per-carrier rule and the WAVE_SHARE soft cap (which the idle-slot pass lifts)
MAX_OPEN_WAVES = 3
PROGRAM_ARM_SHARE = 0.3     # share of the free slots reserved for QUEUED program-arm nodes (``acquire.program_arm_share``)
QUEUED_MAX_AGE_H = 96.0     # QUEUED, never-run nodes older than this are aged out (``acquire.queued_max_age_h``)
CONTROL_BAR: tuple[float, float, int] | None = None   # (z, share, min siblings): ``acquire.control_bar`` (11-08)
AGED_REASON = "dropped_backlog: aged_out_queued"   # the REJECTED reason ``alloc.drop`` writes too (``dropped_backlog: ...``)
PROGRAM_TAG = "program_arm=1"
LOCAL_TAG = "local_confirm="   # an operator manual run's confirm node (loop.localruns.build_confirms): run on purpose

# -- target-aware acquisition (docs §8 "Target-aware acquisition"; operator keys ``loop.acquire`` via configure) ----
#: "incumbent": every value over the incumbent (v3 as designed); "target": over the GAP to the external target
#: (the local mean needed for P(Labless median of 3 >= leader_median + promote_margin) >= promote_p, core.bar);
#: "auto": target when one is configured
TARGET_MODE = "auto"
#: fresh / selected effect ratio (ledger 10-12: a screened effect replicates at ~.55 on fresh seeds)
FRESH_SHRINK = 0.55
#: target mode: a wave / port is dropped when its upper-90% effect (FRESH_SHRINK x the mean only for a raw estimate)
#: < this x the gap. Seed extensions are exempt: their value is information toward a test (a node stuck below the
#: test's seeds would never reach one)
GAP_FLOOR_FRAC = 0.5
#: ... but a seed extension (and an idle-fill seed, ``idle_fill`` (b)) still needs its upper-90% effect over the
#: incumbent >= this x the gap (``acquire.seed_floor_frac``): a node that cannot reach a test against the incumbent
#: is not worth seeds toward one (0 = off)
SEED_FLOOR_FRAC = 0.25
#: FUTILITY STOP (11-09 nanobrain: 4 flat nodes took ~30 screen runs while 15+ fresh ideas waited for a first screen;
#: incumbent mode has no gap floors, and a delta ~0 node keeps P(gain > 0) ~ .5, i.e. maximal "resolve the uncertain"
#: value): a SCREENING node with >= FUTILITY_MIN_SEEDS done seeds whose posterior P(true gain over the incumbent >
#: the futility bar) < FUTILITY_P gets no more seed extensions and no idle-fill (b) seeds -- breadth over depth.
#: Bar: FUTILITY_GAP_FRAC x the gap to the external target when one is measured (in either target mode), else
#: FUTILITY_DELTA. A test stays gated by its own entry rule. ``acquire.futility_*``; FUTILITY_P 0 = off (nanopath)
FUTILITY_P = 0.0
FUTILITY_MIN_SEEDS = 3
FUTILITY_GAP_FRAC = 0.33
FUTILITY_DELTA = 0.0
#: the operator's per-node seed cap (``seeds.max_seeds_per_node``; 0 = only ``Costs.max_seeds``), installed by configure
SEED_CAP = 0
Z90 = 1.2816                    # one-sided 90% upper bound
#: stacks are sub-additive (ledger 10-12: median stack / sum-of-legs .40, stack / best-leg .74, stack ~ .24 sum of
#: legs + .0013, 0% beat the sum): a stack's prior mean = min(sum, max(BEST_LEG x best, SUM x sum + INTERCEPT)) over
#: its shrunk legs, legs of one mechanism (shared leg beyond the base / specific family) counted once (the best)
STACK_BEST_LEG_FRAC = 0.74
STACK_SUM_FRAC = 0.24
STACK_INTERCEPT = 0.0013
#: target mode, gap > the best shrunk available candidate: an idea lane with no fresh test on this base gets its
#: prior variance x this (its tail is unmeasured: the only route across a gap no material in hand can close)
UNTESTED_VAR_MULT = 2.0
#: idle fill (loop.runner / idle_fill): a fill run preempted this often moves to a free primary slot or is dropped;
#: nodes whose runs take more than this x the median run are not used for idle fill (shorter runs survive preemption)
IDLE_MAX_RESTARTS = 2
IDLE_MAX_RUN_FRAC = 1.5
#: families too generic to mean "one mechanism" (wave points / x*, the base roots, the broad taxonomy classes:
#: ``acquire.generic_families``)
GENERIC_FAMILIES = frozenset({"", "design", "baseline", "training", "augmentation", "objective", "optimization",
                              "architecture"})
#: the gap's incumbent level comes from its TEST-pool seeds (unbiased: never selected on) when it has this many
GAP_LEVEL_MIN_TEST = 4
#: idea options: lane -> the lane whose prior values its sessions ('' = no idea option). A lane whose sessions get
#: another lane's brief (ideation.LANE_KIND: factorial -> explore) must not buy them with its own prior
#: (``acquire.idea_lane_prior``; 10-13: the factorial lane bought 45/65 explore sessions with its design prior)
IDEA_LANE_PRIOR: Mapping[str, str] = {"factorial": "explore"}
#: target mode: the gap is capped at this many per-run panel sigmas (``acquire.gap_cap_sigmas``; 0 = off): a bar
#: 10+ sigma away would otherwise drop every wave / port / seed under the gap floors and rank by variance alone
GAP_CAP_SIGMAS = 0.0
#: target-mode hysteresis (:func:`target_active`): the mode in force this tick, set (and persisted in tick state) by
#: ``loop.tick.ModuleCore``; None = no state (target mode whenever the gap is positive)
TARGET_ACTIVE: bool | None = None


@dataclass(frozen=True)
class Target:
    """The external target (``core.bar``, Labless's acceptance rule): ``score`` (labless.leader_median, the leader's
    median on Labless's scale; back-compat labless.bar_score) and/or ``node`` (labless.bar_node: its own local mean
    when no score), plus ``margin`` (labless.promote_margin; back-compat bar_margin). ``p``: the required P(median of
    3 fresh seeds >= score + margin) (labless.promote_p; .5 = the target median itself), ``offset`` (labless.offset,
    0 = no local-vs-Labless credit), ``min_test`` the planning seed count, ``prior_df`` the σ blend's pseudo-df."""
    score: float | None
    margin: float = 0.0
    node: str = ""
    p: float = 0.5
    offset: float = 0.0
    min_test: int = 4
    prior_df: float = 8.0


TARGET: Target | None = None


def target_of(labless: Any) -> Target | None:
    """The target the ``labless`` config defines (None: no leader median / bar_score and no bar_node)."""
    from loop.core import bar as BR
    if labless is None:
        return None
    score, margin = BR.leader_margin(labless)
    node = getattr(labless, "bar_node", "") or ""
    if score is None and not node:
        return None
    return Target(score, margin, node, float(getattr(labless, "promote_p", BR.PROMOTE_P)),
                  float(getattr(labless, "offset", 0.0) or 0.0),
                  int(getattr(labless, "min_test_seeds", BR.MIN_TEST_SEEDS)),
                  float(getattr(labless, "sigma_prior_df", BR.SIGMA_PRIOR_DF)))


def configure(cfg: Any) -> None:
    """Install the operator keys (``loop.acquire``, ``loop.status``) and the target (``loop.labless``) as this
    module's constants (read at call time). A config without a section keeps the values in force."""
    global TARGET_MODE, FRESH_SHRINK, GAP_FLOOR_FRAC, STACK_BEST_LEG_FRAC, STACK_SUM_FRAC, STACK_INTERCEPT, \
        UNTESTED_VAR_MULT, IDLE_MAX_RESTARTS, IDLE_MAX_RUN_FRAC, TARGET, GENERIC_FAMILIES, GAP_LEVEL_MIN_TEST, \
        SEED_FLOOR_FRAC, IDEA_LANE_PRIOR, RELAX_FILL, RELAX_MAX_WAVES, MAX_OPEN_WAVES, FILL_DROP_BELOW, \
        PROGRAM_ARM_SHARE, QUEUED_MAX_AGE_H, STRUCT_BETA, GAP_CAP_SIGMAS, CONTROL_BAR, FUTILITY_P, \
        FUTILITY_MIN_SEEDS, FUTILITY_GAP_FRAC, FUTILITY_DELTA, SEED_CAP
    from loop.core import status as ST
    if (sp := getattr(cfg, "seeds", None)) is not None:
        SEED_CAP = int(getattr(sp, "max_seeds_per_node", SEED_CAP) or 0)
    aq = getattr(cfg, "acquire", None)
    if aq is not None:
        FUTILITY_P = float(getattr(aq, "futility_p", FUTILITY_P) or 0.0)
        FUTILITY_MIN_SEEDS = int(getattr(aq, "futility_min_seeds", FUTILITY_MIN_SEEDS))
        FUTILITY_GAP_FRAC = float(getattr(aq, "futility_gap_frac", FUTILITY_GAP_FRAC))
        FUTILITY_DELTA = float(getattr(aq, "futility_delta", FUTILITY_DELTA))
        TARGET_MODE, FRESH_SHRINK, GAP_FLOOR_FRAC = aq.target_mode, aq.fresh_shrink, aq.gap_floor_frac
        STACK_BEST_LEG_FRAC, STACK_SUM_FRAC, STACK_INTERCEPT = aq.stack_best_leg_frac, aq.stack_sum_frac, \
            aq.stack_intercept
        UNTESTED_VAR_MULT, IDLE_MAX_RESTARTS, IDLE_MAX_RUN_FRAC = aq.untested_var_mult, aq.idle_max_restarts, \
            aq.idle_max_run_frac
        GENERIC_FAMILIES = frozenset(getattr(aq, "generic_families", GENERIC_FAMILIES))
        GAP_LEVEL_MIN_TEST = getattr(aq, "gap_level_min_test", GAP_LEVEL_MIN_TEST)
        SEED_FLOOR_FRAC = float(getattr(aq, "seed_floor_frac", SEED_FLOOR_FRAC))
        IDEA_LANE_PRIOR = dict(getattr(aq, "idea_lane_prior", IDEA_LANE_PRIOR))
        RELAX_FILL = bool(getattr(aq, "relax_fill", RELAX_FILL))
        RELAX_MAX_WAVES = int(getattr(aq, "relax_max_waves", RELAX_MAX_WAVES))
        FILL_DROP_BELOW = bool(getattr(aq, "fill_drop_below", FILL_DROP_BELOW))
        MAX_OPEN_WAVES = int(getattr(aq, "max_open_waves", MAX_OPEN_WAVES))
        PROGRAM_ARM_SHARE = float(getattr(aq, "program_arm_share", PROGRAM_ARM_SHARE))
        QUEUED_MAX_AGE_H = float(getattr(aq, "queued_max_age_h", QUEUED_MAX_AGE_H))
        GAP_CAP_SIGMAS = float(getattr(aq, "gap_cap_sigmas", GAP_CAP_SIGMAS) or 0.0)
        CONTROL_BAR = ((float(aq.control_bar_z), float(aq.control_bar_share), int(aq.control_bar_min_siblings))
                       if getattr(aq, "control_bar", False) else None)
    if (ic := getattr(cfg, "ideation", None)) is not None:
        STRUCT_BETA = float(getattr(ic, "struct_tune_beta", STRUCT_BETA))
    if (sg := getattr(cfg, "status", None)) is not None:
        ST.MAX_SEED_SHARE, ST.SEED_GATE_MIN = sg.max_seed_share, sg.min_seeds
        ST.SHARE_MIN_N = getattr(sg, "share_min_n", ST.SHARE_MIN_N)
        ST.TRIM_MIN_N = int(getattr(sg, "trim_min_n", ST.TRIM_MIN_N))
        ST.SHARE_K = float(getattr(sg, "share_k", ST.SHARE_K))
        ST.OUTLIER_Z = float(getattr(sg, "outlier_z", ST.OUTLIER_Z))
        ST.TRIM_FRAC = float(getattr(sg, "trim_frac", ST.TRIM_FRAC))
        ST.OUTLIER_SD_D = float(getattr(sg, "outlier_sd_d", ST.OUTLIER_SD_D))
    if (tc := getattr(cfg, "test", None)) is not None:
        from loop.core import test as T
        T.configure(cfg)
        T.CATA_SIGMAS = float(getattr(getattr(cfg, "veto", None), "catastrophic_sigmas", T.CATA_SIGMAS) or 0.0)
        ST.SD_POOL = getattr(tc, "sd_pool", ST.SD_POOL)
    TARGET = target_of(getattr(cfg, "labless", None))


def seed_cap(costs: "Costs") -> int:
    """A node's selection-seed cap: ``costs.max_seeds`` within the operator's ``seeds.max_seeds_per_node``."""
    return min(costs.max_seeds, SEED_CAP) if SEED_CAP > 0 else costs.max_seeds


def screen_seeds(runs: Iterable[Any], node: str | None = None) -> dict[str, set[int]]:
    """{node: DISTINCT seeds of its live-or-done SCREEN-stage runs}: what the seed cap counts. A requeue / next
    attempt of a seed is the same seed; a crashed / voided run (an outage kill) holds no seed. In-flight runs DO count
    (11-08 nanobrain: the seed option counted only scored seeds, so a node with 3 done + 3 running got a 7th, 8th ...)."""
    out: dict[str, set[int]] = {}
    for r in runs:
        if (node is None or r.node == node) and r.stage == "screen" and r.status not in DEAD:
            out.setdefault(r.node, set()).add(r.seed)
    return out


def futility_bar(led: Ledger, incumbent: str) -> float:
    """The futility rule's bar over the incumbent: FUTILITY_GAP_FRAC x the measured gap (> 0), else FUTILITY_DELTA."""
    try:
        gap = gap_of(led, incumbent)
    except Exception:  # noqa: BLE001 - no gap: the absolute bar
        gap = None
    return FUTILITY_GAP_FRAC * gap if gap is not None and gap > 0 else FUTILITY_DELTA


#: a struct_tune / program ladder point counts as finished only at this many done seeds (struct_tune.ladder_state,
#: ``struct_min_seeds``): futility never stops it below that, or its ladder would wait out struct_hold_h (11-10 review)
LADDER_FUTILITY_MIN = 3


def futility_exempt(node: Any, n: int) -> bool:
    """A ladder point (``struct_tune=`` / program ``ladder=`` child) with fewer than LADDER_FUTILITY_MIN done seeds. A
    point a moonshot rescue CLOSED (``rescue_closed=1``, loop.core.rescue) is exempt from nothing."""
    notes = f" {getattr(node, 'notes', '') or ''}"
    return n < LADDER_FUTILITY_MIN and " rescue_closed=1" not in notes \
        and (" struct_tune=" in notes or bool(LADDER_RE.search(notes)))


RESCUE_RE = re.compile(r"(?:^|\s)moonshot_rescue=")
RESCUE_K_RE = re.compile(r"(?:^|\s)rescue_best_of=(\d+)")


def futile(m: float, v: float, n: int, bar: float) -> float | None:
    """P(gain > ``bar``) under N(m, v) when the futility rule stops the node (it is < FUTILITY_P with >=
    FUTILITY_MIN_SEEDS done seeds), else None. Off when FUTILITY_P <= 0."""
    if FUTILITY_P <= 0 or n < FUTILITY_MIN_SEEDS:
        return None
    sd = math.sqrt(max(v, 0.0))
    p = (1.0 if m > bar else 0.0) if sd <= 0 else 0.5 * math.erfc((bar - m) / (sd * math.sqrt(2.0)))
    return p if p < FUTILITY_P else None


def target_on() -> bool:
    return TARGET is not None and TARGET_MODE in ("target", "auto")


def gap_se(led: Ledger, incumbent: str, target: Target | None = None) -> tuple[float, float] | None:
    """(signed gap, its se), DIRECT (``core.bar``, no root terms): the LOCAL MEAN NEEDED to promote minus the
    incumbent's level on our infra (official panel, current epoch). Needed mean (``core.bar.need_mean``): the mean m
    at which P(median of 3 fresh seeds >= target median) = ``target.p`` for a candidate with the epoch's pooled
    per-run σ measured on ``target.min_test`` fresh seeds (μ ~ N(m + offset, σ²/n)), i.e. target median + a z_p-like
    adjustment for the median-of-3 spread and the mean's uncertainty (p = .7, σ .0028, n 4: ≈ +.0012); at p = .5 it
    is the target median itself. The incumbent's level comes from its TEST-pool seeds (fresh-test controls: never
    selected on) when it has >= ``GAP_LEVEL_MIN_TEST`` of them, else from all its seeds. The target median is
    ``target.score`` + margin when set, else ``target.node``'s own mean + margin. None: no target, or the incumbent /
    bar not measured."""
    from loop.core import bar as BR
    t = TARGET if target is None else target
    if t is None or not (t.node or t.score is not None):
        return None
    obs = led.in_epoch().by_node().get(led.port(incumbent) or incumbent, ())
    test = [o for o in obs if BR.fresh_test(o)]          # v3 review K8: TEST-stage runs, never a screen
    lv = BR.level(test if len({o.seed for o in test}) >= GAP_LEVEL_MIN_TEST else obs)
    bar = BR.bar_of(led, t.node, t.score, 0.0, t.margin, p=t.p, offset=t.offset, min_test=t.min_test,
                    prior_df=t.prior_df)
    if lv is None or bar is None:
        return None
    need = bar.target - bar.offset
    if bar.p != 0.5:
        try:
            need = BR.plan_need(bar, BR.epoch_sigma(led))
        except (ValueError, ZeroDivisionError):   # no noise fit yet: the target median itself
            pass
    se = lv.se if math.isfinite(lv.se) else 0.0
    return need - lv.mean, se


def target_active(gap: float, se: float, was: bool | None) -> bool:
    """Target mode with hysteresis: entered when the gap exceeds its se, left when it falls below -se (``was``: the
    mode of the previous tick; None: no state, on whenever the gap is positive)."""
    if was is None:
        return gap > 0
    return gap > -se if was else gap > se


def gap_of(led: Ledger, incumbent: str, target: Target | None = None) -> float | None:
    """The gap (:func:`gap_se`) when target mode is in force (``TARGET_ACTIVE``, the tick's hysteresis state; no
    state: a positive gap) and positive; else None (the acquisition then values over the incumbent). With
    ``GAP_CAP_SIGMAS`` > 0 the gap is capped at that many per-run panel sigmas (``core.bar.safe_sigma``; a nan sigma
    leaves it uncapped)."""
    got = gap_se(led, incumbent, target)
    if got is None:
        return None
    on = got[0] > 0 if TARGET_ACTIVE is None else TARGET_ACTIVE
    if not (on and got[0] > 0):
        return None
    if GAP_CAP_SIGMAS > 0:
        from loop.core import bar as BR
        sd = BR.safe_sigma(led)
        if math.isfinite(sd) and sd > 0:
            return min(got[0], GAP_CAP_SIGMAS * sd)
    return got[0]


def legs_of(led: Ledger, nid: str, _seen: set[str] | None = None) -> frozenset[str]:
    """The idea legs a node carries: its explore / exploit-lane ancestors (itself included) through both parents, a
    port resolved to its original; base roots, crossover and factorial nodes are combinations, not legs."""
    seen = set() if _seen is None else _seen
    n = led.nodes.get(nid)
    if n is None or nid in seen or n.parent is None:
        return frozenset()
    seen.add(nid)
    if n.port_of:
        return legs_of(led, n.port_of, seen)
    out = {nid} if n.lane in ("explore", "exploit") else set()
    for p in (n.parent, n.second_parent):
        if p:
            out |= legs_of(led, p, seen)
    return frozenset(out)


def same_mechanism(led: Ledger, a: str, b: str, ref: str | None = None) -> bool:
    """One mechanism: the two share an idea leg (beyond the legs of ``ref``, what both are stacked on), or a specific
    family (not ``GENERIC_FAMILIES``). (A shared dominant gaining axis is NOT one mechanism: it merged ~22% of
    unrelated pairs.)"""
    na, nb = led.nodes.get(a), led.nodes.get(b)
    if na is None or nb is None:
        return False
    common = legs_of(led, ref) if ref else frozenset()
    if (legs_of(led, a) - common) & (legs_of(led, b) - common):
        return True
    return na.family == nb.family and na.family not in GENERIC_FAMILIES


def stack_mean(legs: Sequence[tuple[float, str]]) -> float:
    """A stack's prior mean from its (shrunk leg effect, mechanism id) legs: one mechanism counts once (its best leg:
    no credit for the 2nd), then min(sum, max(STACK_BEST_LEG_FRAC x best, STACK_SUM_FRAC x sum + STACK_INTERCEPT));
    a one-mechanism "stack" is its best leg."""
    best: dict[str, float] = {}
    for m, g in legs:
        best[g] = max(best.get(g, -math.inf), m)
    xs = list(best.values())
    if not xs:
        return 0.0
    if len(xs) < 2:                         # one mechanism: its best leg, no stacking discount
        return xs[0]
    top, tot = max(xs), sum(xs)
    return min(tot, max(STACK_BEST_LEG_FRAC * top, STACK_SUM_FRAC * tot + STACK_INTERCEPT))


def adds_mechanism(led: Ledger, base: str, add: str) -> bool:
    """Does stacking ``add`` onto ``base`` add a NEW mechanism? No when every idea leg of ``add`` is already in
    ``base``, or when ``add``'s family is ``base``'s or one of its legs' (specific families only)."""
    new = legs_of(led, add) - legs_of(led, base)
    if not new or add not in led.nodes:
        return False
    fams = {led.nodes[x].family for x in (base, *legs_of(led, base)) if x in led.nodes} - GENERIC_FAMILIES
    return led.nodes[add].family not in fams


def mechanisms(led: Ledger, nids: Sequence[str], ref: str | None = None) -> dict[str, str]:
    """node -> a mechanism id (union over legs shared beyond ``ref``'s / a specific family)."""
    root = {n: n for n in nids}

    def find(x: str) -> str:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x
    for i, a in enumerate(nids):
        for b in nids[i + 1:]:
            if same_mechanism(led, a, b, ref=ref):
                root[find(a)] = find(b)
    return {n: find(n) for n in nids}


@dataclass(frozen=True)
class Action:
    kind: Kind
    node: str | None
    lane: str
    gpu_h: float
    value: float
    runs: int = 0               # GPU slots it takes now (an idea takes an LLM slot; its runs come later)
    why: str = ""
    below: bool = False         # under a target-mode floor (seed / wave): only :func:`rank`'s relax pass takes it
    pre: bool = False           # launched by the relax pass: preemptible pools only (``RunBook.fill_only``)

    @property
    def rate(self) -> float:
        return self.value / self.gpu_h if self.gpu_h > 0 else math.inf


@dataclass(frozen=True)
class Option:
    kind: Kind
    node: str | None
    lane: str
    m: float                    # belief about the effect vs the incumbent (composite; m_sel for a test)
    v: float
    V: float                    # sampling variance of what the action observes
    gpu_h: float
    runs: int
    p: float = 1.0              # an idea: P(the session yields a runnable node)
    axes: Mapping[str, tuple[float, float]] = field(default_factory=dict)   # axis -> (m, v): the harm check
    tail: bool = False          # valued from the heavy (t3) tail of its prior (an idea, a new idea node's screen)
    h: float | None = None      # target mode / one mechanism: the least it must beat (the gap, its group's leader)
    group: str = ""             # mechanism id of a stack option (ports, crossover screens): only the leader is credited
    raw: bool = False           # m is a raw (selected, unshrunk) estimate: the gap floor shrinks it (FRESH_SHRINK)
    below: bool = False         # under the gap / seed floor (target mode): kept for the relax pass only


def f(z: float) -> float:
    return z * _Phi(z) + _phi(z)


def sigma_tilde(v: float, V: float) -> float:
    return v / math.sqrt(v + V) if v > 0 else 0.0


def kg(m: float, v: float, V: float, h: float) -> float:
    """Knowledge gradient of one more observation of a candidate against the best alternative h."""
    s = sigma_tilde(v, V)
    return s * f(-abs(m - h) / s) if s > 0 else 0.0


def ei(m: float, v: float, V: float, h: float) -> float:
    """Expected improvement over h of a contender observed with variance V: E[max(m' - h, 0)]."""
    s = sigma_tilde(v, V)
    return s * f((m - h) / s) if s > 0 else max(m - h, 0.0)


NU = 3.0                        # the lane prior's tail (model.posterior's Student-t)


def _t3_pdf(t: float) -> float:
    return 6 * math.sqrt(3) / (math.pi * (3 + t * t) ** 2)


def _t3_cdf(t: float) -> float:
    return 0.5 + (math.atan(t / math.sqrt(3)) + math.sqrt(3) * t / (3 + t * t)) / math.pi


def ei_t(m: float, v: float, V: float, h: float) -> float:
    """``ei`` with a Student-t (ν = 3) belief of scale σ̃: ``E[max(m + σ̃ T - h, 0)]`` = σ̃ (z F(z) + (ν + z²)/(ν - 1) f(z)),
    z = (m - h)/σ̃ -- the upper tail of a heavy-tailed prior, so a lane whose mean is negative still has value."""
    s = sigma_tilde(v, V)
    if s <= 0:
        return max(m - h, 0.0)
    z = (m - h) / s
    return s * (z * _t3_cdf(z) + (NU + z * z) / (NU - 1) * _t3_pdf(z))


def harm_ok(axes: Mapping[str, tuple[float, float]], h: Mapping[str, float] = HARM_H, q: float = HARM_Q) -> bool:
    return all(_Phi((-h[a] - m) / math.sqrt(v)) <= q for a, (m, v) in axes.items() if a in h and v > 0)


def value(o: Option, best: float, second: float, bar: float) -> float:
    """``best`` / ``second``: the two highest candidate means (a seed's KG competes with the best other);
    ``bar``: the best selection-shrunk test belief (>= 0) -- what a new idea / wave must beat to win the next test."""
    if o.kind == "rebase":
        return math.inf                         # precondition of every other measurement on the new base
    h = (second if o.m >= best else best) if o.kind == "seed" else 0.0 if o.kind == "test" else bar
    if o.h is not None:                         # target mode: over the gap; a non-leader stack: over its leader
        h = max(h, o.h)
    if o.kind == "seed":
        return kg(o.m, o.v, o.V, h)
    if o.kind == "idea" or o.tail:
        return o.p * ei_t(o.m, o.v, o.V, h)
    return o.p * ei(o.m, o.v, o.V, h)


def score(options: Sequence[Option], *, harm_h: Mapping[str, float] = HARM_H, harm_q: float = HARM_Q
          ) -> list[Action]:
    """Every allowed option valued, best value/GPU-h first (harm-blocked tests left out)."""
    per_node: dict[str, float] = {}
    for o in options:
        if o.kind in ("seed", "test") and o.node is not None and not o.below:
            per_node[o.node] = max(per_node.get(o.node, -math.inf), o.m)
    best, second = (sorted(per_node.values(), reverse=True) + [0.0, 0.0])[:2]
    bar = max([0.0, *(o.m for o in options if o.kind == "test")])
    # a below-floor option (relax pass only) is valued over the incumbent's bar, not the gap it cannot reach (over the
    # gap its value underflows to 0 and the relax pass would have nothing to rank)
    acts = [Action(o.kind, o.node, o.lane, o.gpu_h, value(replace(o, h=None) if o.below else o, best, second, bar),
                   o.runs,
                   f"m={o.m:+.5f} sd={math.sqrt(o.v):.5f}", below=o.below)
            for o in options if o.kind != "test" or harm_ok(o.axes, harm_h, harm_q)]
    return sorted(acts, key=lambda a: -a.rate)


def floors(acts: Sequence[Action], floor: float = VALUE_FLOOR, wave_floor: float = WAVE_FLOOR
           ) -> tuple[float, float]:
    """(the value/GPU-h every non-screen GPU action must reach, the one a wave must reach): ``floor`` x the best
    screen / idea value/GPU-h, ``wave_floor`` x the best screen's (the best idea's when no screen is offered)."""
    ref = max([a.rate for a in acts if a.kind in ("screen", "idea") and a.value > 0], default=0.0)
    scr = max([a.rate for a in acts if a.kind == "screen" and a.value > 0], default=ref)
    return floor * ref, wave_floor * scr


def tag_ok(a: Action, tags: Mapping[str, str] | None, left: dict[str, float]) -> bool:
    """A per-tag GPU-h cap (fork-and-flush branches, ``core.fnf.budgets``): a non-test action on a tagged node fits
    in its tag's GPU-h left (and takes it). Tests are never capped: a fresh-seed test is how a branch is judged."""
    t = (tags or {}).get(a.node or "")
    if t is None or a.kind == "test":
        return True
    if left.get(t, math.inf) < a.gpu_h:
        return False
    left[t] = left.get(t, math.inf) - a.gpu_h
    return True


def group_key(a: Action, groups: Mapping[str, str] | None) -> str:
    """The soft-cap key of an action's source group (v3-c7i): ``group:<llm|code|other>`` for a seed / screen of a
    grouped node, ``group:code`` for a wave; '' otherwise (tests, ports, the re-baseline are never capped)."""
    if not groups:
        return ""
    if a.kind == "wave":
        return "group:code"
    if a.kind in ("seed", "screen") and a.node in groups:
        return f"group:{groups[a.node]}"
    return ""


def rank(options: Sequence[Option], free_slots: int, llm_free: int, *,
         harm_h: Mapping[str, float] = HARM_H, harm_q: float = HARM_Q, floor: float = VALUE_FLOOR,
         wave_floor: float = WAVE_FLOOR, caps: Mapping[str, int] | None = None,
         soft: Mapping[str, int] | None = None, tags: Mapping[str, str] | None = None,
         tag_left: Mapping[str, float] | None = None, relax_caps: Mapping[str, int] | None = None,
         groups: Mapping[str, str] | None = None,
         reserve: tuple[int, Collection[str]] | None = None,
         priority: Collection[str] = (), prefer: Collection[str] = (),
         partial_test: bool | Collection[str] = True, test_order: Mapping[str, float] | None = None) -> list[Action]:
    """The actions to launch now, in launch order: GPU actions (positive value, above the value floors, best
    value/GPU-h first, one per node) while GPU slots last; then one idea session per free LLM slot, round-robin over
    the lanes in value order. The two never compete: an idea takes no GPU slot now (its screening runs are ranked as a
    ``screen`` later). Relax pass (``RELAX_FILL``): slots still free then go to the positive-value seed / wave /
    screen actions the floors left out (``Action.below`` too), best value/GPU-h first, marked ``pre`` (preemptible
    pools only); hard caps and tag caps still hold. ``caps``: kind -> the most actions of that kind (hard, e.g. ports being screened);
    ``soft``: kind -> the most RUNS of that kind (e.g. the wave share) -- lifted for slots nothing else wants;
    ``tags`` / ``tag_left``: node -> tag, tag -> GPU-h left (hard, :func:`tag_ok`). ``relax_caps``: kind -> the most
    relax-pass actions of that kind (e.g. ``RELAX_MAX_WAVES`` minus the open relax waves; default: waves at
    ``RELAX_MAX_WAVES``). ``groups`` (v3-c7i, ``loop.core.alloc``): node -> llm / code / other; a seed / screen
    action of a node counts against ``soft["group:<its group>"]`` (a wave against ``group:code``) -- the LLM-idea
    units' GPU share, lifted like every soft cap for slots nothing else wants. ``reserve`` (v3 review K2): (slots,
    discovery nodes) -- a HARD reservation of that many slots for the seed / screen actions of the discovery nodes
    (seeds 1-2 of new LLM-idea nodes on the incumbent): no other action takes them, the LLM soft cap does not hold
    them back; capped at the discovery runs on offer (an unused reservation never idles a GPU). ``priority`` (10-21,
    ``acquire.program_arm_share``): QUEUED operator-program arm nodes -- their screen / seed actions are placed FIRST
    for up to ``PROGRAM_ARM_SHARE`` of ``free_slots`` (at least one action) and are exempt from the value floors (an
    arm is run on purpose); the rest of the order is unchanged. ``prefer`` (11-09, ``alloc.priority_categories``): QUEUED
    nodes of a priority category (bold structural ideas, and the struct_tune children of one) -- placed in TIERS: (A) the
    program-priority head, the owed tests, the in-flight seed extensions and a re-baseline, (B) the preferred SCREENS
    (their own value order kept; a screen that does not fit whole takes the slots left as a partial screen), (C) the
    rest in the original order. Each tier is placed (strict, then with the soft caps lifted) before the next is looked
    at -- 11-09 17:05-17:11Z: the soft LLM-group cap (``alloc.gpu_split``: 17 LLM runs live, cap 8) skipped every bold
    (LLM-idea) screen in the strict pass while non-LLM actions took the freed slots, so reordering alone never put a
    bold screen first. Program / confirm actions beyond the program quota stay in tier A. The discovery reservation
    still holds (the release pass walks the whole tiered order, A then B then C), nothing is exempt from a floor, and the
    relax pass orders seeds, then the preferred screens, then the rest. By design tier A's relaxed pass lets in-flight
    seed extensions past the soft LLM-group cap before any screen (the operator's order: seeds, then bold, then rest).
    ``partial_test`` (11-10): a TEST (2 x k_test runs, 10) that does not fit the free slots whole takes the slots
    left once at least ``TEST_MIN_SLOTS`` are free (at most one such test per call); its block opens and
    ``Tick._launch`` submits the block's owed runs FIRST on the next ticks. Without it a saturated cluster (gpu_free
    0-2 on nearly every tick) never had 10 free slots at once, so no test ever opened (nanobrain a node: 5 seeds,
    test entry ok, value-ranked first, seed_blocks empty). The tick passes False while an open block still owes runs.
    ``partial_test`` may also be a set of nodes: only their tests may take the partial seat (a challenger's).
    ``test_order`` (11-10): node -> the belief the promotion bar judges (the EB-shrunk PAIRED gain over the incumbent,
    ``bar.paired``): the TEST actions keep the places the value order gave the tests, but fill them best-first by it
    (values and floors unchanged) -- 11-10 16:38Z a re-root port with paired gain -.0008 on 3 seeds took the test
    over a node (+.0013 on 4, P(promote) .84) on a .0001 composite-EI tie."""
    acts = score(options, harm_h=harm_h, harm_q=harm_q)
    if test_order:
        pos = [i for i, a in enumerate(acts) if a.kind == "test"]
        best = sorted((acts[i] for i in pos), key=lambda a: -test_order.get(a.node or "", -math.inf))
        for i, a in zip(pos, best):
            acts[i] = a
    lo, lo_wave = floors(acts, floor, wave_floor)
    prio = set(priority) if PROGRAM_ARM_SHARE > 0 else set()
    gpu = [a for a in acts if a.kind != "idea" and ((a.value > 0 and not a.below
           and (a.kind in ("screen", "rebase") or a.rate >= (lo_wave if a.kind == "wave" else lo)))
           or (a.node in prio and a.kind == "screen"))]
    if prio:
        quota, take, first = math.ceil(PROGRAM_ARM_SHARE * free_slots), 0, []
        for a in gpu:
            if a.node in prio and a.kind in ("screen", "seed") and (not first or take + a.runs <= quota):
                first.append(a)
                take += a.runs
        ids = {id(a) for a in first}
        gpu = first + [a for a in gpu if id(a) not in ids]
    else:
        first = []
    pref = set(prefer)
    if pref:
        head, tail = gpu[:len(first)], gpu[len(first):]
        t_a = head + [a for a in tail if a.kind in PREFER_AHEAD or a.node in prio]   # program / confirm overflow too
        t_b = [a for a in tail if a.kind == "screen" and a.node in pref]
        ids = {id(a) for a in t_a + t_b}
        tiers = [t_a, t_b, [a for a in tail if id(a) not in ids]]
        gpu = [a for t in tiers for a in t]
    else:
        tiers = [gpu]
    out, slots, nodes, n_kind, runs_kind = [], free_slots, set(), {}, {}
    caps, soft, left = dict(caps or {}), dict(soft or {}), dict(tag_left or {})
    disc_nodes = set(reserve[1]) if reserve else set()

    def disc(a: Action) -> bool:
        return a.kind in ("seed", "screen") and a.node in disc_nodes
    want, seen_d = 0, set()
    for a in gpu:
        if disc(a) and a.node not in seen_d:
            seen_d.add(a.node)
            want += a.runs
    held = min(max(0, int(reserve[0])), want, free_slots) if reserve else 0
    # second pass: the soft caps are lifted for idle slots; third pass (v3 review, live 06:30Z: primary 5/8 with 3 idle jobs
    # pending): a discovery reservation whose actions could not be placed (a cap, a tag budget, a node already taken) is
    # RELEASED -- an unplaced reservation must never hold slots empty

    def partial(group: Sequence[Action], keep: bool = False) -> None:
        # (10-22 12:13-13:23Z: 2 effective primary slots empty for 70 min with 14 screenable nodes waiting): a screen is
        # `idea_runs` (3) runs and the passes place it only whole, so with 1-2 slots free NOTHING placed, and the relax
        # pass then handed the slots to preemptible-only seeds that had no open pool. A screen that does not fit whole
        # takes the slots left: its CRN seeds first, the rest follow as seed extensions
        # ``keep`` (rank(prefer=)'s tier B): the discovery reservation stays held -- only a discovery node's own
        # screen may take the reserved slots
        nonlocal slots, held
        for a in group:
            d = keep and disc(a) and held > 0
            room = slots - held if keep and not d else slots
            if room <= 0:
                continue
            if a.kind != "screen" or a.node in nodes or a.runs <= room or n_kind.get(a.kind, 0) >= caps.get(a.kind, math.inf):
                continue
            part = replace(a, runs=room, gpu_h=a.gpu_h * room / max(a.runs, 1),
                           why=f"{a.why} [partial: {room} of {a.runs} runs fit the free slots]")
            if not tag_ok(part, tags, left):
                continue
            out.append(part)
            slots -= part.runs
            nodes.add(a.node)
            n_kind[a.kind] = n_kind.get(a.kind, 0) + 1
            runs_kind[a.kind] = runs_kind.get(a.kind, 0) + part.runs
            if g := group_key(a, groups):
                runs_kind[g] = runs_kind.get(g, 0) + part.runs
            if d:
                held = max(0, held - part.runs)
    # with ``prefer`` each tier is placed (strict, relaxed) before the next; the release pass then walks the whole tiered
    # order (A, B, C), so an action the reservation held back keeps its tier's place (review: a stuck reservation must
    # not hand its slots to tier C ahead of an owed test / a bold screen)
    passes = [(t, relax, False) for t in tiers for relax in (False, True)] + [(gpu, True, True)]
    for tier, relax, release in passes:
        if release:
            held = 0
        for a in tier:
            part = a.kind == "test" and (a.node in partial_test if isinstance(partial_test, (set, frozenset))
                                         else partial_test) and a.runs > slots and a.node not in nodes \
                and slots - held >= TEST_MIN_SLOTS and not n_kind.get("test_partial", 0)
            if part:                                   # a test opens on the slots left; its owed runs go first later
                fit = (slots - held) // 2 * 2         # whole candidate + control pairs
                a = replace(a, runs=fit, gpu_h=a.gpu_h * fit / max(a.runs, 1),
                            why=f"{a.why} [partial: {fit} of {a.runs} runs now, the rest owed by its block]")
            if a.node in nodes or a.runs > slots or n_kind.get(a.kind, 0) >= caps.get(a.kind, math.inf):
                continue
            d = disc(a) and held > 0
            if not d and slots - a.runs < held:       # the discovery reservation: not for this action
                continue
            if not relax and not d and runs_kind.get(a.kind, 0) + a.runs > soft.get(a.kind, math.inf):
                continue
            g = group_key(a, groups)
            if not relax and not d and g and runs_kind.get(g, 0) + a.runs > soft.get(g, math.inf):
                continue
            if not tag_ok(a, tags, left):
                continue
            out.append(a)
            slots -= a.runs
            nodes.add(a.node)
            n_kind[a.kind] = n_kind.get(a.kind, 0) + 1
            if part:
                n_kind["test_partial"] = 1
            runs_kind[a.kind] = runs_kind.get(a.kind, 0) + a.runs
            if g:
                runs_kind[g] = runs_kind.get(g, 0) + a.runs
            if d:
                held = max(0, held - a.runs)
        if pref and len(tiers) > 1 and tier is tiers[1] and relax and slots > 0:   # tier B's partial screens
            partial(tier, keep=True)
    # fourth pass: the partial screens
    if slots > 0:
        partial(gpu)
    if RELAX_FILL and slots > 0 and not any(a.kind == "rebase" for a in acts):
        picked, r_kind = {id(a) for a in out}, {}
        rcaps = {"wave": RELAX_MAX_WAVES, **dict(relax_caps or {})}
        rel = sorted(acts, key=lambda a: 0 if a.kind == "seed" else 1 if a.kind == "screen" and a.node in pref else 2) \
            if pref else acts                        # 11-09: seeds, then the preferred screens, then the rest
        for a in rel:                               # relax: below-floor positive value, best value/GPU-h first
            if slots <= 0:
                break
            if id(a) in picked or a.kind not in RELAX_KINDS or a.value <= 0 or a.node in nodes \
                    or (a.below and FILL_DROP_BELOW) \
                    or a.runs > slots or n_kind.get(a.kind, 0) >= caps.get(a.kind, math.inf) \
                    or r_kind.get(a.kind, 0) >= rcaps.get(a.kind, math.inf) or not tag_ok(a, tags, left):
                continue
            out.append(replace(a, pre=True, why=f"{a.why} [relax: below floor, preemptible]"))
            slots -= a.runs
            nodes.add(a.node)
            n_kind[a.kind] = n_kind.get(a.kind, 0) + 1
            r_kind[a.kind] = r_kind.get(a.kind, 0) + 1
    ideas = sorted((a for a in acts if a.kind == "idea" and a.value > 0), key=lambda a: -a.value)
    out += [ideas[i % len(ideas)] for i in range(llm_free)] if ideas else []
    return out


def idle_fill(led: Ledger, blocks: Sequence[SeedBlock], statuses: Mapping[str, Status], incumbent: str,
              post: "Posterior", slots: int, taken: set[str], costs: "Costs", *,
              tags: Mapping[str, str] | None = None, tag_left: Mapping[str, float] | None = None,
              churned: frozenset[tuple[str, int]] | set[tuple[str, int]] = frozenset(),
              prefer: Collection[str] = ()) -> list[Action]:
    """``fill`` actions for the GPU slots no ranked action takes (v3-rc; the fix-B sim left 16-30 of 64 idle: nothing
    cleared the floors). Idle GPUs are free information; never a test (tests stay gated by ``test_entry_ok``). Each
    fill is SCREEN runs of one node (CRN seeds of its parent first), launched last and preemptible-only (the tick
    marks them; ``RunBook.pool``), at most ``max_seeds`` seeds per node. In order, while slots last:
    (a) far-class calibration: while the current epoch's FAR fit has < ``model.NEAR_MIN_DF`` paired dfs (a far test
        is judged on the conservative all-epochs fallback until then), far-change nodes of the epoch (the incumbent's
        children first, then by belief) at their parent's done SELECT seeds they lack -- each is a same-seed pair;
    (b) extra seeds on the SCREENING nodes with the best posterior (shifted to the incumbent); target mode: a node
        whose upper-90% effect is under SEED_FLOOR_FRAC x the gap is DROPPED (``FILL_DROP_BELOW``; off: ordered last)
        -- it cannot reach a test, so the slot stays idle rather than buy runs with no decision value;
    (c) the screens of QUEUED nodes no ranked action took (their value did not clear zero).
    (d) (11-10) (b) seeds only the nodes AHEAD of the incumbent (shifted belief > 0, not under the seed floor); the
        other screening nodes' seeds come after (c), from the slots it left.
    A tagged node (``tags`` / ``tag_left``: a fork-and-flush branch) gets only the runs its tag's GPU-h left buys.
    ``churned``: (node, seed) of fill runs the idle churn cancelled (``run_cancelled`` events): they keep using their
    node's seed budget (no cancel -> refill loop), and a queued node with one is not screened by fill again.
    ``prefer`` (11-09, ``alloc.priority_categories``, :func:`rank`'s): (c) screens these queued nodes first."""
    from loop.core import model as Mo
    from loop.core.backtest import fit_at as _fit
    from loop.core.data import PANEL, SELECT
    from loop.core.seeds import pool_of
    if not IDLE_FILL or slots <= 0 or post is None:
        return []
    epoch, inc = post.epoch, led.port(incumbent) or incumbent
    seeds: dict[str, set[int]] = {}
    done: dict[str, set[int]] = {}
    cap = seed_cap(costs)
    for r in led.runs:
        if r.status not in DEAD or (r.node, r.seed) in churned:
            seeds.setdefault(r.node, set()).add(r.seed)
        if r.status == "done":
            done.setdefault(r.node, set()).add(r.seed)
    churned_nodes = {n for n, _ in churned}
    out: list[Action] = []
    taken, left, run_h = set(taken), dict(tag_left or {}), run_hours(led)
    took: dict[str, list[float]] = {}
    for o in led.obs:
        took.setdefault(o.node, []).append(o.gpu_h)

    def hours(n: Any) -> float:
        """A run of ``n``: its own median GPU-h, else its parent's, else the base's median run."""
        xs = sorted(took.get(n.id) or took.get(n.parent or "") or [run_h])
        return xs[len(xs) // 2]

    def ok(n: Any) -> bool:                       # preemptible-only: long runs are the ones preemption kills
        return (n.id not in taken and n.base == epoch and n.parent in post.by and n.id != inc
                and not wave_node(n) and len(seeds.get(n.id, ())) < cap
                and hours(n) <= IDLE_MAX_RUN_FRAC * run_h)

    def add(nid: str, k: int, why: str) -> None:
        nonlocal slots
        k = min(k, slots)
        if (t := (tags or {}).get(nid)) in left:             # a capped tag: the runs its GPU-h left buys
            k = max(0, min(k, int(left[t] // run_h)))
            left[t] -= k * run_h
        if k > 0:
            out.append(Action("fill", nid, led.nodes[nid].lane, k * run_h, 0.0, k, why))
            taken.add(nid)
            slots -= k
    # (a) far calibration
    try:
        far = Mo.pairs_by_change(led.nodes, post.near)[1]
        df = _fit(getattr(post, "fit_led", led).in_epoch(epoch), far)[0].df.get(PANEL, 0.0)
    except (ValueError, ZeroDivisionError, KeyError):
        df = 0.0
    if df < Mo.NEAR_MIN_DF:
        cands = [n for n in led.nodes.values() if ok(n) and n.id not in post.near
                 and statuses.get(n.id) in (Status.SCREENING, Status.PASSED, Status.FAILED)]
        cands.sort(key=lambda n: (n.parent != inc, -(post.belief(n.id)[0] if n.id in post.est else -1.0)))
        for n in cands:
            if slots <= 0:
                break
            crn = [s for s in done.get(n.parent, ()) - seeds.get(n.id, set()) if pool_of(s, blocks) == SELECT]
            add(n.id, min(len(crn), cap - len(seeds.get(n.id, ()))),
                f"idle (a) far calibration: epoch far df {df:g} < {Mo.NEAR_MIN_DF}")
    # (b) the best SCREENING nodes' seeds
    live = [n for n in led.nodes.values() if ok(n) and statuses.get(n.id) is Status.SCREENING and n.id in post.est]
    gap = gap_of(led, incumbent) if target_on() else None

    def under(n: Any) -> bool:
        """Below the seed floor (target mode): dropped (FILL_DROP_BELOW), else ordered last."""
        if gap is None or inc not in post.by or SEED_FLOOR_FRAC <= 0:
            return False
        m, v = post.belief(n.id)
        dm, dv = post.shift(n.id, Mo.COMPOSITE, inc)
        return m + dm + Z90 * math.sqrt(max(v + dv, 0.0)) < SEED_FLOOR_FRAC * gap
    if FILL_DROP_BELOW:
        live = [n for n in live if not under(n)]
    if FUTILITY_P > 0:                            # the futility stop: a flat node's slots go to (c)'s fresh screens
        fbar = futility_bar(led, incumbent)

        def stop(n: Any) -> bool:
            m, v = post.belief(n.id)
            dm, dv = post.shift(n.id, Mo.COMPOSITE, inc) if inc in post.by else (0.0, 0.0)
            k = post.est[n.id][Mo.COMPOSITE].n
            return not futility_exempt(n, k) and futile(m + dm, v + dv, k, fbar) is not None
        live = [n for n in live if not stop(n)]
    def shifted(n: Any) -> float:
        return post.belief(n.id)[0] + post.shift(n.id, Mo.COMPOSITE, inc)[0] if inc in post.by else post.belief(n.id)[0]
    live.sort(key=lambda n: (under(n), -shifted(n)))
    # 11-10 (nanobrain): (b) went before (c) whatever the node's belief, so 5-seed replicates of nodes at -.005 to
    # -.009 (a node / a node / a node) took the idle slots while ~20 QUEUED ideas had no run. Now only a node AHEAD of
    # the incumbent (shifted belief > 0: on its way to a test) is seeded before the queued screens; the rest after them
    ahead = [n for n in live if shifted(n) > 0 and not under(n)]
    behind = [n for n in live if n not in ahead]

    def seed_nodes(group: Sequence[Any], why: str) -> None:
        for n in group:
            if slots <= 0:
                break
            if n.id not in taken:
                add(n.id, cap - len(seeds.get(n.id, ())), why)
    seed_nodes(ahead, "idle (b) seed a top-posterior screening node")
    # (c) the screens nobody took: the preferred (bold) ones first, then shortest runs first
    queued = [n for nid, st in statuses.items() if st is Status.QUEUED and (n := led.nodes.get(nid)) is not None
              and n.id not in taken and n.base == epoch and n.parent in post.by and not wave_node(n)
              and n.id not in churned_nodes and hours(n) <= IDLE_MAX_RUN_FRAC * run_h]
    pref = set(prefer)
    for n in sorted(queued, key=lambda n: (n.id not in pref, hours(n))):
        if slots <= 0:
            break
        add(n.id, costs.idea_runs, "idle (c) screen a queued node" + (" (priority category)" if n.id in pref else ""))
    seed_nodes(behind, "idle (d) seed a screening node not ahead of the incumbent")   # only the slots (c) left
    return out


# -- options from the ledger ------------------------------------------------------------------------------------
#: base epochs (docs §8 "Base epochs"): an old-epoch prior point's weight in the cell means
OLD_EPOCH_W = 0.5
#: P(a port pays: its graft applies AND its screening on the new base shows a positive own increment). TUNABLE
#: (``acquire.port_p``) as the PRIOR mean; :func:`port_p` updates it with the measured port outcomes of the epoch
#: (``PORT_P_N`` pseudo-ports of prior weight). 360/400 grafts merged cleanly, but a clean graft is not a gain: the old
#: posterior was fitted on another base, with another ρ.
PORT_P = 0.5
PORT_P_N = 4.0
#: max-of-K strength in ``m_sel = m - β √v E_K`` (a test's / port's belief as the best of K siblings). TUNABLE
#: (``acquire.max_of_k_beta``; the meta-cycle sets it through ``core.meta.apply``), read at call time. Default 0 from
#: the held-out backtest (docs §7): against the parent's multi-seed mean + CUPED the selection bias is mostly gone
#: already -- 80% coverage .83 / .81 (held-out / all) at β = 0 vs .33 / .51 at β = 1; refit β -.24 / -.19.
MAX_OF_K_BETA = 0.0
#: a structural tuning ladder's child (notes ``struct_tune=``, loop.core.struct_tune) is the best of <= 5 tuned points: its
#: test entry is max-of-K shrunk with AT LEAST this strength (``ideation.struct_tune_beta``), whatever MAX_OF_K_BETA is
STRUCT_BETA = 1.0


def beta_for(node: Any) -> float:
    """The max-of-K strength for ``node``'s test entry / port belief (:data:`MAX_OF_K_BETA`, raised to
    :data:`STRUCT_BETA` for a structural ladder child)."""
    notes = f" {getattr(node, 'notes', '') or ''}"
    return max(MAX_OF_K_BETA, STRUCT_BETA) if " struct_tune=" in notes or " rescue_best_of=" in notes else MAX_OF_K_BETA
#: the test-entry rule's ``P(gain > 0) >= p_min`` (``status.test_entry_ok``). TUNABLE (``acquire.test_p_min``, bounds
#: [.6, .9]: outside the LOCKED ``test.`` prefix -- it sets test VOLUME, never the test's verdict), read at call time
TEST_P_MIN = 0.7
#: a test that does not fit the free slots whole opens once this many are free (one candidate + control pair):
#: :func:`rank`'s ``partial_test``
TEST_MIN_SLOTS = 2
DEAD = frozenset({"crash", "void"})


@dataclass(frozen=True)
class Costs:
    idea_runs: int = 3          # screening runs a new idea needs
    llm_gpu_h: float = 0.5      # one research session, priced in GPU-h
    k_test: int = 5
    max_seeds: int = 5          # a candidate's selection seeds (then only a test adds data)
    port_gpu_h: float = 0.1     # an adapter graft + smoke of an old-epoch node onto the new base
    rebase_seeds: int = 5       # the re-baseline after a base change: seeds each for the new root and incumbent port


def rebase_owed(led: Ledger, incumbent: str, costs: Costs = Costs()) -> dict[str, int]:
    """After a base change, the re-baseline seeds still owed: {new base root: k, incumbent's port: k} (the incumbent
    itself while it is not yet ported). Empty once both have ``rebase_seeds`` live-or-done seeds, and always for a
    one-epoch ledger."""
    if len(led.bases) < 2:
        return {}
    seeds: dict[str, set[int]] = {}
    for r in led.runs:
        if r.status not in DEAD:
            seeds.setdefault(r.node, set()).add(r.seed)
    port = led.port(incumbent)
    owed = {led.base_root: costs.rebase_seeds - len(seeds.get(led.base_root, ())),
            port or incumbent: costs.rebase_seeds - (len(seeds.get(port, ())) if port else 0)}
    return {n: k for n, k in owed.items() if k > 0}


class Posterior:
    """The core posterior of one tick, from the screening data (open test blocks hidden): per base epoch the noise
    (ρ from one-change pairs; ``far`` = several-changes pairs), every node's CUPED increment over its parent, the
    hierarchical prior (old-epoch increments at weight ``OLD_EPOCH_W``, ports left out: a port is a whole chain) and
    each node's own-increment composite posterior :meth:`belief`. Shared by acquisition and ideation."""

    @property
    def TARGETS(self) -> tuple[str, ...]:              # noqa: N802 - call time: the installed project's harm axes
        return (M.COMPOSITE, *HARM_H)

    def __init__(self, led: Ledger, blocks: Sequence[SeedBlock], *, near: frozenset[str] = frozenset(),
                 evalonly: frozenset[str] = frozenset(), eo_floor: float = 0.0, eo_min_df: int = 4,
                 eo_runs: frozenset[str] = frozenset()) -> None:
        self.led = led = replace(led, obs=tuple(screening_obs(led.obs, blocks)))
        #: v3-c5b M1: the noise fits never see an eval-only run (``eo_runs``: runs with an eval plan, incl. the null
        #: control): its pair with the parent's same-seed checkpoint has no training noise, and a flood of near-zero
        #: eval-only deltas would drag every class's sigma_d down / rho up. Only the dedicated eval-only noise
        #: (loop.evalonly.posterior_noise, from ``self.by``) and the CUPED estimates read them
        self.fit_led = replace(led, obs=tuple(o for o in led.obs if o.run_id not in eo_runs)) if eo_runs else led
        self.epoch, self.parents = led.epoch, led.change_parents()
        #: change classes (fix A): ``near`` = nodes one config key from their parent, config only; every other node
        #: (code change, several keys, crossover, port) is judged with the far fit
        self.near = frozenset(near)
        self.pairs = M.pairs_by_change(led.nodes, self.near)
        self._fits: dict[tuple[str, bool], tuple[M.Noise, dict[str, float]]] = {}
        self.by = by = led.by_node()
        self.noise, self.weights = self.fit(self.epoch)
        self.est: dict[str, dict[str, M.Estimate]] = {}
        eo_fits: dict[tuple[str, bool], Any] = {}
        for nid, o in by.items():
            node = led.nodes.get(nid)
            if node is not None and node.parent is not None and node.parent in by:   # parent and child: one epoch
                nz, wz = self.fit(node.base, nid not in self.near or bool(node.port_of))   # its change class
                if nid in evalonly:          # v3-c5 I5: paired on the parent's own checkpoints (loop.evalonly)
                    from loop.evalonly import posterior_noise
                    key = (node.base, nid not in self.near)
                    if key not in eo_fits:
                        eo_fits[key] = posterior_noise(nz, by, self.parents, [
                            n for n in evalonly if n in led.nodes and led.nodes[n].base == node.base], wz, eo_floor,
                            eo_min_df)
                    nz = eo_fits[key] or nz
                self.est[nid] = {t: e for t in self.TARGETS
                                 if (e := M.cuped(o, by[node.parent], t, nz, wz)) is not None}
        # ports are a whole chain; wave points (and the diff knobs ported for a wave) share ONE seed per block, so
        # their increments are correlated draws of a design, not independent ideas: neither is a prior point
        pts = [(n, e[M.COMPOSITE]) for n, e in self.est.items()
               if not led.nodes[n].port_of and not wave_node(led.nodes[n])]
        self.prior = M.fit_prior([M.PriorPoint(self.cell(n), e.delta, e.var, w=self.weight(n)) for n, e in pts])
        self.lanes = M.fit_prior([M.PriorPoint((self.cell(n)[0], led.nodes[n].lane, ""), e.delta, e.var,
                                               w=self.weight(n)) for n, e in pts])

    def fit(self, base: str, far: bool = False) -> tuple[M.Noise, dict[str, float]]:
        """The epoch's noise and composite weights from its NEAR pairs (one config key, config only) or, ``far``, its
        far pairs (``model.pairs_by_change``); the far fit keeps the near weights. No near pairs on the base: the far
        fit stands in; too few far pairs: far pairs over every epoch, then (tiny ledgers) every child / parent pair."""
        key = (base, far)
        if key in self._fits:
            return self._fits[key]
        led, pairs = self.fit_led, self.pairs[1] if far else self.pairs[0]
        try:
            got = fit_at(led.in_epoch(base), pairs)
        except (ValueError, ZeroDivisionError):
            got = None
        if got is None and not far:
            self._fits[key] = self.fit(base, True)
            return self._fits[key]
        for alt in ((led, pairs), (led, self.parents)):
            if got is not None:
                break
            try:
                got = fit_at(*alt)
            except (ValueError, ZeroDivisionError):
                got = None
        if got is None:
            raise ValueError(f"no noise fit for base {base[:7]}")
        if far and (base, False) in self._fits:
            got = (got[0], self._fits[base, False][1])
        self._fits[key] = got
        return got

    def history(self) -> M.Noise:
        """Noise over every epoch's screening data, every child / parent pair (what the replay of historical,
        not-yet-frozen blocks judges with: unchanged by the change classes)."""
        if (None, False) not in self._fits:
            self._fits[None, False] = fit_at(self.fit_led, self.parents)   # type: ignore[index]
        return self._fits[None, False][0]                                   # type: ignore[index]

    def weight(self, nid: str) -> float:
        return 1.0 if self.led.nodes[nid].base == self.epoch else OLD_EPOCH_W

    def cell(self, nid: str) -> tuple[str, str, str]:
        """The prior cell; an old-epoch carrier maps to its port (unported: a cell of its own)."""
        n = self.led.nodes[nid]
        if n.base == self.epoch or n.carrier == ROOT:
            return M.cell_of(n)
        return (self.led.port(n.carrier) or f"{n.carrier}@{n.base[:7]}", n.family, n.lane)

    def belief(self, nid: str) -> tuple[float, float]:
        """(m, v): the composite posterior of the node's own increment over its parent (old-epoch evidence V/w)."""
        e = self.est[nid][M.COMPOSITE]
        if self.weight(nid) < 1.0:
            e = M.Estimate(e.delta, e.var / self.weight(nid), e.n, e.paired)
        pm, t2, _ = self.prior.cell(self.cell(nid))
        return M.posterior(pm, t2, e)

    def beliefs(self) -> dict[str, tuple[float, float, int]]:
        return {n: (*self.belief(n), e[M.COMPOSITE].n) for n, e in self.est.items()}

    def shift(self, nid: str, t: str, to: str) -> tuple[float, float]:
        """(mean, variance) of parent - ``to`` (the incumbent of the node's epoch) on target t."""
        p, base = self.led.nodes[nid].parent, self.led.nodes[nid].base
        if p == to:
            return 0.0, 0.0
        nz, wz = self.fit(base)
        (mp, n_p), (mi, n_i) = M.parent_mean(self.by[p], t, wz), M.parent_mean(self.by[to], t, wz)
        return mp - mi, nz.sigma2[t] * (1 / n_p + 1 / n_i)


def wave_node(node: Any) -> bool:
    """A wave point (launched by its wave, never seeded / tested on its own: x* is) or a diff knob ported for a wave."""
    from loop.core.factorial import is_point
    notes = getattr(node, "notes", "") or ""
    return is_point(notes) or "for_wave=" in notes


def test_entry_ok(nid: str, post: "Posterior", incumbent: str, k: int) -> bool:
    """May ``nid`` open a fresh-seed test? ``status.test_entry_ok`` (fix A): the selection-shrunk belief vs the
    incumbent clears half the test's MDE or P(gain > 0) >= ``TEST_P_MIN`` (.7), against the σ_d of the node's change
    class (near only for a one-config-key node), at its upper confidence bound. A negative or coin-flip candidate is
    never worth 10 runs. Then the seed-artifact gate (``status.seed_artifact``): its paired gain over its control
    must survive leaving out any one seed, and no seed may carry more than ``status.max_seed_share`` of it."""
    return entry_check(nid, post, incumbent, k)[0]


def entry_check(nid: str, post: "Posterior", incumbent: str, k: int) -> tuple[bool, str]:
    """(:func:`test_entry_ok`, why not)."""
    from loop.core import status as ST
    ok, why = ST.test_entry_ok(nid, post, incumbent, near=nid in post.near, beta=beta_for(post.led.nodes.get(nid)), k=k,
                               p_min=TEST_P_MIN)
    if not ok:
        return False, str(why)
    node = post.led.nodes[nid]
    try:
        w = post.fit(node.base)[1]
    except ValueError:
        w = post.weights
    try:                                    # v3 review K6: the outlier test's sd_d -- the fit test_entry_ok uses
        sd_d = post.fit(node.base, nid not in post.near)[0].sd_d(M.COMPOSITE)
    except (ValueError, KeyError, ZeroDivisionError, TypeError, AttributeError):
        sd_d = None
    ok, why = ST.seed_artifact(post.by.get(nid, ()), post.by.get(node.parent, ()), w, sd_d=sd_d)
    if ok and CONTROL_BAR is not None:
        ok, why = control_bar(nid, post, w, *CONTROL_BAR)
    return ok, why


def control_bar(nid: str, post: "Posterior", w: Mapping[str, float], z: float, share: float,
                min_sib: int) -> tuple[bool, str]:
    """``acquire.control_bar`` (11-08): (ok, why). The shared-control check of :func:`entry_check` -- the node's
    control (its parent) on the node's screen seeds vs the control on its OTHER seeds, in per-run sds of the composite;
    when it ran low there (z <= -``z``) while >= ``share`` of its >= ``min_sib`` measured children look positive, the
    node's belief must stay > 0 after subtracting the shortfall."""
    node = post.led.nodes.get(nid)
    if node is None or not node.parent:
        return True, ""
    mine = M.seed_means(post.by.get(nid, ()), M.COMPOSITE, w)
    ctl = M.seed_means(post.by.get(node.parent, ()), M.COMPOSITE, w)
    on, off = [ctl[s] for s in mine if s in ctl], [v for s, v in ctl.items() if s not in mine]
    if not on or len(off) < 2:
        return True, ""
    try:
        sig = math.sqrt(post.fit(node.base)[0].sigma2[M.COMPOSITE])
    except (ValueError, KeyError, TypeError, AttributeError):
        return True, ""
    gap = sum(on) / len(on) - sum(off) / len(off)
    zz = gap / (sig * math.sqrt(1 / len(on) + 1 / len(off))) if sig > 0 else 0.0
    if zz > -z:
        return True, ""
    sibs = [n for n, x in post.led.nodes.items() if x.parent == node.parent and n in post.est]
    pos = sum(post.belief(n)[0] > 0 for n in sibs)
    if len(sibs) < min_sib or pos < share * len(sibs):
        return True, ""
    m = post.belief(nid)[0]
    if m + gap > 0:
        return True, ""
    return False, (f"control_offset: {node.parent} ran {gap:+.4f} (z {zz:.1f}) on {nid}'s screen seeds vs its other "
                   f"seeds and {pos}/{len(sibs)} of its children look positive; {nid}'s belief {m:+.4f} does not clear "
                   "that shortfall")


def retest_candidates(led: Ledger, blocks: Sequence[SeedBlock], post: "Posterior") -> list[str]:
    """PASSED nodes that did not switch and can be re-tested against the current incumbent (``status``, fix A)."""
    from loop.core import status as ST
    return list(ST.retest_candidates(led, blocks, post.history()))


def port_p(led: Ledger, statuses: Mapping[str, Status], post: "Posterior", costs: "Costs") -> tuple[float, int, int]:
    """(p, paid, measured): ``PORT_P`` (the prior, ``PORT_P_N`` pseudo-ports) updated with this epoch's measured ports.
    A port pays when its own increment over its parent is positive once it has its ``idea_runs`` screening seeds; a
    port that did not graft (REJECTED: conflict / already contained in the base) is a miss. The incumbent's port and
    ports made for a wave are not port decisions."""
    inc = next((n for n, s in statuses.items() if s is Status.INCUMBENT), "")
    paid = measured = 0
    for n in led.nodes.values():
        if not n.port_of or n.base != post.epoch or n.id == inc or "for_wave=" in n.notes \
                or "port_src=" in n.notes or RESCUE_RE.search(n.notes or ""):   # a rescue port: no port decision                         # an alt-root auto-port is not an old-epoch port decision
            continue
        if statuses.get(n.id) is Status.VOID:
            measured += 1
        elif n.id in post.est and post.est[n.id][M.COMPOSITE].n >= costs.idea_runs:
            measured += 1
            paid += post.belief(n.id)[0] > 0
    return (PORT_P * PORT_P_N + paid) / (PORT_P_N + measured), paid, measured


def open_ports(led: Ledger, statuses: Mapping[str, Status], costs: "Costs") -> int:
    """Ports of the current epoch still being screened: QUEUED, or fewer than ``idea_runs`` finished runs. A moonshot
    rescue's port (``moonshot_rescue=``) is not counted: the rescue's own cap bounds it (11-11 review: its held points
    would otherwise sit in PORT_OPEN_MAX for good)."""
    fin: dict[str, int] = {}
    for r in led.runs:
        if r.status in ("done", "crash"):
            fin[r.node] = fin.get(r.node, 0) + 1
    return sum(1 for n in led.nodes.values() if n.port_of and n.base == led.epoch and "for_wave=" not in n.notes
               and not RESCUE_RE.search(n.notes or "")
               and statuses.get(n.id) in (Status.QUEUED, Status.SCREENING) and fin.get(n.id, 0) < costs.idea_runs)


LIVE = frozenset({"submitting", "running"})


def budgets(led: Ledger, statuses: Mapping[str, Status], free_slots: int, costs: "Costs") -> dict[str, dict[str, int]]:
    """``rank``'s ``caps`` (ports: ``PORT_OPEN_MAX`` minus those being screened; waves: ``MAX_OPEN_WAVES`` minus the
    open waves) and ``soft`` (wave runs: ``WAVE_SHARE`` of all slots minus the wave runs in flight)."""
    from loop.core import factorial as F
    live = [r for r in led.runs if r.status in LIVE]
    waves = sum(1 for r in live if r.node in led.nodes and wave_node(led.nodes[r.node]))
    total = free_slots + len(live)
    return {"caps": {"port": max(0, PORT_OPEN_MAX - open_ports(led, statuses, costs)),
                     "wave": max(0, MAX_OPEN_WAVES - len(F.open_waves(led, closed=CLOSED_WAVES)))},
            "soft": {"wave": max(0, int(WAVE_SHARE * total) - waves)}}


def run_hours(led: Ledger) -> float:
    """The median GPU-h of a run on the current base (all bases until it has runs)."""
    cur = [o.gpu_h for o in led.obs if o.base == led.epoch] or [o.gpu_h for o in led.obs]
    return sorted(cur)[len(cur) // 2] if cur else 1.0


def struct_k(led: Ledger, node: Any, siblings: int, booked: Mapping[str, int] | None = None) -> int:
    """K of the max-of-K selection shrink for ``node``. A structural-ladder child (notes ``struct_tune=``) is the best of
    ALL the points its ladder booked (``booked``: parent -> ``len(K_STRUCT[parent].children)``; else the ladder children
    the ledger holds under that parent): the LIVE sibling count falls as siblings resolve, which would shrink the
    selection correction exactly when the best point is the one left."""
    if (mk := RESCUE_K_RE.search(getattr(node, "notes", "") or "")) is not None:   # a moonshot rescue's chosen point
        return max(siblings, int(mk.group(1)))
    if " struct_tune=" not in f" {getattr(node, 'notes', '') or ''}":
        return siblings
    n = int((booked or {}).get(node.parent or "", 0))
    if n <= 0:
        n = sum(1 for x in led.nodes.values() if x.parent == node.parent and " struct_tune=" in f" {x.notes or ''}")
    return max(siblings, n)


def ledger_options(led: Ledger, blocks: Sequence[SeedBlock], statuses: Mapping[str, Status], incumbent: str,
                   costs: Costs = Costs(), post: Posterior | None = None,
                   knobs: Mapping[str, Sequence["BankKnob"]] | None = None, dropped: list | None = None,
                   booked: Mapping[str, int] | None = None) -> list[Option]:
    """seed / test options for every SCREENING node of the current base epoch (never a wave point: x* is), a port
    option for every SCREENING node of an older epoch with >= ``PORT_MIN_SEEDS`` seeds, a positive selection-shrunk
    belief and no port yet, a screen option per QUEUED node, a wave option per knob bank, and one idea option per lane
    (:class:`Posterior` beliefs moved from the parent to the incumbent of the node's epoch by the difference of their
    multi-seed means; a test's -- and a port's -- belief is shrunk for the max of its siblings, ``max_of_k``). A test
    needs :func:`test_entry_ok`. Noise, CUPED pairs and parent means never cross an epoch. While :func:`rebase_owed`
    is not empty the only option is the re-baseline. Then :func:`target_aware` (mechanism groups of the port /
    crossover-screen stacks; target mode: the gap as every option's bar, the gap floor -> ``dropped``)."""
    full, run_h = led, run_hours(led)
    if owed := rebase_owed(led, incumbent, costs):
        n = sum(owed.values())
        return [Option("rebase", incumbent, "rebase", 0.0, 0.0, 0.0,
                       n * run_h + (costs.port_gpu_h if led.port(incumbent) is None else 0.0), n)]
    post = post or Posterior(led, blocks)
    led, by, est, epoch = post.led, post.by, post.est, post.epoch
    inc = led.port(incumbent) or incumbent
    if inc not in by:
        return []
    s2, rho = post.noise.sigma2[M.COMPOSITE], post.noise.rho[M.COMPOSITE]
    live = [n for n in est if statuses.get(n) is Status.SCREENING and not wave_node(led.nodes[n])]
    retest = [n for n in retest_candidates(led, blocks, post)
              if n in est and n not in live and led.nodes[n].base == epoch and not wave_node(led.nodes[n])]
    siblings: dict[str | None, int] = {}
    for n in (*live, *retest):
        siblings[led.nodes[n].parent] = siblings.get(led.nodes[n].parent, 0) + 1
    pp = port_p(led, statuses, post, costs)[0]
    scr, cap = screen_seeds(full.runs), seed_cap(costs)
    fbar = futility_bar(full, incumbent) if FUTILITY_P > 0 else 0.0
    out = []
    for nid in (*live, *retest):
        node = led.nodes[nid]
        old = node.base != epoch
        to = inc if not old else (led.port(incumbent, node.base) or incumbent)
        if (old and led.port(nid) is not None) or to not in by:
            continue                              # ported: the port is the node now
        m, v = post.belief(nid)
        dm, dv = post.shift(nid, M.COMPOSITE, to)
        m, v = m + dm, v + dv
        k, n_own = min(16, struct_k(led, node, siblings[node.parent], booked)), est[nid][M.COMPOSITE].n
        m_sel = M.max_of_k(m, v, k, beta_for(node))
        if old:                                   # a new contender on the new base: graft, then screen; only a
            if n_own >= PORT_MIN_SEEDS and m_sel > 0:    # measured (>= 2 seeds), still-positive old node
                out.append(Option("port", nid, node.lane, m_sel, v, s2 * (1 - rho ** 2) / costs.idea_runs,
                                  costs.idea_runs * run_h + costs.port_gpu_h, costs.idea_runs, pp))
            continue
        sh = {a: post.shift(nid, a, to) for a in HARM_H if a in est[nid]}
        axes = {a: (est[nid][a].delta + sh[a][0], est[nid][a].var + sh[a][1]) for a in sh}
        nz_c = post.fit(epoch, nid not in post.near)[0]       # its change class: near only for one config key
        s2_c, rho_c = nz_c.sigma2[M.COMPOSITE], nz_c.rho[M.COMPOSITE]
        if max(n_own, len(scr.get(nid, ()))) < cap and nid not in retest and (
                futility_exempt(node, n_own) or futile(m, v, n_own, fbar) is None):
            out.append(Option("seed", nid, node.lane, m, v, s2_c * (1 - rho_c ** 2), run_h, 1))
        if test_entry_ok(nid, post, to, k):
            out.append(Option("test", nid, node.lane, m_sel, v, 2 * (1 - rho_c) * s2_c / costs.k_test,
                              2 * costs.k_test * run_h, 2 * costs.k_test, axes=axes))
    out += _screen_options(led, statuses, inc, post, costs, run_h)
    out += _wave_options(led, inc, post, knobs or {}, run_h)
    stacks = [o.node for o in out if o.node in led.nodes and (o.kind == "port" or o.kind == "screen"
                                                              and led.nodes[o.node].lane == "crossover")]
    mech = mechanisms(led, stacks, ref=inc)        # legs the incumbent holds are priced by the shift already
    out = [replace(o, group=mech[o.node]) if o.kind in ("port", "screen") and o.node in mech else o for o in out]
    gap = gap_of(full, incumbent) if target_on() else None
    tested = {led.nodes[c].lane for b in blocks if b.pool == TEST for c in b.node_ids[:-1]
              if c in led.nodes and led.nodes[c].base == epoch}
    widen = gap is not None and gap > best_available(out)
    for lane in LANES:
        if not (src := IDEA_LANE_PRIOR.get(lane, lane)):    # its sessions are valued on the brief's lane prior
            continue
        made = [n for n in led.nodes.values() if n.lane == src]
        p = sum(n.id in by for n in made) / len(made) if made else 0.5
        m, t2, _ = post.lanes.cell((inc, src, ""))
        if widen and src not in tested:             # an unmeasured tail is the only route across the gap
            t2 *= UNTESTED_VAR_MULT
        out.append(Option("idea", None, lane, m, t2, s2 * (1 - rho ** 2) / costs.idea_runs,
                          costs.idea_runs * run_h + costs.llm_gpu_h, costs.idea_runs, p))
    return target_aware(out, gap, dropped)


def best_available(options: Sequence[Option]) -> float:
    """The best shrunk effect the material in hand offers: FRESH_SHRINK x the best candidate / port / wave belief."""
    return FRESH_SHRINK * max([o.m for o in options if o.kind in ("seed", "test", "port", "wave")], default=0.0)


def gap_floor_ub(o: Option) -> float:
    """The option's upper-90% effect, what :func:`target_aware`'s gap floor compares: ``m + Z90 sd``, the mean shrunk
    once by FRESH_SHRINK only when it is a raw estimate (``Option.raw``); a posterior belief / forecast (ports, waves:
    their stack legs already carry FRESH_SHRINK) is not shrunk again, nor is its sd."""
    return (FRESH_SHRINK * o.m if o.raw else o.m) + Z90 * math.sqrt(max(o.v, 0.0))


def target_aware(options: Sequence[Option], gap: float | None, dropped: list | None = None) -> list[Option]:
    """One mechanism, one credit: within a mechanism group (``Option.group``) only the leader (highest m) is valued
    over the usual bar; the others must beat the leader (``h``). Target mode (``gap``): every option's ``h`` is at
    least the gap (a TEST keeps h = 0 and is exempt from the floor), and a wave / port whose upper-90% effect
    (:func:`gap_floor_ub`) is below GAP_FLOOR_FRAC x gap is dropped (appended to ``dropped`` as (kind, node, ub);
    10-13: a gap-floored wave is dropped outright too -- its carrier gets no relax wave that tick (w003); a seed under
    the seed floor is kept marked ``below``: :func:`rank` gives it idle slots only, none under ``FILL_DROP_BELOW``).
    Seed extensions are exempt from that floor (their value is information toward a test) but not from the lower
    SEED_FLOOR_FRAC x gap one: a node whose upper-90% effect over the incumbent is below it cannot reach a test."""
    lead: dict[str, float] = {}
    for o in options:
        if o.group:
            lead[o.group] = max(lead.get(o.group, -math.inf), o.m)
    out = []
    for o in options:
        if o.kind == "test":                    # value over the incumbent (h = 0), no gap floor: the entry rule judges it
            out.append(o)
            continue
        below = False
        if gap is not None and o.kind in ("wave", "port") and (ub := gap_floor_ub(o)) < GAP_FLOOR_FRAC * gap:
            if dropped is not None:
                dropped.append((o.kind, o.node, ub))
            continue                            # a port / wave under the gap floor: no relax wave on that carrier
        if gap is not None and o.kind == "seed" and SEED_FLOOR_FRAC > 0 \
                and (ub := gap_floor_ub(o)) < SEED_FLOOR_FRAC * gap:
            if dropped is not None:
                dropped.append((o.kind, o.node, ub))
            below = True                        # a seed under the seed floor: relax pass only
        h = [x for x in (gap, lead.get(o.group) if o.group and o.m < lead[o.group] else None) if x is not None]
        o = replace(o, below=True) if below else o
        out.append(replace(o, h=max(h)) if h else o)
    return out


def harm_blocked(led: Ledger, statuses: Mapping[str, Status] | None, incumbent: str, post: Posterior, *,
                 harm_h: Mapping[str, float] = HARM_H, harm_q: float = HARM_Q
                 ) -> list[tuple[str, float, float, float]]:
    """(node, composite m vs the incumbent, robustness m, robustness sd) of the current-epoch candidates with a
    positive belief whose test the harm limit blocks (``statuses`` None: every node with an estimate)."""
    inc = led.port(incumbent) or incumbent
    if inc not in post.by:
        return []
    out = []
    for nid, est in post.est.items():
        node = led.nodes[nid]
        if node.base != post.epoch or wave_node(node) or nid == inc or (
                statuses is not None and statuses.get(nid) is not Status.SCREENING):
            continue
        m = post.belief(nid)[0] + post.shift(nid, M.COMPOSITE, inc)[0]
        sh = {a: post.shift(nid, a, inc) for a in harm_h if a in est}
        axes = {a: (est[a].delta + sh[a][0], est[a].var + sh[a][1]) for a in sh}
        if m > 0 and axes and not harm_ok(axes, harm_h, harm_q):
            rm, rv = next(iter(axes.values()))
            out.append((nid, m, rm, math.sqrt(rv)))
    return sorted(out, key=lambda x: -x[1])


def cell_carrier(led: Ledger, nid: str) -> str:
    """The carrier a child of ``nid`` is filed under in the prior: ``nid`` itself once it is a carrier, else its own."""
    return nid if nid in {c for _, c in led.carriers} or nid not in led.nodes else led.nodes[nid].carrier


def _wave_options(led: Ledger, inc: str, post: Posterior, knobs: Mapping[str, Sequence["BankKnob"]],
                  run_h: float) -> list[Option]:
    """A ``wave`` option per carrier of the current base with knobs and no open wave (cap: one open wave per
    carrier), costing its N runs. Belief: a bank of retro knobs (old-epoch effects, whose pooled Σβ ran ~3x what any
    x* ever delivered) is priced by the carrier x factorial cell prior -- what factorial nodes on that carrier have
    measured --; any other bank by the expected gain of its x* (``factorial.forecast``). Moved from the carrier to
    the incumbent by the difference of their multi-seed means."""
    from loop.core import factorial as F
    busy = set(F.open_waves(led, closed=CLOSED_WAVES).values())
    nz, wz = post.fit(post.epoch)
    s2 = (1 - nz.rho[M.COMPOSITE]) * nz.sigma2[M.COMPOSITE]
    out = []
    for carrier, bank in knobs.items():
        ks = F.choose(bank)
        node = led.nodes.get(carrier)
        if not ks or carrier in busy or node is None or node.base != post.epoch or carrier not in post.by:
            continue
        m, v, big_v, n = F.forecast(ks, s2)
        legs = [(g, b.knob.key) for g, b in zip(F.knob_gains(ks, s2), ks) if b.knob.scale == "diff"]
        if len(legs) >= 2:                      # a stack of node diffs: sub-additive, one mechanism counted once
            m = m - sum(g for g, _ in legs) + stack_mean(_leg_mechanisms(led, legs, carrier))
        if all(b.source == "retro" for b in ks):
            m, v, _ = post.prior.cell((cell_carrier(led, carrier), F.FACTORIAL_FAMILY, "factorial"))
        if carrier != inc and inc in post.by:
            (mc, nc), (mi, ni) = M.parent_mean(post.by[carrier], M.COMPOSITE, wz), \
                M.parent_mean(post.by[inc], M.COMPOSITE, wz)
            m, v = m + mc - mi, v + nz.sigma2[M.COMPOSITE] * (1 / nc + 1 / ni)
        n += F.n_blocks(len(ks))                # v3-c7a A2: one carrier control run per block seed
        out.append(Option("wave", carrier, "factorial", m, v, big_v, n * run_h, n))
    return out


def _leg_mechanisms(led: Ledger, legs: Sequence[tuple[float, str]], carrier: str) -> list[tuple[float, str]]:
    """(FRESH_SHRINK x leg effect, mechanism id) per (effect, leg node) stacked on ``carrier``: legs of one mechanism
    (a leg beyond the carrier's or a specific family: :func:`same_mechanism`) share the id of the first of them. The
    one shrink of a wave's stack legs (the gap floor does not shrink a wave again)."""
    ids: list[str] = []
    for i, (_, a) in enumerate(legs):
        ids.append(next((ids[j] for j in range(i) if same_mechanism(led, a, legs[j][1], carrier)), a))
    return [(FRESH_SHRINK * g, k) for (g, _), k in zip(legs, ids)]


def stack_over_parent(led: Ledger, post: Posterior, parent: str, other: str) -> float | None:
    """The prior mean of ``parent`` + ``other``'s idea over ``parent`` (a crossover), both legs on ONE scale: the
    parent's measured gain over its base root and ``other``'s own increment, each x FRESH_SHRINK (the scale the
    measured stack rule applies to); :func:`stack_mean` of the two minus the parent's shrunk leg. So the partner is
    worth a stack only when it beats the parent's discount: ~ own gain > parent leg / STACK_BEST_LEG_FRAC (the median
    stack keeps .74 of its best leg), or by the regression term when the parent's leg is small; one mechanism:
    max(0, other's leg - the parent's).
    None when ``other`` is not measured on the parent's base, or no noise fit (``post.fit`` ValueError)."""
    if other not in post.est or parent not in post.by or led.nodes[other].base != led.nodes[parent].base:
        return None
    root = next((n for _, n, b in reversed(led.bases) if b == led.nodes[parent].base), None)
    try:
        wz = post.fit(led.nodes[parent].base)[1]
    except ValueError:
        return None
    p_leg = 0.0
    if root is not None and root != parent and root in post.by:
        p_leg = FRESH_SHRINK * (M.parent_mean(post.by[parent], M.COMPOSITE, wz)[0]
                                - M.parent_mean(post.by[root], M.COMPOSITE, wz)[0])
    o_leg = FRESH_SHRINK * post.belief(other)[0]
    new = adds_mechanism(led, parent, other)
    return stack_mean([(p_leg, parent), (o_leg, other if new else parent)]) - p_leg


def _screen_options(led: Ledger, statuses: Mapping[str, Status], inc: str, post: Posterior, costs: Costs,
                    run_h: float) -> list[Option]:
    """A ``screen`` option per QUEUED node of the current base whose parent has data (wave points are launched by their
    wave; a failed port is REJECTED, never QUEUED). Belief: an x*'s own prior, a port's original's posterior; a new
    idea node is valued like the idea it came from -- its lane prior's heavy (t3) tail -- so the screens of fresh
    ideas are not starved by the GPU actions on measured nodes."""
    from loop.core.factorial import node_prior
    s2, rho = post.noise.sigma2[M.COMPOSITE], post.noise.rho[M.COMPOSITE]
    out = []
    for nid, st in statuses.items():
        node = led.nodes.get(nid)
        if st is not Status.QUEUED or node is None or node.base != post.epoch or node.parent not in post.by \
                or wave_node(node):
            continue
        tail = False
        if (pr := node_prior(node.notes)) is not None:
            m, v = pr
        elif node.port_of in post.est:
            m, v = post.belief(node.port_of)
        else:
            m, v, _ = post.lanes.cell((post.cell(nid)[0], node.lane, ""))
            tail = True
            if node.lane == "crossover" and node.second_parent and (
                    sm := stack_over_parent(led, post, node.parent, node.second_parent)) is not None:
                m = sm                          # a stack: sub-additive, no credit for a leg of the same mechanism
        if node.parent != inc and inc in post.by:
            dm, dv = post.shift(nid, M.COMPOSITE, inc)
            m, v = m + dm, v + dv
        out.append(Option("screen", nid, node.lane, m, v, s2 * (1 - rho ** 2) / costs.idea_runs,
                          costs.idea_runs * run_h, costs.idea_runs, tail=tail))
    return out


# ---------------------------------------------------------------------------------------------- program priority, ageing
#: a program dose / isolation / opposite-dose node (``loop.core.program``: notes ``ladder=<program node> ...``). The tune
#: batches' ``ladder=+0`` / ``ladder=<number>`` (a letter must follow) are not these
LADDER_RE = re.compile(r"(?:^|\s)ladder=[A-Za-z]")


def program_child(node: Any) -> bool:
    """An operator-intended node: a program arm (``program_arm=1``), a program ladder dose / isolation child, or a
    manual run's confirm node (``local_confirm=``). They are run on purpose, so they are screened first
    (:func:`program_nodes`), are never aged out or dropped as starved, and do not count as idea overproduction in the
    brief throttle (``alloc.backlog``)."""
    notes = getattr(node, "notes", None) or ""
    return PROGRAM_TAG in notes or LOCAL_TAG in notes or LADDER_RE.search(notes) is not None


BANK_RE = re.compile(r"(?:^|\s)bank=(\S+)")


def operator_bank_node(node: Any) -> bool:
    """A node of an operator-bank idea (notes ``bank=<id>`` from ``loop.idea_banks``: not a ``scout-`` bank) or a
    root-audit arm (``root_audit=1``): ``alloc.keep_bank_nodes`` exempts them from the backlog drops."""
    notes = getattr(node, "notes", None) or ""
    if re.search(r"(?:^|\s)root_audit=1(?:\s|$)", notes):
        return True
    m = BANK_RE.search(notes)
    return m is not None and not m.group(1).startswith("scout-")


def program_nodes(led: Ledger, statuses: Mapping[str, Status]) -> set[str]:
    """QUEUED program arm and program ladder-dose nodes (:func:`program_child`): :func:`rank`'s ``priority``. A
    manual run's confirm node keeps the priority while SCREENING too (11-09: the operator's measured winners -- their
    seed extensions go first, not only the first screen)."""
    return {nid for nid, n in led.nodes.items() if program_child(n) and (
        statuses.get(nid) is Status.QUEUED
        or (LOCAL_TAG in (n.notes or "") and statuses.get(nid) is Status.SCREENING))}


def aged_queued(led: Ledger, statuses: Mapping[str, Status], now: float, max_age_h: float | None = None,
                touched: Mapping[str, float] | None = None) -> list[str]:
    """QUEUED nodes (derived: no run row at all) older than ``max_age_h`` (``QUEUED_MAX_AGE_H``; <= 0 = none): they sat
    behind the backlog and will not screen (live 10-21: 218 stored-queued nodes older than 96 h, 4 of them never run).
    Age = the later of the node's created time and ``touched`` (store ``updated_ts``: a restored node gets a fresh
    window). Exempt: program arms, wave points, carrier-introduced / ported nodes whose notes mark them so."""
    h = QUEUED_MAX_AGE_H if max_age_h is None else max_age_h
    if h <= 0:
        return []
    out = []
    for nid, node in led.nodes.items():
        if statuses.get(nid) is not Status.QUEUED or program_child(node) or wave_node(node) \
                or nid == ROOT or node.port_of:
            continue
        t = max(float(node.created_ts or 0.0), float((touched or {}).get(nid, 0.0)))
        if t > 0 and now - t > h * 3600.0:
            out.append(nid)
    return sorted(out)


def age_out(store: Any, ids: Sequence[str], now: float, max_age_h: float | None = None) -> list[str]:
    """Mark ``ids`` aged out the way ``alloc.drop`` drops a backlog node: stored status REJECTED (derived VOID), reason
    ``dropped_backlog: aged_out_queued ...``, an ``aged_out=1`` note and a ``queued_aged_out`` event -- the node, its
    commit and notes stay, so :func:`restore_aged` brings it back. Not a measured failure."""
    from loop.contracts import NodeStatus
    done = []
    try:                                  # nodes an open seed block still waits on: never aged out from under it
        held = {n for b in store.seed_blocks() if getattr(b, "open", False) for n in b.node_ids}
    except Exception:  # noqa: BLE001 - no block view: the run-row check stands
        held = set()
    for nid in ids:
        node = store.get_node(nid)
        if node is None or node.status is not NodeStatus.QUEUED or nid in held:
            continue
        try:                              # ... or one that launched since the statuses were derived (a run row exists)
            if store.runs(node_id=nid):
                continue
        except Exception:  # noqa: BLE001 - no run view: the derived QUEUED stands
            pass
        age = (now - float(node.created_ts or now)) / 3600.0
        store.update_node(nid, status=NodeStatus.REJECTED, notes=f"{node.notes or ''} aged_out=1".strip(),
                          reject_reason=f"{AGED_REASON}: queued {age:.0f} h, never ran (not a measured failure)")
        store.log("queued_aged_out", node_id=nid, age_h=round(age, 1),
                  max_age_h=QUEUED_MAX_AGE_H if max_age_h is None else max_age_h)
        done.append(nid)
    return done


def restore_aged(store: Any, ids: Sequence[str] | None = None) -> list[str]:
    """Undo :func:`age_out` (and ``alloc.drop``): every (or the given) ``dropped_backlog`` node goes back to QUEUED with a
    fresh age window (its ``updated_ts``)."""
    from loop.contracts import NodeStatus
    out = []
    for node in store.nodes():
        if not (node.reject_reason or "").startswith("dropped_backlog") or node.status is not NodeStatus.REJECTED \
                or (ids is not None and node.id not in ids):
            continue
        store.update_node(node.id, status=NodeStatus.QUEUED, reject_reason="",
                          notes=re.sub(r"\s*aged_out=1", "", node.notes or "").strip())
        store.log("queued_restored", node_id=node.id)
        out.append(node.id)
    return out
