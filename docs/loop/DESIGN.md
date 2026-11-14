# loop/ — a lean, task-agnostic research loop

Design of record. Records and the adapter seam: `loop/contracts.py`; config schema: `loop/spec.py`.
Every feature exists in its simplest working form: bandit, islands, tournament, near-dup, relevance,
crossover, tag/family shrinkage, lever trends, postmortems, big reviews, first-principles reviews,
bounded self-tuning, e-values, held-out audit, designed batches.

## 0. Ground rules (enforced in code)

1. **The task spec is the single source of truth**: surface, axes (`slots.vector`, else `slots.weights`
   keys) and weights, sigmas, seeds, families and the `loop:` block, decoded strictly by `loop.spec`.
2. **Measured noise or no run.** `run` refuses (exit 2, naming `calibrate`) while any sigma is
   `UNMEASURED`. The noise model comes from the ACTIVE spec.
3. **Clean slate.** Every stored row carries `task_id`; the store reads only its task, refuses a foreign
   write, and raises `ForeignTaskError` on an untagged (v1) row. Everything is read through the store.
4. **No root `config` import under `loop/`**, except the lazy `llm` / `agent.llm` imports in
   `SdkLLMClient`. Tests import every module and assert it.
5. **The carried lesson.** A change that breaks model init is much worse: a negative prior on family
   `architecture` and a hard init-compat check before any GPU spend.
6. **Code computes evidence; LLMs write explanations.** Reviews add hypotheses, probes and a capped,
   decaying explore bias; no LLM output edits a number. Everything (rejections, voids) is kept as rows.

## 1. Architecture

```
 driver (plain Python, no LLM; a thread pool for LLM work, SLURM for GPU work, all state in the store)
   +-- ExecutionAdapter (sim | nanopath_v2): worktrees, diff, smoke, init_compat, submit/poll/cancel
   +-- SqliteStore        +-- ComputedWorldModel: noise, shrinkage, utility/veto, levers, e-values
   +-- PortfolioSelector: bandit arms, islands, intake, tournament, Thompson queue
   +-- agents: LLMResearcher, LLMAnalyst (postmortem, big review), LLMJudge, briefs
```
`run`, `calibrate`, `review` hold an exclusive `flock` on `<db>.lock` (a second instance exits 3).

## 2. Data model (SQLite over the `kernel.ledger` connection)

DB: `<loop.state_dir | spec.home/loop | --state-dir>/<task_id>.db`. One table per record, columns
derived from the dataclass (structured fields as JSON), all indexed on `task_id`; a field added later
gets its column (at its default) when an older DB is opened.

| table | key | notes |
|---|---|---|
| `nodes` | id | commit, parent_id, second_parent_id, island, strategy, operator, family, mechanism_tag, diff_hash, diff_path, hypothesis_id, status, levers (registered), debug_depth, confirm_alt (frozen δ1), confirm_e, prior_bonus, embedding, reject_reason |
| `runs` | `{node}:{stage}:{seed}`, UNIQUE(node, seed, stage) | status, void_type, scores, aggregate, submitted_ts (first), gpu_h, output_dir, log_tail, attempt, pool, mitigation (OOM ladder step, §5) |
| `hypotheses`, `hypothesis_links` | id; (hypothesis, node) | kind trial/meta, claim, tag, family, predictions, falsification, levers, source, review_id (evidence is computed, not stored); link sign ±1, auto |
| `postmortems`, `handoffs`, `reviews` | id | reviews carry `probe_briefs` as `(meta_id or None, brief)` pairs and a raw explore bias, `design` (a batch's knobs, JSON), `selftune` (asked knob changes, JSON) and `proposals` (operator rule changes) |
| `arm_pulls`, `state`, `log` | rowid / key | bandit pulls; frozen confirm book, hypothesis e-value lists, noise and variance caches, islands, audit book, design book, self-tune overrides, acked proposals; event log (incl. `meta_decision`, 6.3) |

Node `root` (strategy `baseline`) holds the baseline runs. Node statuses: `rejected | queued -> screening
-> (screened_out | vetoed | promoting) -> (screened_out | vetoed | confirming) -> (confirmed |
not_confirmed | vetoed)`; any active stage -> `void | crashed`; `confirmed -> audit_failed`. Runs:
`submitting -> running -> done | void | crash`; void types `infra` (retried) and `eval` (no scores).
Stages: `baseline, calibrate` (root), `screen, promote, confirm`, `parent` (a parent's control run at a
child's seed), `audit` (held out: the world model never reads it).

## 3. World model (`loop/world/`, computed; a belief is a query)

Targets t ∈ {panel} ∪ axes; weights w from `slots.weights`, normalised.

**3.1 Pairing.** A node's control at seed s is root's baseline run at s when its parent is root, else its
parent's `parent`-stage run at s — never the parent's selection or confirm runs. Delta `d = y_i(s) −
y_ctl(s)` where both are `done` (panel via `kernel.stats.paired_delta`); voids/crashes never count and a
void/crashed node contributes nothing.

**3.2 Noise.** σ_t starts at the declared sd; once `noise_min_df` within-node df exist, `σ² = (df₀σ₀² +
SS)/(df₀ + df)`, df₀ = calibration seeds − 1. A confirming node's confirm runs never feed it.

**3.3 Increments and utility.** Every child paired against one control RUN shares its deviation
`b_c ~ N(0, σ²/2)`. Each delta is corrected to `d − b̂_c(−i)`, `b̂` the shrunk mean of the OTHER
children's deltas on that run (each `N(b_c, σ²/2 + τ_s²)`, `τ_s` the pooled within-control spread less
σ²/2), with variance `σ²/2 + Var(b̂)` (one child: `b̂ = 0`, variance σ²). The selection increment is
their precision-weighted mean over screen+promote seeds. Confirm and audit deltas are corrected the
same way (audit siblings: other audited nodes minus their estimate); the frozen σ bound covers them. Because it is paired against the parent's own controls it does not depend on the
parent's posterior; `U_i = U_{p(i)} + ε_i` (root ≡ 0; an unscored parent's U is its parent's U plus the
(family, tag) predictive).

**3.4 Shrinkage.** `ε_i ~ N(μ_g, τ₁²)`, `μ_g ~ N(μ_f, τ₂²)`, `μ_f ~ N(m_f, τ₃_f²)` (declared family
priors, else N(0, τ_family²)). Exact posterior by one two-pass Gaussian belief propagation on the tree.
Cold start (`< prior.refit_min` scored): τ₁ = τ₀, τ₂ = τ₃ = 0. Every `prior.refit_every` newly scored
nodes: DerSimonian–Laird per level, clipped to `[tau_floor, tau_max]`; a level DL cannot estimate keeps
its current τ (τ₀ when cold), never the floor. Tag means are published as `family/tag`. Predictive for
an unrun (f, g): seen tag -> `N(μ_g, Var + τ₁²)`; else family (+τ₂²); else its prior.

**3.5 Veto and incumbent.** `P(U>0) = Φ(E/sd)`. Veto iff some axis has `P(θ_a < −harm_sigmas·σ_a) ≥
veto.prob`. The incumbent is the CONFIRMED, non-vetoed node with the best confirm-only estimate (> 0),
else root; `P(U > U_inc)` compares confirm-only estimates wherever a node has them.

**3.6 Winner's curse.** Selection uses screen+promote seeds; confirmation uses fresh `confirm_pool`
seeds; the held-out audit uses `audit` seeds; all four sets are disjoint (`SeedPlan`). A node's
`confirm_estimate` shrinks its confirm increments by its cavity prior, on top of its parent's
confirm-only U. The bandit gain and the incumbent use only confirm estimates.

**3.7 Confirmation e-value.** At confirm start freeze `δ₁ = max(E[ε], delta_min_sigmas·σ_P)` (the driver
stores it as `confirm_alt`) and a σ UPPER bound (90% one-sided chi-square, Wilson–Hilferty, at the df in
force), both in state `world.confirm`. `E = Π exp((c_s δ₁ − δ₁²/2)/σ²)` over confirm increments. With
σ ≥ σ_true, E is a test martingale under H₀ (ε ≤ 0) that later refits never move. Confirmed iff
`E ≥ 1/alpha_confirm` (0.05) and not vetoed; not confirmed iff `E ≤ futility_e` or seeds are spent.

**3.8 Hypothesis e-values.** Primary target: panel if predicted, else the first predicted axis. When a
linked trial is first scored, its estimate `x = sign_link·sign_pred·x̂` is frozen, where `x̂` pools its
selection-seed (screen + promote) paired deltas, each corrected for its control run's deviation using
ONLY siblings submitted before it (`d − b̂(earlier)`, variance `σ²/2 + Var(b|earlier)`: the measurement
variance, never a shrunk posterior), plus `τ_node² + τ_tag²` for a META hypothesis (its claim is the
mean over implementations). With `δ₁ = clip(predicted size, delta_min_sigmas·σ, 2σ)` and
`LR = exp((δ₁x − δ₁²/2)/v)`: `e_for = min(cap, LR)` (H0: effect ≤ 0) and `e_against = min(cap, 1/LR)`
(H0: effect ≥ δ₁); cap = 4 for META (⌈log₄(K/alpha)⌉ concordant trials to decide: 5 at K = 66), 20 for TRIAL (> 1/alpha). The
causal correction makes each e-value valid given earlier trials, so shared-control siblings do not
multiply as independent, and the Gaussian LR is valid under the adaptive promote. Status = fold over
that list in submission order, stopping at the first crossing of `K/alpha` for a META hypothesis (K =
the number of META hypotheses registered: Bonferroni on e-values, so the chance that ANY meta shows a
false verdict is ≤ alpha, whatever the dependence between metas sharing trials) or `1/alpha` for a
TRIAL one, or at `max_trials` (24; inconclusive). The running product is a test supermartingale, so
stopping at the crossing is anytime-valid; `max_trials` is only a cost bound. A TRIAL hypothesis closes
on its one trial and its verdict is never revoked; a META status is a report at the current family size
(a later review's metas raise the bar), while the book's rows stay append-only. Only trials first
submitted after the hypothesis was created count. (The retired `κp^{κ−1}` calibrator was unbounded:
one z = −10.6 gave E = 1e15. `world.hypotheses.v2` moved to bounded LRs; `world.hypotheses.v3` to the
links below and the family bar. Each move re-derives the book once — TRIAL rows carried over verbatim —
and logs `hypotheses_rederived` with every status or trial list that changed.)

**3.9 Meta-hypotheses** carry a lever vector M and link later trials (not rejected, not a designed
batch's factorial point), strongest first: EXPLICIT — the node was researched from one of the meta's
own review probe briefs (an `auto=False` link written at intake), or its registered hypothesis is or
names the meta id; MECHANISTIC — the node's targets (leaf names of its changed config keys plus the
snake_case identifiers on its diff's changed lines) overlap the meta's key names (identifiers and
known config keys named in its claim, falsification and own probe briefs, frozen in
`world.meta_keys` the first time the meta is seen); LEVER — `L·M ≠ 0`, used only while the meta has
no explicit or mechanistic link. Sign = sign(L·M) when non-zero, else +1.

**3.10 Lever trends.** Vocabulary: `training_strength, regularization, effective_lr,
augmentation_strength, capacity, objective_difficulty, data_curation, readout` + `levers.extra`. Per
target, Bayesian ridge of each trial's residual `e − E[family mean without its own tag]` on its
REGISTERED {−1,0,+1} vector, weight `1/(v + Var + τ₁² + τ₂²)`, priors `β ~ N(0, prior_sd²)`. So the trend
explains what differs BETWEEN mechanisms, and `clip(L·β, ±levers.cap)` enters a candidate's predictive
only when its tag is unseen (a seen tag's mean already carries its levers): counted once.

## 4. Selection (`loop/select/`)

**4.1 Arms.** exploit (a frontier parent by Thompson draw of U − σ_P·log(1+offspring)), explore (a family
that is under-sampled and wide), crossover (two confirmed non-vetoed nodes of different families, not
tried before), revisit (debug a crash below `max_debug`, else improve a near miss `0.3 < P(U>0) <
p_promote`). A debug child's parent is the crashed node's parent (its control) and the crash is its
`second_parent_id`; it diffs against that control. Arms with no plan are masked. Review probes are served first, each once.
**Winner crosses** (standing exploitation, outside the bandit like fix-and-continue): whenever >= 2 CONFIRMED
non-vetoed nodes exist, every untried pair where neither contains the other (built from it, or a config-only
change set that is a subset of the other's) gets ONE crossover request (`driver.winner_cross`), most complementary
first (different family, then least changed-key overlap), capped only by researcher concurrency. A merge that
confirms joins the pool (compounding). Logged `winner_cross queued A x B`.
**Fix-and-continue.** Outside the bandit, every non-design node that closes CRASHED (own run, after the OOM
ladder of §5) gets ONE DEBUG request (strategy revisit) at once, ahead of the tuning and bandit requests
(within `researcher.max_concurrent`), while its `debug_depth < max_debug`; book `driver.debug` {node: request},
an outage frees the entry. The researcher sees the crash run's log tail (and the ladder step it already had)
plus the diff. Nodes that crashed before this existed are queued the first time the driver sees them.

**4.2 Bandit.** Discounted Gaussian Thompson over the arms, rebuilt from `arm_pulls`; floor mixing keeps
≥ `floor` per arm. Reward on close: `(1{confirmed}·max(0, U_conf − U_conf,inc)/σ_P + info_weight·IG) /
max(gpu_h, gpu_h_min)`, IG = ½ log variance ratio of ε (prior: the (family, tag) predictive before) and of each registered
lever's β.

**4.3 Islands.** Off until `split_at` scored nodes; then families are split greedily into `n` islands
(a new family joins the smallest). Migration is a parent pool: a confirmation joins every other island's
frontier. Every `reset_every` confirmations the weakest island (by its OWN best confirmed U) restarts
from the global incumbent.

**4.4 Intake** — `admit()` returns `""` or the first rule that fired: empty_diff, surface_locked,
surface_not_writable, protected_key, exact_dup (diff hash), near_dup (hashed 1–3-gram cosine of claim +
changed lines > `near_dup_cos`), parent_contains_knob (every changed key already equals the ancestors'
`config_diff`), config_noop, unmapped_path / ambiguous_family (every changed key and path must map to
exactly ONE family rule; a `*` glob is a catch-all), unknown_family, forbidden_reference (an added,
non-comment line of a diff touching a family with `forbid` regexes matches one), multi_family /
family_mismatch (only crossover and moonshot may span families; a `combine` family may pair with ONE other
family and is then the declared one; a COMPOUND -- 2 families where every changed key of one equals an already-positive knob, shrunk P(U>0) >= `intake.compound_min_p` (0.8) -- is admitted, declared as the NEW family, tagged `compound=<a>+<b>` in notes, grouped as `shape` in the quality table, and the stackable knobs are listed in the context pack and the `check` reject), init_compat_missing / init_incompatible (an `init_check` family or a change
under `init_compat.paths`), irrelevant (the Judge says the diff does not test the claim). A designed
batch's node (4.6) skips near_dup, near_dup_inflight, the family rules and the Judge; a design point also
skips exact_dup, and no candidate is an exact dup of a point.

**4.5 Tournament and queue.** Admitted nodes are shuffled onto a ring; ≤ `tournament_pairs` Judge
compares score wins − losses, scaled to a prior bonus of at most 0.5 σ_P. Queue order: Thompson draw of
`U_parent` + a draw of the predictive + bonus + review bias (each review's family bias, decayed over
`bias_decay_evals`, capped at 0.5 σ_P each and in total).

**4.6 Designed batch** (`loop/design.py`). Started when a review returns `design` knobs, or when
`design.stall_closes` closes passed with no confirmation and no batch since (0 = review only). Knobs
(4..10): the review's `{key, low (default root's), high, levers}`, topped up with the most-probed config
keys not designed before (outside `init_check` families); an auto knob's high is a `design.step` move
from root's value toward the side its probes' shrunk U favours (a float in (0.5, 1) moves its distance to
1; a bool flips; a non-number takes the best positive probe's value), its levers are the probes' levers
signed by that move. Design: 2^(K−p) = 16 (K ≤ 8) or 32 runs (K > 8, or `design.runs: 32`), standard
resolution-IV generators, even words negated so row 0 is all-low (= root when every low is root's
value: root's own runs are that point). Every other row is a config-only node (strategy/family `design`,
operator none, levers = its moved knobs' levers) that screens on ONE shared block vs root, prefers
preemptible pools, closes `screened_out` after screen, gets no postmortem and is never a parent. When all
points are closed: OLS of the panel score on a per-seed intercept (the per-control correction: every
point at a seed shares that control run) + the K ±1 columns; effect = 2β ± se from the residual
variance; each 2FI alias chain's contrast on the residuals is flagged at |c| ≥ z·se. One stacked node
(operator draft, family `design`, a trial hypothesis predicting the summed gain) sets every significant
knob to its better level and takes the normal screen/promote/confirm/audit path. **Memory pre-smoke:** when
a knob is a memory key (`size|views|batch|width|_dim|depth|blocks|layers|heads`), the heaviest row (most such
knobs at their larger level, heavier than root's) is smoked with its own batch shape (`smoke(memory=True)`)
in the thread pool before ANY point launches (`held`); an OOM climbs the §5 ladder, and the step that passes
(the strongest when none does) is written as `oom=<step>` on every point; any other smoke failure just
releases the batch. Points that still OOM get the run-level retry. Points are scored
nodes, so their levers feed the lever trend; results show in `status`, the review prompt and the
researchers' "what we've learned".

**4.7 Novelty engine** (`loop/novelty.py`; `novelty: {enabled, sweep_redirect, weight}`), against drift into
hyperparameter sweeps. *Operator separation* (AIRA/AIDE): on an explore / moonshot request (bank ideas too; never
exploit, tuning, debug, crossover, design) a diff that only sets EXISTING config keys to other numbers is rejected
`sweep_only` (the researcher's `check` says so) and its knobs + levels (levers = its hypothesis') join the design
knob pool `novelty.knob_pool`, which tops up the next batch after the review's knobs (4.6; consumed when used).
*MAP-Elites cells* (AlphaEvolve/OpenEvolve): each node's `mtype=` note is one of loss_objective, data_sampling,
augmentation, architecture_addon, training_dynamics, readout, regularization, optimization (the researcher's
declared `mechanism_type` kept when the diff shows a sign of it, else inferred from paths + changed lines; older
nodes are classified from their diff); a cell = family x type. The arm's own explore / moonshot DRAFT request
targets the emptiest type (of its family; moonshot: over all), and its prompt shows the fill counts and the cell's
best node (claim + changed lines) as inspiration; the scout prompt lists the types by fill. *Novelty bonus*
(novelty search / ShinkaEvolve): a closed, run node (not void / crashed / rejected: near-0 GPU-h would
inflate it up to weight / gpu_h_min) whose diff touches code earns `weight` (0.5) x (1 - max cosine to
every earlier node's claim + diff embedding) per GPU-h, kept in `novelty.bonus` apart from the raw pull; the
audited component `novelty` (6.4) decides per request whether the arm choice sees it. *Personas* (co-scientist):
those requests rotate theorist, contrarian (shown the last first-principles review's broken / unknown assumptions),
cross_domain, literature (web sessions only), noted `persona=`; loop.meta's `persona` record tracks which one hits
(bias `meta.persona_bias`). Every ideation prompt asks for a one-line self-critique (real mechanism? already in the
recipe? init-compatible?) before finish; a web session's finish bounces until `hypothesis.prior_work` names the
prior work and why it is not already in the recipe (appended to the claim).

## 5. Seeds and promotion (`loop/seeds.py`, pure)

Screen on the node's block (a parent's first `seeds.block_children` (4) children share block 0 =
`seeds.screen`, later blocks sit above every declared seed; a node with screen runs keeps its block) -> `P(U>0) ≥ p_promote` -> `promote` seeds -> `P(U>U_inc) ≥ p_confirm` ->
confirm on `confirm_pool` seeds the parent did not confirm on, one at a time. Every compared seed needs
the control run (3.1); it is scheduled first. An infra void is retried up to `caps.infra_retry` times; a
control or audit run's crash or eval void is retried once (deterministic). Then the dependent node is
`void` (`void:partner`, typed infra, no postmortem). Own crash or eval void closes the node (an eval
void on a confirm seed: `not_confirmed`); a confirm seed whose infra retries are spent is skipped.
**OOM ladder.** Before any verdict, a run that crashed or infra-voided with a GPU OOM in its log tail
(`contracts.is_oom`) of a live node (or a control/root run) is relaunched at once as the SAME row (attempt +
1) at the adapter's next numerics-neutral step (`oom_ladder(commit)`, cumulative; nanopath: only
`expandable_segments` = `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). The step is stored on the run
(`mitigation`) and as `oom=<step>` in the node's notes, and every later run of the node starts there. No new
node, so a step MUST leave the math alone: activation checkpointing does not (the FLOP counter counts the
recompute and train.py keys LR/WD/teacher/EMA/KDE schedules on `train_flops / max_train_flops`); a node that
needs it gets it as its own code change via debug. Ladder spent: the crash
stands and fix-and-continue (4.1) takes it. On the first tick of a process, a CRASHED node (an idea node, or a
point of the running batch) whose crash runs are all OOM with the ladder unspent is reopened at the stage it
crashed in, its crash runs owed infra retries at the next step.

**Held-out audit.** Every tick asks `labless_ready` for the incumbent, then every CONFIRMED node, so each
unaudited confirmed node's audit is queued (the incumbent always audited next); queued nodes are re-run, one
at a time, with root on the `audit` seeds. The claimed gain and the σ upper
bound (as 3.7) are frozen at audit start; the likelihood-ratio E against 1/alpha_confirm decides. A failure marks the chain from that node up to its
last audited ancestor `audit_failed`. Only an audited node may go to Labless: `loop.labless` submits each passed, still-confirmed node
once (spacing `min_hours_between`, 0 = none; a failed try backs off >= 1 h; success = a stored `submission_id`). `labless.recorded` entries
(submissions made by hand, e.g. root's baseline) are written into that book at startup and never
resubmitted. `status` and the `hb` line show `labless: <gate state>`.

## 6. Driver (`loop/driver.py`, `loop/__main__.py`)

**Tick** (every `heartbeat.poll_s`): reap orphan research worktrees (first tick of a process) -> harvest
LLM work -> reopen pre-ladder OOM crashes (first tick, §5) -> resolve `submitting` rows -> poll (OOM relaunches) -> refit -> transitions (a closed node's running jobs are
cancelled) and bandit rewards -> postmortems (loss after screen with `P(U>0) < 0.3`, not_confirmed,
vetoed, crash, eval void) and the big review (`every_confirmations`, `every_evals`, or a stall of
`stall_evals`; a first-principles review, 6.1, takes precedence when due) -> designed batch (4.6) and its
memory smoke -> audit -> fill slots -> fix-and-continue DEBUG requests (4.1) -> winner crosses (4.1) -> tuning requests ->
researchers while `queued < queue_low_water`.

**Slots.** Capacity is the sum of `loop.slurm.pools` slots (else `loop.slots`). Owed retries, the
audit, active nodes' needs (controls, confirm, promote, screen; by P(U > U_inc)), then the Thompson head
of the queue. Each launch/retry takes the first pool, in declared order, with a free slot (stored on the
run row); `status` and `hb` show use per pool.

**Resume.** A run row is written `submitting` BEFORE `adapter.submit`; restart resolves it through
`adapter.find` before any resubmit; UNIQUE(node, seed, stage) refuses a double launch; a retry reuses the
row (attempt + 1, first `submitted_ts` kept). Liveness is only `adapter.poll`; a void/crash result is
cancelled. A postmortem or review counts as done only once stored: failures retry up to 3 times, an
outage (`BackendUnavailable`) is not counted. A researcher that hit an outage creates no node; a
finished research whose intake hits a judge outage is kept and re-intaken next tick, and a tournament
outage only skips the bonus. `status` and `--dry-run` fit on an in-memory state overlay: nothing is
written.

**6.1 First principles and self-tuning.** A FIRST-PRINCIPLES review (role `review`, prompt
`first_principles.md`, trigger `first_principles`) runs after `review.fp_every_reviews` (3) big reviews or
`fp_stall_closes` (30) closes without a confirmation, at most once per `fp_min_hours` (12; a given-up one
counts). It lists the loop's assumptions in five lenses (measurement, ml_problem, search, constraints -- hard
external vs self-imposed --, loop_system), marks each holds/broken/unknown with evidence, and returns
meta-hypotheses, `[moonshot]` briefs, assumption-test probes (proxy validity, A/A controls), self-tune asks and
operator proposals. Extra code-computed evidence: throughput/slots/GPU-h by outcome, declared vs current sigma
and the per-dataset sd over root's seeds (the last `probe` event of each root run's `metrics.jsonl`), and the
latest root run's summary plus first/last training record (final lr / wd / momentum, flop vs sample fraction).
It is stored as an ordinary review (assumptions as findings; probes served like any review probe).
`loop/selftune.py`: big and first-principles reviews may ask `[{knob, value, reason}]`. A WHITELISTED knob
(bandit floors 0.05-0.4 with the five arms' floors summing <= 0.9, crossover_min_p 0.75-0.95,
researcher.max_concurrent 1-8, review.every_evals 5-30 / stall_evals 5-40, promotion.p_promote 0.6-0.9,
seeds.block_children 2-8, design.stall_closes 0-60 / z 1.5-3, `moonshot_tune.advocate_bias` and the `meta.*_bias` knobs -1..1, 6.3) is clamped and stored as a runtime override in
state (logged `selftune`; the spec file is untouched); every other knob -- budget, locks, eval, seed cap,
alphas, Labless, init-compat, SLURM -- is refused (`selftune_refused`) and becomes an operator proposal. Each
tick the driver installs `ctx.cfg = spec ⊕ overrides` (every component reads `ctx.cfg`; `status` shows the
same). Open proposals show in `status` and the `hb` line (`proposals: N awaiting operator`) until
`python -m loop ack --task T <id>...|all`.

**6.3 Meta-learning** (`loop/meta.py`). Every LLM decision-maker is scored against what the loop later
measured, in code (no LLM call), from the records the loop keeps (hypotheses, postmortems, reviews, the design
and tuning books) plus a decision log (`meta_decision` events: tournament bonuses, relevance flags, postmortem
and review decisions, which review a probe request served). Components and outcomes: researcher (predicted
panel sign/size vs the shrunk increment, `quality.predictions`), judge (not-relevant flags: flagged vs unflagged
hit rate), tournament (bonus vs measured increment, Spearman; before the log: the stored `prior_bonus`),
postmortem (`under_tuned` vs whether the node's tuning pass beat it; `suggested_next` followed by a later node
-- a child with >= 20% or any node with >= 40% of its content words -- and whether that node hit), review and
first_principles (their metas' e-value verdicts; hit rate of the nodes their probes spawned vs the loop's;
closes and hit rate 12 h before vs after each self-tune change), design (fraction of knobs significant,
review-picked vs auto), crossover (crossover/compound nodes vs their best parent's U), advocate (moved from
`loop.tune`: grants that promoted). `track_record(component)` = headline, GPU-h per success, 2-5 right and
wrong calls, a calibration note and the bias line; it is in that component's prompt (the judge's relevance and
compare questions get theirs, refreshed by the driver at each intake; a crossover researcher also sees the
crossover record; the big review also sees the design record). Each has a bias knob `meta.<c>_bias` (advocate:
`moonshot_tune.advocate_bias`) in [-1, 1], self-tunable, rendered at |b| >= 0.25 as "be MORE/LESS X than your
default". The scorecard (one line per component + its knob) is in `status`, the dashboard's World-model tab and
both review prompts, which may move the knobs through `selftune`.

**6.4 Causal component audits** (`loop/components.py`). Each OPTIONAL component -- tournament, relevance judge,
postmortems (research prompts / failure briefs, `under_tuned` -> tuning pass), review probes, lever-trend prior,
design auto-knobs, crossover pair heuristic, advocate grants, scout (runs, its ideas), web research, idea banks,
novelty bonus in the arm choice (4.7) -- sits behind `components.use` at each decision it influences; with probability = its ablation
rate (`components.ablation_rate` 0.15, per-component `rates`) the decision is bypassed (seeded draw, event
`component_gate`, printed `ablate <component> <decision>`). Units are the nodes the decision produced (its request's
nodes, the queue heads a ranking launched, a batch's stacked node; advocate / `under_tuned`: did the lineage hit).
The audit (with every first-principles review, and `python -m loop audit-components`) compares WITH vs WITHOUT:
screened hit rate (shrunk P(U>0) > 0.5) difference ± se (inverse-variance pooled within bypass-rate strata, so
a mode change is never confounded with time), confirmations, GPU-h per success; verdict useful (>= +2 se)
/ harmful (<= -2 se) / no_measurable_value / insufficient (< `min_ablated` 30 ablated decisions). Until ablations
accumulate, a component's own log may stand in ("natural"): judge flagged vs unflagged (0 flags = never changed a
decision), tournament top vs bottom bonus half. Actions: no value -> `down` (bypassed at `down_rate` 0.5), at >= 2 x
min_ablated -> `off` (still used on `reprobe_rate` 5%, so it can return); harmful -> off; useful -> on
(`components.modes`). On a ledger copy this turned the tournament (hit 29/50 top vs 29/50 bottom, n=100)
and the judge (0 of 189 flagged) off. Statistics gates, fresh-seed confirm, the audit, locks, budget, the eval guard,
the Labless gate and init-compat are not components (`use` refuses them). Table in `status`, the dashboard's World
model tab and (concise) the review prompts.

**CLI.** `python -m loop run|calibrate|status|review|ack|dashboard|audit-components --task T [--state-dir D] [--dry-run] [--max-ticks N]
[--fake-llm] [--seeds k]` (calibrate default: 8 seeds); `run --dry-run` prints one tick's plan (or the calibration plan) only.

**6.2 Dashboard** (`loop/dashboard.py`). One self-contained HTML page (inline CSS/JS, no network, dark/light,
phone width; every stored text escaped: LLM output is untrusted): header (incumbent, confirmed, best screened,
slots per pool, arm shares, Labless gate, alerts, LLM routing), the IDEAS feed (newest first, filterable by
status/family/brief/arm/text: claim, per-axis predictions vs shrunk deltas, falsification, handoff, status
timeline from `node_transition`, U ± sd, P(U>0), veto axes, postmortems, the stored diff), REVIEWS (findings,
meta-hypotheses with e-values and trials, probes used/pending, self-tune, proposals open/acked), WORLD MODEL
(lever trends per target, family/tag posteriors, designed batches, `quality_groups`, the meta-learning scorecard, a predicted-vs-measured
SVG) and OPERATIONS (runs in flight, recent void/crash, last 24 h). The driver renders it from the tick's
already-fitted snapshot every `dashboard.every_ticks` (1; 0 = off) to `<state_dir>/dashboard/index.html`
(temp file + rename); a failed render is logged, never fails the tick. The page reloads every 60 s, keeping
filters (URL hash), scroll and open sections. `python -m loop dashboard --task T [--state-dir D] [--out F]`
renders once read-only; `python -m loop serve --state-dir D [--port 8765]` serves `<D>/dashboard` on
127.0.0.1 only (GET/HEAD, no listings).

## 7. Calibration (`loop/calibrate.py`)

Resumable. Root at `seeds.calibration` (default screen + promote + first 4 confirm seeds = 8; the
noise prior's df₀ uses the same default, so `--seeds k ≠ 8` should be declared as `seeds.calibration`) × 2 replicates
(`baseline` — they stay root's baselines — and `calibrate`). Per target: `seed_sd`, `replicate_sd`, paired
sd `√2·seed_sd`, rewritten line by line into `noise.seed_transfer_sd`, `noise.replicate_sd`,
`loop.panel_sd`, `loop.axis_sd.*`, stamped `# measured <date> by loop calibrate`; a zero sd is skipped.

## 8. The `loop:` block (defaults)

```yaml
loop:
  execution_adapter: loop.adapters.sim:SimAdapter   # module:Class, constructed with the context
  slots: 8                  # state_dir: "" -> spec.home/loop; not with slurm.pools (they set capacity)
  slurm: {time_limit: "", pools: []}   # pool: {name, account, qos, partition, slots: 16, preemptible: false}
  labless: {enabled: false, min_hours_between: 4, recorded: {}}   # recorded: {node: {submission_id, run_id}}
  panel_sd: UNMEASURED
  axis_sd: {<axis>: UNMEASURED}
  seeds: {screen: [11, 12], promote: [13, 14], confirm_pool: [101, ..., 106], audit: [201, ..., 204], calibration: []}
  promotion: {p_promote: 0.7, p_confirm: 0.7, alpha_confirm: 0.05, futility_e: 0.1, confirm_max_seeds: 6, delta_min_sigmas: 0.5}
  prior: {tau0: 0.005, tau_floor: 0.0005, tau_max: 0.05, refit_every: 10, refit_min: 10, noise_min_df: 10,
          families: {architecture: {mean: -0.02, sd: 0.005}}}
  veto: {harm_sigmas: 1.0, prob: 0.8}
  hypotheses: {alpha: 0.1, kappa: 0.5, max_trials: 24}
  levers: {extra: [], prior_sd: 0.005, cap: 0.005}
  families: {<name>: {keys: [glob], paths: [glob], init_check: false, combine: false, forbid: [regex],
                      forbid_config: ["file:dotpath"]}}   # forbid_config: names/paths under that block of
                      # spec.repo's config join forbid at load (unreadable = load error)
  init_compat: {paths: []}
  islands: {n: 3, split_at: 30, reset_every: 10}
  bandit: {floor: 0.10, discount: 0.95, info_weight: 1.0, gpu_h_min: 0.1}
  intake: {near_dup_cos: 0.95, tournament_pairs: 6, queue_low_water: 2, queue_max: 6}
  researcher: {max_turns: 30, timeout_min: 20, max_debug: 3, max_concurrent: 2}
  review: {every_confirmations: 3, every_evals: 25, stall_evals: 20, bias_decay_evals: 50,
           fp_every_reviews: 3, fp_stall_closes: 30, fp_min_hours: 12}   # first-principles (6.1); 0 = off
  design: {stall_closes: 0, runs: 16, z: 2.0, step: 1.5}   # 0: only on a review's `design` request
  meta: {researcher_bias: 0, judge_bias: 0, tournament_bias: 0, postmortem_bias: 0, review_bias: 0,
         first_principles_bias: 0, design_bias: 0, crossover_bias: 0, scout_bias: 0, persona_bias: 0}   # 6.3
  novelty: {enabled: true, sweep_redirect: true, weight: 0.5}   # 4.7
  components: {ablation_rate: 0.15, down_rate: 0.5, reprobe_rate: 0.05, min_ablated: 30, seed: 0,
               enabled: {}, rates: {}}   # 6.4; enabled: {name: false} = off for good (operator)
  heartbeat: {poll_s: 30, freeze_min: 20}
  dashboard: {every_ticks: 1}          # <state_dir>/dashboard/index.html (6.2); 0 = off
```

## 9. Agents (`loop/agents/`, prompts in `loop/prompts/*.md`)

- **LLM client.** `FakeLLMClient` (scripted per role) in every test and `--fake-llm`. `SdkLLMClient`:
  roles `researcher`, `review` (and every tool-using agent) go to Claude through the root `llm` OAuth/SDK
  path; `postmortem`, `judge` go to the qwen OpenAI-compatible endpoint (`OVFX_LOOP_QWEN_URL`). JSON answers are schema-validated and re-asked; a dead backend raises
  `BackendUnavailable`. `ToolGate` enforces the `finish` schema and the wall clock.
- **Researcher.** Operators draft / improve / debug (≤ `max_debug` per lineage) / crossover (the adapter
  merges; the agent resolves semantically). Tools read, list, edit, write, bash, smoke; writes honour
  writable minus locked; bash is confined to the worktree (the base clone is readable) and refuses git
  state changes, SLURM, network, substitution and v1 state. `finish` must name a family from
  `cfg.families`. The harness diffs, re-checks the surface, smokes, commits and runs init-compat.
- **Analyst.** `postmortem(node)`: why it failed (mechanism, dose, implementation or noise), lever tags,
  related hypotheses, next probe — from diff, log tail, per-seed paired deltas vs the node's control and
  posteriors. `big_review()`: full ledger table, lever trends per target, hypothesis e-values, island
  frontiers, postmortems and previous reviews -> findings, meta-hypotheses (lever vector, predicted sign,
  falsification, probes) and an explore bias.
- **Briefs.** literature, failure, frontier (literature until a confirmed node exists), probe (a
  meta-hypothesis's claim, lever directions and the review's probe text).
- **Judge.** `relevance(candidate) -> bool`; `compare(a, b) -> ±1/0` for the tournament.

## 10. Adapters (`loop/adapters/`)

- **sim.** In-process simulator: content-hash commits, JSON jobs advanced one step per poll, a hidden
  planted lever truth ("train weaker" helps; an architecture change without `INIT_FIX` is catastrophic),
  seeded noise from the declared axis sds, injectable faults (crash, hang, infra, eval), 3-way merge.
- **nanopath_v2.** Root pinned at `BASE_SHA` in the base clone `spec.repo`. Runs, worktrees and
  branches are namespaced `lp<tag>` (tag = hash of the state dir). Research worktrees on branch
  `loop/<tag>-<name>` (kept, so commits survive reaping); run worktrees detached per node; both symlink
  the base `.venv`. A run is `./submit/train_1gpu.sbatch configs/main.yaml output_dir=… seed=…`
  (stdin=/dev/null, `WANDB_MODE=offline`, `LABLESS_AUTOSUBMIT_FILE=""`, unique `SBATCH_JOB_NAME`);
  receipts before and after sbatch and job-id recovery by name mean resume never double-submits.
  The run's pool sets SBATCH_ACCOUNT/QOS/PARTITION; a preemptible pool adds `SBATCH_REQUEUE=` (empty =
  `--requeue`), so SLURM requeues a preempted job under the same id (PENDING again, then a new start
  time: the freeze clock restarts). A job that ends PREEMPTED / NODE_FAIL, or sits `JobHeld*` (past
  MaxBatchRequeue), is scancelled and relaunched from the receipt as a new job, the receipt following
  its id: a free retry (10 max), never an infra void. The
  smoke is the same launcher on the smoke config (`SBATCH_WAIT`, 20-min limit, scancelled on timeout),
  so the driver needs no GPU. Scores from a `summary.json` written after submit, gpu hours
  from sacct. A started job with no output/log progress for `freeze_min` is scancelled -> infra void;
  finished: no axes -> eval void, FAILED/OOM -> crash (infra on a hardware signal, unless the tail is a
  GPU OOM), else infra. `oom_ladder(commit)` and the mitigated launch/smoke are §5's.
  `file::Qual.name` (AST) or `file::start...end` (source span) entries in `surface.locked` or a family's
  `paths` pin code inside a file; changing one adds the entry to the changed paths (a lock is then
  rejected, a family entry maps to its family), and the file itself is dropped when nothing outside its
  entries changed. nanopath_v2's eval-time code (ViT.forward/_forward_tokens/encode_image/probe_features/
  _suppress, the calibration draw and post-training fit) is family `eval_readout`: `combine` (a readout
  trained during pretraining = eval_readout + one training-side family), `init_check`, and `forbid` +
  `forbid_config: configs/main.yaml:probe` (labels, probe block/datasets/roots/results, benchmarking,
  labless). The two lines charging the calibration draw to the budget stay locked.
- **initcompat.** A CPU script (worktree python) measures child vs parent: construction, finite forward,
  initial loss / feature scale, parameter ratio, checkpoint keys; `evaluate` fails closed.

## 11. Known limits (deliberately not built)

- With 8 calibration seeds (df 7) the frozen σ bound is ~1.6× σ, so early confirmations need large
  effects under alpha 0.05; calibrate more seeds to tighten it.
- Promote seeds rotate per block through `seeds.promote` (`promote_per_node` at a time); a child's
  `parent`-stage control at a promote seed can repeat the seed the parent was promoted on.
- The shared-control correction (3.3) treats siblings' corrected deltas as independent in the tree:
  siblings pooled in one tag/family still average the residual `b - b_hat` a little; and a common
  real effect of all siblings of one control is (partly) read as control luck, by construction.
- Run worktrees are not reaped (cheap, reused by `parent`/`audit` runs). Lock strings must not contain
  glob characters.
- The `forbid` guard is a static text check of added lines: an indirect reference (a computed key or path)
  passes it. It does not bound how many training tiles the post-training fit reads; only the draw that
  is charged to the budget (the locked lines) is counted.
- A designed batch is built synchronously in one tick (one worktree, a commit per point; only the heaviest point is memory-smoked (4.6); a
  crashing point is just a missing row, OLS runs on what finished). A process death mid-build leaves its
  points running without a book entry (they close as design points, unanalysed). With 2 seeds, se is
  from ~16–22 residual df; the residual also carries every unmodelled interaction.
- Self-tune overrides are never undone automatically: a later review must ask for the old value. The
  first-principles evidence reads root runs' `metrics.jsonl` / `summary.json` (nanopath's format; absent
  files just print "none").
- A free retry reuses the run's output dir (train.py wipes it on a fresh launch); its gpu_h is the last
  job's only.
