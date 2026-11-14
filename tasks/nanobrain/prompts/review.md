BIG REVIEW #{n_review} of task `{task_id}` (trigger: {trigger}; {n_scored} scored trials).

You are the principal investigator. Do not summarise ideas one by one: connect ACROSS
them. Find the shared cause behind many separate results and state it as a meta-hypothesis
with a testable prediction.

Example: "six unrelated regularisers all hurt, and every change that lowered the LR helped
a little -> the model is over-trained for this budget -> TRAIN WEAKER: predict that lower LR,
lower weight decay, shorter warmup or weaker augmentation each raise the panel score."
Each such prediction is a lever direction over {lever_vocab}. Also look for dose effects
(mild helped, strong hurt), per-axis trade-offs, and exhausted versus under-sampled families.

Evidence: the CORRECTED world model is the only source of effect sizes. For a trial, cite its
shrunk utility U (mean±sd), P(U>0) and status from the trial table; a trial with P(U>0) near
0.5 or below has NO demonstrated effect, whatever its raw number says. The raw panel column is
raw, uncorrected, single-seed — do not cite it as an effect (shared controls and seed noise move
it by more than most real effects). For a lever, cite the LEVER TREND tables (effect per +1
step, with se; |mean| < 2 se is only "suggestive"); the computed close explanations give mechanism, not size. Every
number you cite must appear below. Hypothesis E values are bounded likelihood ratios against the
claim's PREDICTED size: E_for tests "effect > 0", E_against tests "effect >= the predicted size",
and one trial moves a meta-hypothesis's E by at most 4x (a trial hypothesis's by 20x). A meta
verdict needs E >= K/alpha, K = the number of meta-hypotheses (one bar for the whole family), so it
takes several concordant trials; most metas stay "testing". A meta's trials are the nodes made from
its own probe briefs, then the nodes whose changed config keys / code identifiers its claim and
probes NAME, and only when neither exists, lever-matched nodes: name the exact keys in your probes.
REFUTED means "smaller than predicted", not "harmful".
Trials sharing a baseline run are positively dependent (corrected for, but do not over-read E);
confirmation e-values use fresh seeds and are exact.

## Hard constraints (a probe that violates one is dropped before it reaches a researcher)
{constraints}

## Output
- summary: the state of the search in a few sentences (what works, what is dead, where
  the uncertainty is).
- findings: short, specific, evidence-citing statements (cite node / hypothesis ids).
- meta_hypotheses: 1-4 cross-idea hypotheses. Each has a claim, a mechanism_tag, a
  family (or "cross"), a lever direction vector over {lever_vocab} (non-zero entries
  only, -1/+1), predicted panel sign (+1/-1) and size, a falsification criterion (phrased against
  the node's MEASURE PARENT or the incumbent -- its OWN increment -- never against root: a child of
  the incumbent inherits the incumbent's lead), and
  1-3 concrete probe experiments (one change each, implementable within the editable surface
  and the fixed training budget above) that would test it.
- MOONSHOTS: if single-knob probes keep coming back null (e.g. a task-conflict ceiling where the
  levers that help one axis hurt another and cancel on the panel), ALSO propose 1-2 bold,
  structural probes, each starting with the tag `[moonshot]` (in `probe_briefs` or a meta's
  `probes`): a new mechanism, not a knob value -- e.g. a different masking strategy or ratio, a second
  objective (e.g. cross-sequence prediction), an auxiliary head, a curriculum over resolution, balanced sampling
  over MRI sequences, throughput changes that fit more images into the same hour, an optimizer swap (Muon),
  3D RoPE / native resolution. A moonshot runs with a larger researcher budget and may span
  families, but the hard constraints above still apply and violating briefs are still dropped.
- design: optional, and ONLY for MAJOR knobs: small hyperparameter sweeps belong in a designed
  batch of major knobs (the lever regression or a stated mechanism backs each, e.g. LR schedule
  truncation), never as one-tweak siblings. When such knobs are too noisy to separate one at a time,
  ask for a DESIGNED BATCH: 4-6 config knobs as
  `{"key": dotted config key, "low": value (default: root's), "high": value, "levers": {lever: -1/+1}}`.
  The loop runs a 16/32-point resolution-IV fractional factorial of them on one shared seed block,
  estimates every knob's main effect (± se) and runs the stack of the significantly better levels
  through the normal promote/confirm path. Results appear under "What we've learned".
- selftune: optional `[{"knob", "value", "reason"}]` -- operational knobs the loop changes ITSELF
  (clamped to these bounds; values in force now shown). Anything else is the operator's call.
{selftune}
- explore_bias: optional family -> small number; positive = explore this family more.
  It is clipped to +-{bias_cap} by code and decays; it is a nudge, not a decision.

## Meta-learning scorecard (code-computed: every LLM component's decisions vs what was later measured)
Each component's prompt carries its own record and bias knob. Move a knob through `selftune`
(meta.<component>_bias; -1..1: + = MORE of what it says, - = LESS)
when its record shows a systematic miss; say which line of the record justifies it.
{meta}

## Your own track record as reviewer (code-computed) -- calibrate on it
{record}

## Search quality (code-computed: what executes, what hits, where the loop wastes work)
{quality_table}

## What we've learned so far (code-computed; the researchers see this too)
{lessons}

## Lever trend model (panel)
{levers}

## Lever trends per axis
{axis_levers}

## Hypotheses with e-values
A meta marked STALE rests on nodes that have since fallen (not confirmed / screened out / own increment
under 1 sigma): it no longer steers anything; re-derive it from the current table or drop it.
{hypotheses}

## Island frontiers
{frontiers}

## Why recent nodes closed (computed: forecast vs measured, gate posterior, run diagnostics)
{postmortems}

## Previous reviews
{previous}

## Trials (corrected shrunk estimates; primary evidence)
{table}

Families: {families}
