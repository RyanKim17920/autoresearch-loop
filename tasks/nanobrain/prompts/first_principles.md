FIRST-PRINCIPLES REVIEW of task `{task_id}` ({n_scored} scored trials; {why}).

Step back from individual ideas. The loop runs on a stack of ASSUMPTIONS -- about how it measures,
what limits the model, how it searches, what it may not change, and how it spends its GPUs. Some of
them are wrong, and a wrong one wastes every experiment built on it. Your job: list them, test each
against the evidence below, and turn the broken or unknown ones into action.

## 1. List the assumptions, in five lenses
- measurement: is the panel/axis noise what the loop thinks (per-axis and per-dataset spread below)?
  does a 2-seed screen resolve the effects we look for? is the proxy (the probe panel) valid for the
  real target? are shared controls and the corrected estimates trustworthy?
- ml_problem: what actually limits the model at THIS budget? Read the run metrics: the budget is the
  one-hour wall clock, so how many steps / images did the run reach (throughput), and what fraction of each
  schedule is reached at the stop -- a schedule keyed to a step count the hour never reaches does not anneal, so
  its LR / WD end values are never the ones it was designed to reach (LR annealing is a known leaderboard lever). What are the final lr/wd/ema?
- search: are the arms, briefs, families and knob sizes able to find a gain the size of the MDE?
  which families are exhausted, which never really tried?
- constraints: separate HARD external rules (locked files, protected keys, the training budget, eval
  identity, the seed cap, the Labless gate) from SELF-IMPOSED ones (intake rules, family boundaries,
  researcher habits, thresholds). Only self-imposed ones can be proposed away.
- loop_system: throughput, slot use, void/crash waste, time from idea to verdict, queue starvation.

## 2. Test each: holds | broken | unknown, with the evidence (cite numbers that appear below)

## 3. Output
- summary: the few assumptions that matter most and what follows.
- assumptions: [{lens, assumption, status, evidence}] -- every lens, evidence cited.
- meta_hypotheses: cross-idea hypotheses, each with a lever vector over {lever_vocab}, predicted
  sign/size, a falsification criterion (against the measure parent / incumbent, never root) and 1-3
  probes (same schema as a big review).
- moonshots: 0-3 bold structural probe briefs (a new mechanism, not a knob value).
- assumption_probes: 0-3 probes that TEST an assumption directly rather than chase score, e.g.
  proxy validity, or a trajectory-perturbing A/A control (a change that should be neutral -- such
  as a different data-order offset -- run through the full pipeline to measure the loop's false
  positive rate). Each must be ONE implementable change within the hard constraints.
- selftune: optional [{knob, value, reason}] -- operational knobs the loop applies ITSELF, clamped to
  these bounds (values in force now shown):
{selftune}
- proposals: [{change, rationale}] for the OPERATOR -- anything outside the self-tune whitelist
  (budget, locks, eval, seed cap, alphas, the Labless gate, init-compat, SLURM, new code in the loop).
  A self-imposed rule you found broken belongs here. Say what evidence would settle it.

## Hard constraints (a probe that violates one is dropped before it reaches a researcher)
{constraints}

## Throughput and slots (code-computed)
{throughput}

## Noise: per-axis (declared vs current) and per-dataset spread across root seeds (code-computed)
{variance}

## Run metrics of the latest finished root run (summary + first/last training record)
{run_metrics}

## Meta-learning scorecard (code-computed: every LLM component's decisions vs what was later measured)
Each component's prompt carries its own record and bias knob. Move a knob through `selftune`
(meta.<component>_bias; -1..1: + = MORE of what it says, - = LESS)
when its record shows a systematic miss; say which line of the record justifies it.
{meta}

## Your own track record as reviewer (code-computed) -- calibrate on it
{record}

## Search quality (code-computed)
{quality_table}

## What we've learned so far (code-computed)
{lessons}

## Lever trend model (panel)
{levers}

## Hypotheses with e-values
(a meta marked STALE rests on nodes that have since fallen: it steers nothing; re-derive or drop it)
{hypotheses}

## Why recent nodes closed (computed: forecast vs measured, gate posterior, run diagnostics)
{postmortems}

## Previous reviews
{previous}

## Trials (corrected shrunk estimates; primary evidence)
{table}

Families: {families}
