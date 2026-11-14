PLATEAU REVIEW of task `{task_id}`: {plateau}. Incumbent {incumbent}.

The search has stopped improving. Diagnose the system, not one idea: what limits the incumbent, and why the recent
experiments could not find it. Use only the numbers below; say when they cannot decide a question.

## Per-axis headroom
{headroom}

## Incumbent diagnostics (bottleneck flags from its runs)
{diagnostics}

## Lane yields (posterior of each node's own increment over its parent)
{lanes}
(Older crossovers were measured against the parents' common ancestor, so their own gain includes a parent's lead.)

## Fresh-seed test credit per source (lane / brief / model / prompt; test-block gains only)
{credit}

## Clusters of ideas with a negative measured effect (largest first)
{clusters}

## Mechanism engagement (did the declared diagnostic move after the first seed)
{engagement}

## Computed facts
{facts}
Read the axis trades before proposing: a mechanism that bought classification by costing robustness / segmentation is
a trade, not a free gain. A crash-censored region was never measured (every run crashed or ran out of memory): it is
open, not refuted. The coverage lists what the carrier's lineage already tried; a dose question over keys it has not
covered can go in `knobs` (they become a factorial wave on the incumbent).

## Output
- diagnosis: the few system-level reasons for the plateau, each citing numbers above.
- hypotheses: 1-4, each {claim, evidence, decisive_experiment, prediction, knobs}:
  - decisive_experiment: ONE change whose outcome settles the claim either way; expected composite size at least
    {mde} (smaller is not resolvable by a 5-seed test) or a stated diagnostic effect.
  - prediction: {diagnostic, direction (+1 / -1), axes {axis: -1 / 0 / 1}, size}; diagnostic is one of: {quantities}
  - knobs (optional): [{key, lo, hi, scale: lin | log | diff}] when the question is a dose, not a mechanism: config
    keys (or node ids with scale diff) whose main effects a fractional factorial should estimate.
