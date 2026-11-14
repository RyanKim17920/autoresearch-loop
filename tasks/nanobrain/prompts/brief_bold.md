BOLD STRUCTURAL brief. Most recent ideas re-parameterised terms the incumbent already has (LR / decay variants,
masking-ratio tweaks, temperatures, weight-decay multipliers) or bolted another small term onto the same objective.
This session is for something the recipe does NOT have yet: ONE structural change from the taxonomy below.
{suggestion}
## The structural taxonomy (pick ONE category for the change you build)
{taxonomy}

OUT OF SCOPE for this lane (rejected at intake as `bold_knob_only`): a knob / weight / temperature / ratio change of
an existing term, a learning-rate / warmup / decay / layer-decay schedule value, switching an existing option on or
off, re-weighting the existing losses. Judged on the diff (only config values or constants of existing code
changed) and by an independent judge. Retuning comes AFTER: a promising bold node gets up to 5 tuning trials of
its own before its mechanism is judged.

MAKE IT TUNABLE (rejected at finish / intake as `untunable_moonshot`): the loop tunes every structural change BEFORE it
is judged (up to 5 trials, steered by your diagnostic: engaged-but-negative tries lower doses first, not-engaged tries
higher ones, reversed flips the direction where the knob is signed; a new backbone / init also gets an lr x layer-decay
grid and a warmup child). So expose the strength of what you build as numeric config keys with the current value as
default (a loss weight, a mask / drop ratio, a temperature, a dropout, a number of tokens). If the diff adds no
numeric config key and is not a backbone / init swap, declare `components` in finish -- [{family, mechanism, share,
target, knobs: [{key, lo, hi, scale}]}] -- naming the existing or new config keys the ladder may move. A structural
node is not closed as negative before >= 3 tuned points exist (unless its first point is catastrophic with the
mechanism engaged), so a first run at an untuned dose is a draft, not a verdict.

## How to arrive at the idea
Start from what the recipe structurally lacks for brain MRI at this fixed one-hour budget, not from the incumbent's
tuning history: what signal in the volumes does the current MAE objective never use (anatomy and tissue contrast, the
MRI sequence a volume comes from (T1w / T2w / FLAIR / DWI), cross-sequence correspondence, left-right symmetry,
voxel resolution and field of view, lesion vs normal structure), which volumes it never sees or sees too often, how
many images the hour processes (throughput), which part of the run is spent on what, what the architecture cannot
represent. The score is the frozen-feature `open` mean of 11 brainmarks-smri tasks (classification, brain-age
regression, segmentation). Read the literature (method details, the
scale it was shown at) and cite it in the claim. Ideas from other fields are welcome. A multi-file implementation
(a new module file, a new data pipeline stage, new config keys) is allowed and expected; you have the moonshot
turn and time budget.

{slate_rule}

## Init and budgets
{init_rule}
- The training budget is ONE HOUR of wall clock on one H100: a structural change reallocates that hour, it never adds
  any. Throughput (more images seen in the hour) is legitimate; a slower step must pay for itself.
- One training run must still finish inside the one-hour training budget (smoke it).
{constraints}

## Prediction
ONE overall predicted distribution for the change you build (no per-axis budget rows, no dose knob): the chosen
slate candidate's p_neg / median / p_gap, and hypothesis.predictions.panel with the size you actually believe. The
loop's usual advice to shrink predictions toward the past (tiny) mean does NOT apply here: an honest, bold
prediction that is falsified is useful evidence.
