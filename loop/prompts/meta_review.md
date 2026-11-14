META-REVIEW of an automated ML experiment loop. Every 3 hours the loop measures itself (report below). Your job:
find the ONE or two changes to how the loop searches that the numbers most support, and propose them. Doing
nothing is a valid answer when the numbers do not support a change.

Think like an experimental designer, not a tuner:
- Find the binding constraint first (the line in the report that limits progress most: e.g. tests that cannot
  resolve the effect sizes being proposed, a source that eats GPU-h without fresh-seed gains, a miscalibrated belief,
  ideas that do not engage their own mechanism, intake rejecting most sessions). Name it with its numbers.
- Prefer a change that makes each GPU-hour more informative (better-designed or more decisive experiments, sharper
  briefs, fewer wasted sessions) over a change that just runs more of the same. Brute force is never the answer.
- Each proposal must predict a measurable effect on a report line, so the next report can falsify it.
- Small samples: say so. A rate over fewer than ~10 events is not evidence by itself.

## The report
{report}

## TUNABLE config (the only keys you may change; values must stay inside the bounds)
{tunables}
A config change is accepted only if a held-out replay of the ledger (the last 72 h of fresh-seed outcomes) shows
a gain of at least 2 bootstrap SE with no guard (ranking rho, calibration, coverage, robustness harm) worse.

## Prompts you may patch ({ab})
A prompt change runs as an A/B arm on 30% of research sessions and is adopted only if its fresh-seed credit beats
the control after 10 sessions per arm. Patch = {"find": exact text that occurs once, "replace": new text}.
{prompts}

## Earlier proposals (most recent last)
{history}

## Hard limits
LOCKED (any proposal touching them is refused): {locked}, the ledger, the adapter's eval / probe locks, the test's
alpha and seed counts, the seed pools, the official panel weights, the task spec.

## Output
- diagnosis: 2-5 sentences: the binding constraint and why, citing report numbers.
- proposals: 0-3 items {kind: "config" | "prompt", path, value (config) | patch (prompt), rationale,
  predicted_effect}. path is the TUNABLE key (e.g. "acquire.harm_q") or the prompt file (e.g. "prompts/ideation.md").
