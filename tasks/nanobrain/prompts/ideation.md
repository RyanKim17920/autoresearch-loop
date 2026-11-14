## Think from first principles (required: `finish` bounces without this record)
Start from the incumbent and work forward, not from a technique you would like to try.
(a) MECHANISTIC MODEL -- why does the parent behave as it does? Use the diagnostics below, the per-axis table in
    the brief, the run metrics and the code: where does signal get lost?
(b) BINDING CONSTRAINT -- which ONE limit, if lifted, would move the panel most at this fixed budget, and why the
    others are not binding.
(c) DERIVED CHANGE -- the intervention that follows from (b). Any assumption of the recipe may be wrong, the init
    and architecture included.
(d) MECHANISM PREDICTION -- what your change moves, besides the score:
    `hypothesis.mechanism_prediction` = {diagnostic, direction, axes, budget}
    - diagnostic: ONE measured quantity of the run diagnostics: {quantities}
    - direction: +1 (up) or -1 (down) vs the parent on the same seed. After the first finished seed the loop checks
      whether it moved (engaged / did not engage / reversed); the result is shown to later sessions.
    - axes: the probe axes you expect to move, e.g. {"cls": 1, "seg": 0} (cls, reg, seg).
    - budget: your size, PER AXIS: [{"axis": "cls", "row": "D-cls", "gain": 0.012}, ...] -- each axis you move, its
      expected level gain (negative for a cost you accept) and the id of the brief row (the bracketed [..] ids:
      deficits, headroom, external / per-dataset gaps, compute, repair candidates) whose headroom it draws on. The
      panel claim is the sum of official weight x gain; state the size you believe (the brief's "Effect size": no floor, it is scored).
(e) WHY EARLIER ATTEMPTS FAILED -- name the nodes that touched this part and what was wrong with them.
Put them in `hypothesis.mechanistic_model`, `hypothesis.bottleneck`, `hypothesis.derivation`,
`hypothesis.mechanism_prediction` and `hypothesis.prior_attempts`.

{diagnostics}
{slate}
## Knobs (optional)
`knobs` = [{key, lo, hi, scale}] names the doses of your change that deserve a designed sweep: scale `lin` / `log`
over a config key that exists after your edit, or `diff` with key = a node id to switch that node's change on / off.
The loop estimates their main effects in a fractional factorial around your node; you write no extra code.
