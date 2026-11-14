# loop core v3: one decision engine

The v3 core replaces separate gates, lanes, a stored world model and a bandit with one engine: a pure statistical
model over the ledger, one acquisition rule, one sequential test, one HP / stacking engine, ideation, and a
meta-cycle that tunes only what a held-out replay can measure. Section numbers are referenced from docstrings.
The design of record for records, the adapter seam and the world model is [loop/DESIGN.md](loop/DESIGN.md).

## 1. Module map

| module | role |
|---|---|
| `tick.py` | one tick: poll -> derive -> acquire -> launch -> ideate -> meta |
| `runner.py` | the RunBook: pools, slots, submit / poll / retry, OOM ladder, preemptible routing. No decisions. |
| `spec.py` | the config, split into LOCKED keys and TUNABLE keys (the meta-cycle may move only the latter, within bounds) |
| `contracts.py` | Node, Run, SeedBlock, Action, Status, the adapter protocol |
| `core/data.py` | a read-only typed view of the ledger |
| `core/model.py` | variance components, CUPED, the composite, the empirical-Bayes prior, the posterior, max-of-K |
| `core/seeds.py` | seed pools and their enforcement |
| `core/status.py` | pure `derive()` of every node status and the incumbent chain |
| `core/test.py` | the sequential fresh-seed test |
| `core/acquire.py` | one ranking rule over every possible action |
| `core/factorial.py` | the HP / stacking engine |
| `core/ideation.py` | briefs, headroom, plateau review, anti-repeat, credit |
| `core/meta.py` | meta-report, meta-review, config versions, locks |
| `core/backtest.py` | time-ordered replay over the ledger |

## 2. Statistical core (`core/model.py`)

Only DONE runs are observations. Per axis and for the composite, `y[c,s] = mu[c] + b[s] + e[c,s]` with a shared
seed effect (`Var b = rho sigma^2`) and a residual (`Var e = (1 - rho) sigma^2`), fitted by method of moments and
refitted every tick; nothing is hardcoded. A child is compared with its parent's multi-seed mean, with a CUPED
correction by the parent's run on the same seed, so control luck on one seed does not become a "gain". The
screening composite down-weights axes whose between-config signal is small relative to seed noise; incumbent
switches are judged on the official panel. A hierarchical prior (global, carrier, family, lane), each level a
shrunk running mean with an empirically estimated between-cell variance, gives the posterior. A best-of-K sibling
is shrunk by a max-of-K term whose coefficient is calibrated by the meta-cycle. Statuses are derived from the
ledger, never stored. Selection, test and audit seeds come from disjoint pools.

## 3. Acquisition (`core/acquire.py`)

Every possible action (a new seed of a candidate, a test, a wave point, an ideation session, an idle fill) is
ranked by one value-of-information rule: expected improvement in the incumbent per GPU-hour, with harm guards on
protected axes and floors that keep exploration, ideation and testing from starving. Idle GPUs are filled with
the most informative extra seeds.

## 4. The fresh-seed test (`core/test.py`)

A candidate that ranks best is tested against the incumbent on fresh seeds in blocks. The test is a sequential
procedure with a frozen alternative and a variance bound, so looking after every block keeps its error rate;
it can pass, fail, extend (a bounded number of extra seeds) or stop for futility. A pass switches the incumbent.

## 5. One HP / stacking engine (`core/factorial.py`)

Knob changes are not tuned one at a time: factorial waves over a few knobs (two-level, resolution IV when large)
estimate main effects with a per-block intercept, and the significant ones are stacked into one node that takes
the normal screen / test path. Structural nodes get a bounded tuning ladder before they can count as negative.

## 6. Ideation (`core/ideation.py`, `agents/researcher.py`)

Briefs carry computed evidence: per-axis headroom, the decisive effect size, what families have and have not
paid, near-duplicates to avoid, and the task's knowledge file. Sessions are rotated across brief kinds (literature,
frontier, failure analysis, bold structural, probes, alternative bases, fork-and-flush branches) with credit per
GPU-hour. A researcher session edits a worktree, smoke-tests it and finishes with a structured hypothesis.

## 7. The meta-cycle (`core/meta.py`)

Every few hours of loop time a report measures calibration (posterior vs fresh outcome), test error rates and
throughput; an LLM review may propose changes to TUNABLE keys only. A change is accepted only if a held-out
time-ordered replay (`core/backtest.py`) does not get worse, is stored as a config version, and is reverted
automatically if a guard metric goes out of bounds.

## 8. Base epochs, external bar and re-roots

**Base epochs.** A new root starts an epoch: nothing is paired across epochs, the root is re-baselined, and
old candidates may be ported onto the new root (a port is screened with its original's belief).

**Target-aware acquisition.** When an external target exists (a leaderboard leader's score), acquisition values
the gap to that target, and an incumbent is submitted only when it beats the target by the configured margin on
fresh seeds (`labless.leader_median`, `promote_margin`, `promote_p`, `min_test_seeds`).

**Re-roots.** A confirmed win can become the new root (local re-root), after which ports of earlier candidates
re-test on it.
