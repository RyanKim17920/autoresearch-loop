# Loop generalization audit

**Goal:** "make the loop generalistic". The v3 loop should run any training-research task from a
task YAML plus a small adapter, with no code tuned to one project.

**Scope.** Every module under `loop/`: `tick.py`, `core/`, `world/`, `select/`, `agents/`, `adapters/` and the runner /
driver / spec layer. The prompts under `loop/prompts` and the task YAMLs are included too. The audit was four parallel
read-only sweeps, plus a key-by-key comparison of the resolved configs (`loop effective-config`).

**Classes:**
- **(a)** move into the adapter interface;
- **(b)** move into the task YAML with a generic default;
- **(c)** a generic improvement to turn ON for every task;
- **(d)** fine as is.

**Status column:** FIXED = changed on this branch; PINNED = the task's value now lives in its YAML; OPEN = a
recommendation, not done.

## Summary

| class | found | fixed / pinned | open |
|---|---|---|---|
| (a) adapter interface | 17 | 6 | 11 |
| (b) YAML + generic default | 38 | 27 | 11 |
| (c) generic, ON by default | 17 | 11 (incl. 3 already unconditional) | 6 |
| (d) fine | 14 | – | – |

**Top couplings, by blast radius:**
1. **The project registry defaults are nanopath's.** These are the axes `cls…rob`, the probe keys, guard `rob`,
   `harm_h` and `axis_exclude surv`. A task without a `loop.project` block silently ran on nanopath's axes. `fake_loop`
   did, for every test. **FIXED:** a project is derived from `slots`.
2. **Absolute score-unit constants at nanopath's scale.** These are `outlier_sd_d` .0034, `parent_tol` .004,
   `stack_intercept` .0013, `local_reroot_min_gain` .002, `sibling_gain` .0081, `promote_margin` .004, prior `tau0` /
   floor / max, `resume_tol`, the eval-only floors and so on. nanobrain hand-rescaled some of them; for the rest it
   silently inherited nanopath's scale. **FIXED:** they are now k × the measured `panel_sd` (`SIGMA_SCALED`), and both
   live tasks pin their values.
3. **`NanopathV2Adapter` is the de-facto base class.** nanobrain inherits about 30 methods unchanged. The nanopath
   behaviour still inherited is: `latest.pt` / `train.resume` resume arming in `poll`, `interrupt`, `postprocess`, the
   eval-only shim (`model.ViT`), `PROBE_FUNCS`, `HF_HUB_OFFLINE`, and the `WANDB_MODE` / `LABLESS_AUTOSUBMIT_FILE`
   launch env. Each one is inert for nanobrain only because a feature flag is off there. **OPEN (a):** split a generic
   git+SLURM base out of it (see §Adapters).
4. **The world-model cell prior used (1.4826 MAD)² − mean V.** This is the defect `core.bar` fixed for its EB priors
   . **FIXED:** `fit_prior` now uses the robust Paule–Mandel τ² (`model.tau2_robust`), behind
   `loop.project.cell_tau2`. It is ON for nanobrain as a bug fix; nanopath pins the old estimator.
5. **Generic features were OFF by default.** Local re-root, re-root ports and futility early-stop were ON for nanobrain
   only. **FIXED:** they are ON by default.
6. **Domain text.** The notes-tool regime ("DINOv2 ViT pathology … TCGA tiles") reached nanobrain's research
   sessions. "Outside pathology" appeared in the far-reading instruction and the scout persona, and the intake
   init-illegal message was pathology-specific. **FIXED.** `plateau.md`, `meta_review.md`, the `bold` category lexicon
   and the `novelty` / `legs` mechanism regexes are still nanopath-worded: **OPEN (b)**.
7. **`selfcheck` imported `nanobrain.model.ViTMAE3D`** and set `BRAINMARKS_SMRI_DATA` in generic code. **FIXED:**
   these are now the adapter hooks `SELFCHECK_MODEL`, `SELFCHECK_ENV` and `SELFCHECK_OVERRIDES`.

## Findings

### Statistics / bar / noise (loop/core, loop/world)

| where | what | class | status |
|---|---|---|---|
| core/bar.py `tau2_robust` | robust Paule–Mandel τ² for the paired / μ EB priors | (c) | already unconditional |
| core/model.py `fit_prior` | cell τ² = (1.4826 MAD)² − mean V; it floors on heteroscedastic cells (many 1–2-seed nodes) | (c) | FIXED: `tau2_robust` (`loop.project.cell_tau2: robust`, the default for every declared or derived project); nanopath_v2 PINS `mad_minus_mean` (the operator decides); with no project installed the old estimator is used (parity) |
| core/model.py `fit_prior` | root fallback τ² 1e-4 when there are no points | (d) | – |
| spec PriorCfg tau0 .005 / tau_floor .0005 / tau_max .05 (world/shrinkage DL) | score scale | (b) | FIXED: SIGMA_SCALED 1.5 / .15 / 15 × panel_sd; PINNED in both live tasks |
| world/shrinkage.py DerSimonian–Laird per family | noisy with 1–2-member families | (c) | OPEN: use `tau2_robust` |
| world/predictive.py:195 initial group τ² .003², TAU_MAX2 .05² | score scale | (b) | OPEN |
| core/status.py OUTLIER_SD_D .0034 / spec `status.outlier_sd_d` | score scale (a paired sd) | (b) | FIXED: 1.0 × panel_sd; PINNED (np .0034, nb .0063) |
| core/status.py `local_reroot` min_gain .002 / spec `test.local_reroot_min_gain` | score scale | (b) | FIXED: 0.6 × panel_sd; PINNED (np .002, nb .002) |
| core/acquire.py STACK_INTERCEPT .0013 / `acquire.stack_intercept` | score scale | (b) | FIXED: 0.4 × panel_sd; PINNED (np .0013, nb .0024) |
| core/acquire.py FRESH_SHRINK .55 | nanopath's empirical fresh/selected ratio | (b) | OPEN: estimate online |
| core/pairs.py `parent_tol` .004 / `axis_leg_tol` .002 | score scale | (b) | FIXED: 1.2 / 0.6 × panel_sd; PINNED |
| core/pairs.py PLAN_EPS .0005 | log threshold only | (d) | – |
| spec LablessCfg `promote_margin` .004, `sibling_gain` .0081 | score scale; nanobrain inherited .0081 | (b) | FIXED: 1.2 / 2.5 × panel_sd; PINNED (nb `sibling_gain` .0081, unused while exploratory is off) |
| spec EvalOnlyCfg `sigma_floor` .0005 / `null_tol` .0001 | score scale | (b) | FIXED: SIGMA_SCALED; PINNED in nanopath_v2 (nanobrain: eval-only is off, so the scaled value is inert) |
| spec BackfillCfg `min_gain` .0003 | the backfill section is dead in v3 (test_batch_c_yaml) | (d) | – |
| spec `slurm.resume_tol` .001 | score scale | (b) | FIXED: 0.3 × panel_sd; PINNED (np .001, nb .0019) |
| spec `alloc.prune_stale_family.max_best_delta` .004 | score scale (only when the block is declared) | (b) | FIXED: 1.2 × panel_sd when absent |
| spec FamilyPrior architecture mean −.02 / sd .005, `kind_weight_floor` .08, `structural_gain` .008, `tail_gain` .004, `unit_noise_sd` .0025, `family_block_mean` −.004 | score scale | (b) | OPEN (lower traffic) |
| diagnostics.py:53 PANEL_SD_FALLBACK .0033 | score scale | (b) | OPEN: use `cfg.panel_sd` |
| core/bar.py K_MEDIAN = 3 | Labless's maintainer rule (median of 3 fresh seeds) | (b) | OPEN: `labless.k_median` |
| project `noise_floor_seeds` (0 = off; nb 5) | a root with fewer than k seeds uses max(fit, measured sd) | (c) | FIXED: default 5 for every task with a project block |
| loop calibrate | panel / axis sd measured from root replicates (+ `noise_prior`) | (d) | already data-driven |
| test partial placement (6286fc3), preempt-backoff cap (974a4df), research spend breaker (`researcher.breaker_*`) | generic | (c) | already unconditional |
| core/test.py ALPHA / K / K_MAX / HARM_Z / CATA_SIGMAS | dimensionless | (d) | – |

### Registry / spec defaults (loop/project.py, loop/spec.py)

| where | what | class | status |
|---|---|---|---|
| project.py defaults (`_AXES`, `_PROBE_KEYS`, guard rob, `harm_h` rob .008, `axis_exclude` surv) | a spec without a project block got nanopath's registry | (b) | FIXED: `spec.derived_project` (axes = `slots.vector`, no guard); kept only when the axes ARE nanopath's keys |
| spec `test.local_reroot` False | local-incumbent tier | (c) | FIXED: default ON |
| spec `test.reroot_port` False | re-root ports | (c) | FIXED: default ON |
| spec `acquire.futility_p` 0 | futility early stop | (c) | FIXED: default .02 (nanobrain's measured choice) |
| spec `ideation.pairwise_center` False | a bias fix for the pairwise prior | (c) | FIXED: default ON |
| spec `acquire.control_bar`, `gap_cap_sigmas` | shared-control offset bar, gap cap | (c) | OPEN: recommended (nb ON) |
| spec `ideation.require_evidence`, `bank_v3`, `scout.v3`, `root_audit.enabled`, `intake.noop_fingerprint` (needs an adapter) | research features (nb ON) | (c) | OPEN: LLM-spend policy, operator call |
| spec `holdout.enabled` True | the deterministic pick reads nanopath `final_probe_*` / `pathorob_*` keys | (b) | FIXED: default OFF; PINNED ON in nanopath_v2 |
| spec `diagnostics.enabled` True, `ideation.diagnose_precompute` True | `latest.pt` + `cfg["dino"]` extractor; nanopath feature_diag job | (b) | FIXED: default OFF; PINNED ON in nanopath_v2 |
| spec `slurm.smoke_probe` "readout" | drives nanopath's `probe.py` | (b) | FIXED: default off; PINNED readout in nanopath_v2 |
| spec `ideation.axis_tasks` nanopath `probe.*` keys | – | (b) | FIXED: default {}; PINNED in nanopath_v2 |
| spec DatabankCfg root / python / smoke_dataset = /data/nanopath-*, a nanopath venv | – | (b) | FIXED: default ""; python '' = the project python; root required when enabled; PINNED in nanopath_v2 |
| spec `local_runs.scorer` = `loop.adapters.nanobrain:task_score` | a nanobrain symbol as the generic default | (a) | FIXED: '' = `<execution adapter module>:task_score`, checked at spec load when local_runs is enabled |
| agents/diagnose.py databank tool | listed /data/nanopath-databank for every task (cwd once root = '') | (b) | FIXED: "no data bank configured" unless the bank is enabled and has a root |
| core/data.load_checked (backtest / seeds CLIs) | installed the project but not its noise floor | (b) | FIXED: `spec.install_registry`, shared with load_context |
| calibrate --dry-run provisional sigmas | the score-unit defaults kept the literals | (b) | FIXED: `sigma_scaled` on the provisional panel_sd |
| spec `ideation.external_task_*`, `task_metric_key` (THUNDER) | used only with `external_task_csv` | (b) | OPEN (inert without a csv) |
| spec SeedPlan literals, `fresh_base` 100000 | policy | (d) | – |
| spec TUNABLES `harm_h_rob` | names nanopath's axis | (b) | OPEN (hidden without a guard) |
| `disk_floor_gb`, `smoke_primary_hold`, smoke wait bounds | SLURM courtesy policy | (b) | OPEN |

### Adapters

| where | what | class | status |
|---|---|---|---|
| contracts.ExecutionAdapter | 13 methods; about 45 more duck-typed and undocumented | (a) | FIXED: `ADAPTER_REQUIRED` (+`config_rel`), `ADAPTER_OPTIONAL` catalogue (absent → what the loop does), `adapter_conformance()`; a test keeps the catalogue complete |
| nanopath_v2 `poll` / `_arm` / `_launch` / `interrupt` | `latest.pt`, `train.resume`, `train.save_every` resume protocol | (a) | OPEN: nanobrain overrides `resume_state` (f29e5b9) and runs `slurm.resume: off`; `interrupt` would raise for nanobrain |
| nanopath_v2 `postprocess`, diagnostics.py, diag_ckpt.py | `latest.pt`, `cfg["dino"]` | (a) | OPEN (diagnostics now default off) |
| nanopath_v2 `eval_only` / SHIM / EO_* / PROBE_FUNCS | nanopath `model.ViT` readout lane | (a) | OPEN (evalonly is off by default) |
| nanopath_v2 `_sbatch` env `WANDB_MODE=offline`, `LABLESS_AUTOSUBMIT_FILE`; OFFLINE_ENV | applied to every subclass | (b) | OPEN: `launch_env` |
| nanopath_v2 BASE_SHA / RUNS_ROOT / WORKTREES_ROOT / CONFIG_REL / LAUNCHER / SMOKE_OVERRIDES literals | fallbacks behind `loop.project` | (b) | OPEN: empty defaults in a generic base |
| evalonly.py (×8), prune.py PATTERNS | `latest.pt` hardcoded | (a) | OPEN: adapter checkpoint name. The obvious fix (match the adapter's checkpoint name in `tick.crash_class`) was tried and REVERTED: nanobrain's normal log prints `saving checkpoint …/checkpoint-last.pth`, so every crash would have become an infra `crash:checkpoint`. A bare file name is not a crash signature |
| tick.py:221 `latest\.pt` in the `crash:checkpoint` regex | the same risk for nanopath if its train log names `latest.pt` | (a) | OPEN: verify |
| tick.py:195–217 `probe.py` + Coxnet / ConvergenceWarning ill-conditioned class and repair briefs | nanopath's survival probe | (b) | OPEN (inert without probe.py in the tail) |
| tick.py `getattr(adapter, "config_rel", "config.yaml")` | – | (a) | FIXED: `config_rel` is REQUIRED |
| selfcheck.py `from nanobrain.model import ViTMAE3D`, `BRAINMARKS_SMRI_DATA`, nanobrain dotlist overrides | nanobrain code in a generic module | (a) | FIXED: `SELFCHECK_MODEL` / `SELFCHECK_ENV` / `SELFCHECK_OVERRIDES` (nanobrain sets them; byte-identical probe) |
| core/program.py:831 imports `loop.adapters.nanopath_v2._span` | an AST span util in the wrong module | (d) | works for any Python source |
| core/holdout.py | entirely nanopath summary keys | (a) | OPEN (default off now) |
| localruns.py `summary.json`, `eval/aggregate.json`, `checkpoint-last.pth`, `nb_*` keys | nanobrain run layout | (a) | OPEN (opt-in feature) |
| labless.py default submit argv (nanopath `submit_to_labless.py`) | – | (b) | OPEN (nanobrain uses `submit_cmd`) |
| knowledge.py:100, labless.py:451, tested.py:208 | strip `nb_` / `final_probe_` prefixes | (a) | OPEN: `P.short_axis` (display only) |

### Domain text (prompts)

Every `P.rule()` default is nanopath text: about 60 keys, of which nanobrain sets about 40. A new task without rules
and overlays briefs its researchers as pathology SSL. The template now lists the key rules and the prompts to
overlay. **OPEN (b):** make selfcheck warn when a project falls back to a nanopath rule default.

| where | what | class | status |
|---|---|---|---|
| agents/notes.py REGIME (DINOv2 pathology, TCGA tiles) at import time | reached nanobrain sessions | (b) | FIXED: `domain.rules.regime` at call time (`notes.tool_defs()`); nanobrain rule added |
| agents/notes.py "OUTSIDE pathology / SSL" | – | (b) | FIXED: `rule('analogy_field')` (nb: brain-MRI / MAE) |
| meta.py scout persona "outside pathology" | – | (b) | FIXED: "outside the task's own field" |
| select/intake.py init_illegal / allowlist messages "pathology-trained (or GenBio)" | – | (b) | FIXED: generic (`init_compat.deny` / `allow`) |
| prompts/plateau.md (classification / robustness / segmentation), meta_review.md | not overlaid by nanobrain | (b) | OPEN |
| prompts/ideation.md, first_principles.md (nanopath axes, Labless) | nanobrain overlays both | (b) | OPEN: template `{axes}` |
| core/bold.py CATEGORY lexicon / descriptions (WSI, stain, jepa, retile) | regexes are rule-overridable, descriptions are not | (b) | OPEN |
| novelty.py `_HINTS`, legs.py `_MECH`, design.py MEMORY_KEY, databank.py | nanopath mechanism vocabulary | (b) | OPEN |
| select/intake.py DISALLOWED_DATA `genbio` | fires only on genbio | (d) | – |
| core/bold.py JUDGE_SYSTEM, briefs.* defaults | already `P.rule`-routed; nanobrain overrides | (d) | – |
| researcher.py `_DENY` (labless / overfitX / nanopath-old / /data) | – | (b) | OPEN |

### Fine as is (d)

`tick.py` has no project branches, no `isinstance(adapter)` checks, no SLURM or seed literals, and no score
constants. SLURM accounts and pools come from the YAML. There is no `if name ==` anywhere in `loop/`. The research
breaker is generic. calibrate, scout, root_audit and the breaker contain no task strings. The eval-only and evalonly
harm_tol axis names are keyed by the YAML.

## Effective resolved config: before → after

Generated with `python -m loop effective-config --task T --diff docs/effective_config/T.before.json`. The before
snapshot is taken at d8e68c4. The after dumps are `docs/effective_config/T.after.json`.

### nanobrain (`tasks/nanobrain.yaml`): unchanged, except unused strings and bug-fix text

```
cfg.databank.python: "/home/user/nanopath/base/.venv/bin/python" -> ""   (data bank disabled: unused)
cfg.databank.root: "/data/nanopath-databank" -> ""                                  (unused)
cfg.databank.smoke_dataset: "/data/nanopath_arrow" -> ""                            (unused)
cfg.local_runs.scorer: "loop.adapters.nanobrain:task_score" -> ""                   ('' resolves to the SAME function)
cfg.evalonly.sigma_floor: 0.0005 -> 0.00104, null_tol: 0.0001 -> 0.00021            (evalonly.enabled false: inert)
cfg.project.cell_tau2: <absent> -> "robust"                                         (BUG FIX: the robust cell tau2)
cfg.domain.rules.regime: <absent> -> "a 3D ViT-B masked autoencoder on FOMO300K ..." (BUG FIX: sessions were told
                                                                                      the pathology regime)
```

nanobrain's run-time behaviour changes in two ways that are not config:
- The research-notes far-reading text now says "outside brain-MRI / MAE".
- The world-model cell prior uses robust τ² (`cfg.project.cell_tau2: robust`, the bug fix above).

A shadow tick on a copy of the live nanobrain ledger gives the same incumbent, statuses, actions and 22 would-submit
runs before and after. The only differences are the screen values (.009993 → .010084), from the τ² fix.

Every generic default that was turned on was already ON for nanobrain. Every scaled constant is pinned at the value
nanobrain ran on. Pinned at the OLD (nanopath-scale) values, flagged for an operator decision: `prior.tau0` / `tau_floor`
/ `tau_max` and `labless.sibling_gain`. The generic scale would be `tau0` .0104 and `sibling_gain` .0173.

### nanopath (`tasks/nanopath_v2.yaml`, the live nanopath loop task)

```
cfg.acquire.futility_p: 0.0 -> 0.02              (c) futility early stop ON
cfg.ideation.pairwise_center: false -> true      (c) bias fix ON
cfg.project.noise_floor_seeds: 0 -> 5            (c) root noise floor (a no-op while the root has >= 5 seeds)
cfg.test.local_reroot: false -> true             (c) local-incumbent tier ON
cfg.test.reroot_port: false -> true              (c) re-root ports ON
cfg.local_runs.scorer: "loop.adapters.nanobrain:task_score" -> ""   (feature off; '' = the adapter module's task_score)
registry.noise_floor.*: {} -> per-axis sigma floors (panel .00230, cls .00322, ...)
```

Every other nanopath value is unchanged. The nanopath-only defaults that became generic are pinned in its YAML:
- holdout ON, diagnose_precompute ON, `smoke_probe: readout`, axis_tasks;
- the databank python / smoke_dataset;
- `outlier_sd_d`, `parent_tol`, `axis_leg_tol`, `stack_intercept`, `local_reroot_min_gain`;
- the prior τ's, `resume_tol`, and the eval-only floors.

The cell-prior estimator is pinned too: `project.cell_tau2: mad_minus_mean`. A shadow tick on a copy of the live
nanopath ledger is identical before and after (that loop is idle: no actions).

`tasks/nanopath.yaml` (and `fake.yaml`, `localcv.yaml`) are LEGACY-SDK specs: they have no `loop:` block and the v3
loop never reads them. They only need to load. The nanopath v3 task is `nanopath_v2.yaml`.

### fake_loop (the test task)

The core tests replay a nanopath-keyed fixture ledger under this spec. It therefore declares `project: null`
EXPLICITLY, which keeps the built-in registry; an ABSENT block now derives the axes. It also pins the old score-unit
literals and run diagnostics. Its diff is the (c) flips plus holdout off.

`tests/loop/test_task_conformance.py` covers the generic paths:
- derived axes;
- sigma scaling;
- the lifecycle on nanobrain, nanopath_v2 and the template.
