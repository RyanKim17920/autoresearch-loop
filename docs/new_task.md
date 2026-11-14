# Add a task to the loop in 7 steps

The v3 loop (`python -m loop run`) is driven by a task YAML and an adapter. Everything else is generic.

1. **Copy the template.** Copy `tasks/_template.yaml` to `tasks/<name>.yaml`. Fill in `repo`, `slots`, `surface` and
   `loop.project`. The axes in `loop.project.axes` must name exactly the keys in `slots.vector`. A YAML without a
   `project:` block gets one derived from `slots`, with no guard axis.

2. **Write the adapter.** Implement `loop.contracts.ExecutionAdapter` (13 methods) plus a `config_rel` (or
   `CONFIG_REL`) attribute. Check it with `loop.contracts.adapter_conformance(MyAdapter) == []`.
   - The quickest path for a git + SLURM training repo is to subclass `loop.adapters.nanopath_v2.NanopathV2Adapter`.
     `loop.adapters.nanobrain.NanobrainAdapter` is the worked example.
   - Override at least these class constants: `CONFIG_REL`, `LAUNCHER`, `SMOKE_OVERRIDES`, `MEMORY_KEEP`, `FLOPS_RE`.
   - Override these methods: `_read_scores` (a run dir to `{axis key: value}`), `smoke_config`, `resume_state`
     (with `interrupt`, or set `loop.slurm.resume: off`), and `config_violations` / `lock_violations` if the defaults
     do not fit.
   - The base class's nanopath behaviour that you will otherwise INHERIT is listed in
     `docs/generalization_audit.md` §Adapters.
   - Each optional capability (eval-only lane, grafts, pool capacity, the selfcheck model build, and so on) is listed
     in `loop.contracts.ADAPTER_OPTIONAL`, together with what the loop does without it.

3. **Set the domain text.** Under `loop.domain.rules`, set `analogy_field` and `regime`. For any prompt that names
   another domain, add an overlay `<prompt>.md` in `loop.domain.prompt_dir`.
   - The prompts in `loop/prompts/*.md` are written for nanopath, especially `ideation.md`, `first_principles.md`
     and `plateau.md`.
   - `tasks/nanobrain/prompts/` is the worked example.

4. **Set families, protected keys and pools.** Fill in `loop.families` (config key globs and paths for each mechanism
   family, with no overlaps), `surface.protected_keys` (budget, eval identity and data identity) and
   `loop.slurm.pools`.

5. **Check the resolved config.** Run `uv run python -m loop effective-config --task tasks/<name>.yaml`.
   - Read what the generic defaults resolve to. In particular, the `SIGMA_SCALED` thresholds become k x `panel_sd`
     once `panel_sd` is measured.
   - Pin a value in the YAML only when you have a task-specific reason.

6. **Run the conformance test.** Run `uv run pytest tests/loop/test_task_conformance.py`. It runs every
   `tasks/*.yaml`: the YAML loads and resolves, its adapter conforms, and a dry tick on the offline executor goes
   through smoke, screen, test and re-root on the task's own config.

7. **Calibrate, check, then run.**
   - `uv run python -m loop calibrate --task tasks/<name>.yaml` measures the noise from root replicates and writes the
     sigmas into the YAML.
   - `uv run python -m loop selfcheck --task tasks/<name>.yaml --state-dir <tmp>` runs the LLM-free preflight.
   - `uv run python -m loop run --task tasks/<name>.yaml` starts the loop. Watch it until the first smoke passes.
