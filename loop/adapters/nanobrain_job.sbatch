#!/usr/bin/env bash
#
# The loop's nanobrain job (loop.adapters.nanobrain): pretrain, then the brainmarks-smri "open" eval, in ONE job.
#
#   sbatch nanobrain_job.sbatch CFG OUT SEED AUTO_RESUME REPO [MODE]
#
#   CFG          config merged over the worktree's default_pretrain.yaml (nanobrain.train --config)
#   OUT          the loop's per-run output dir (train writes OUT/, the eval OUT/eval/aggregate.json)
#   SEED         seed override
#   AUTO_RESUME  true | false (false for TEST runs: a requeued TEST run restarts from step 0)
#   REPO         the nanobrain base checkout: its .venv runs everything (or $NB_PYTHON, loop.project.python), its .env
#                sets BRAINMARKS_SMRI_DATA
#   MODE         full (default): train + eval.  smoke: train + a checkpoint load / forward check (no eval)
#
# Run from the node's worktree (sbatch's cwd = SLURM_SUBMIT_DIR). Every phase runs through GUARD (below) with -I:
# sys.path = <worktree>/src + the base venv only (the base checkout's editable src is DROPPED, so a module the
# candidate deleted can never silently fall back to the base copy of the nanobrain namespace package; no cwd /
# PYTHONPATH entries, so a planted brainmarks_smri / torch / sitecustomize cannot shadow the real one). The run trains
# (and the eval embeds with) the CANDIDATE's code and misc.git_info records the worktree commit.
#
# Integrity bookkeeping the candidate's code does not drive: OUT.loopmeta/instances.jsonl (one line per event, with
# `date +%s`): train_start (resume = checkpoint-last.pth existed and auto_resume) / train_end (rc, done, stopped) /
# train_done / eval_start / eval_end / eval_fabricated. The adapter sums the train-phase wall of every instance from
# it (the 60-minute budget), counts resumed instances, and refuses an eval dir that existed before the first eval.
# OUT.loopmeta/heartbeat is touched each minute the running phase's processes gained CPU time (a futex-hung job does
# not, so freeze detection still fires).
#
# Training sees a SHADOW data root holding only fomo300k (the eval datasets are not reachable through
# BRAINMARKS_SMRI_DATA while candidate training code runs); the eval gets the real root.
#
# NEVER calls scripts/submit.py: the loop submits to Labless itself (loop.labless), and only its incumbent.
# Requeue-safe: a requeued job re-runs this script; training resumes from OUT/checkpoint-last.pth (auto_resume) or is
# skipped once this script recorded train_done, and the eval skips every task whose json exists.
# SIGTERM (preemption / time limit / scancel) is forwarded to the running phase, which then saves and exits.
# SBATCH_ACCOUNT / QOS / PARTITION / TIMELIMIT / REQUEUE / JOB_NAME come from the loop's env (they beat these).
#SBATCH --job-name=nanobrain-loop
#SBATCH --gpus=1
#SBATCH --open-mode=append
#SBATCH --output=slurm/%j.out

set -euo pipefail

CFG=$1
OUT=$2
SEED=$3
AUTO_RESUME=$4
REPO=$5
MODE=${6:-full}

cd "${SLURM_SUBMIT_DIR:-$PWD}"
WT=$PWD

if [[ -f "${REPO}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO}/.env"
    set +a
fi
: "${BRAINMARKS_SMRI_DATA:?BRAINMARKS_SMRI_DATA is not set (no ${REPO}/.env?)}"
REAL_DATA=${BRAINMARKS_SMRI_DATA}
export BRAINMARKS_SMRI_DATA
export PYTHONPATH="${WT}/src${PYTHONPATH:+:${PYTHONPATH}}"     # informational: GUARD's -I builds sys.path itself
read -ra EXTRA_OV <<< "${NB_OVERRIDES:-}"   # NB_OVERRIDES: space-separated key=value dotlist appended to train --overrides (per-arm tweaks, no new yaml)
PY="${NB_PYTHON:-${REPO}/.venv/bin/python}"      # NB_PYTHON: loop.project.python (a re-rooted base with its own venv)
export NB_VENV="$(dirname "$(dirname "${PY}")")"   # GUARD requires torch / numpy / ... to resolve under it
META="${OUT}.loopmeta"
HEARTBEAT_S=${NANOBRAIN_HEARTBEAT_S:-60}

echo "nanobrain_job: mode=${MODE} worktree=${WT} out=${OUT} seed=${SEED} auto_resume=${AUTO_RESUME}" \
     "job=${SLURM_JOB_ID:-} restarts=${SLURM_RESTART_COUNT:-0}"
mkdir -p "$(dirname "${OUT}")" "${META}"

# per-job scratch: the shadow data root and the inductor / triton caches (never shared between jobs)
JOBTMP=$(mktemp -d "${TMPDIR:-/tmp}/nanobrain-loop-${SLURM_JOB_ID:-$$}.XXXXXX")
mkdir -p "${JOBTMP}/data" "${JOBTMP}/inductor" "${JOBTMP}/triton"
ln -s "${REAL_DATA}/fomo300k" "${JOBTMP}/data/fomo300k"
export TORCHINDUCTOR_CACHE_DIR="${JOBTMP}/inductor" TRITON_CACHE_DIR="${JOBTMP}/triton"

PHASE_PID=""
HB_PID=""
cleanup() {
    [[ -n "${HB_PID}" ]] && kill "${HB_PID}" 2>/dev/null || true
    [[ -n "${PHASE_PID}" ]] && kill -KILL -- "-${PHASE_PID}" 2>/dev/null || true
    rm -rf "${JOBTMP}"
}
trap cleanup EXIT
on_term() {
    echo "nanobrain_job: SIGTERM, forwarded to ${PHASE_PID:-no phase}"
    [[ -n "${PHASE_PID}" ]] && kill -TERM "${PHASE_PID}" 2>/dev/null || true
}
trap on_term TERM

event() {   # event NAME [,"key":value...]
    printf '{"event":"%s","ts":%s,"job":"%s","restart":%s%s}\n' "$1" "$(date +%s)" "${SLURM_JOB_ID:-}" \
           "${SLURM_RESTART_COUNT:-0}" "${2:-}" >> "${META}/instances.jsonl"
}
seen() { [[ -f "${META}/instances.jsonl" ]] && grep -q "\"event\":\"$1\"" "${META}/instances.jsonl"; }

cpu_s() { ps -s "$1" -o times= 2>/dev/null | awk '{s += $1} END {print s + 0}'; }
heartbeat() {   # heartbeat SID: touch META/heartbeat each interval the session's processes gained CPU time
    local sid=$1 last now
    last=$(cpu_s "${sid}")
    while sleep "${HEARTBEAT_S}"; do
        now=$(cpu_s "${sid}")
        if [[ "${now}" != "${last}" ]]; then
            date +%s > "${META}/heartbeat"
            last=${now}
        fi
    done
}

run_phase() {   # run_phase TARGET ARGS...: TARGET (a module / a .py) under GUARD in its own session (setsid);
                # TERM forwarded, leftovers killed; returns its rc
    local rc=0
    setsid "${PY}" -I -c "${GUARD}" "${WT}" "${REPO}" "$@" &
    PHASE_PID=$!
    heartbeat "${PHASE_PID}" </dev/null >/dev/null 2>&1 &
    HB_PID=$!
    wait "${PHASE_PID}" || rc=$?
    while kill -0 "${PHASE_PID}" 2>/dev/null; do      # a trapped TERM interrupted wait: the phase is still saving
        rc=0
        wait "${PHASE_PID}" || rc=$?
    done
    kill "${HB_PID}" 2>/dev/null || true
    wait "${HB_PID}" 2>/dev/null || true
    kill -KILL -- "-${PHASE_PID}" 2>/dev/null || true  # loader workers / anything the phase left behind
    PHASE_PID=""
    HB_PID=""
    return "${rc}"
}

# M1: the worktree's src may hold the nanobrain package only (no second top-level package / sitecustomize)
for entry in "${WT}/src"/* "${WT}/src"/.[!.]*; do
    [[ -e "${entry}" ]] || continue
    case "$(basename "${entry}")" in
        nanobrain|nanobrain.egg-info|__pycache__) ;;
        *) echo "nanobrain_job: REFUSED: src/$(basename "${entry}") is not the nanobrain package"
           event refused ',"why":"foreign src entry"'
           exit 2 ;;
    esac
done

GUARD='
import importlib.util, os, runpy, sys
wt, base, target = sys.argv[1], sys.argv[2], sys.argv[3]
drop = {os.path.realpath(os.path.join(base, "src")), os.path.realpath(wt)}
# a sys.path entry holding ANY other nanobrain tree is dropped too: the editable src of the checkout that owns the venv
# (loop.project.python = another checkout venv; live 11-08: wave1-base/src shadowed the worktree, every job failed)
sys.path[:] = [p for p in sys.path if p and os.path.realpath(p) not in drop
               and not os.path.isdir(os.path.join(p, "nanobrain"))]
sys.path.insert(0, os.path.join(wt, "src"))
venv = os.path.realpath(os.environ.get("NB_VENV") or os.path.join(base, ".venv")) + os.sep
def fail(why):
    print(f"nanobrain_guard: FAILED {why}", flush=True)
    sys.exit(3)
for name in ("brainmarks_smri", "torch", "numpy", "omegaconf"):
    spec = importlib.util.find_spec(name)
    origin = os.path.realpath(spec.origin) if spec is not None and spec.origin else ""
    if not origin.startswith(venv):
        fail(f"{name} resolves to {origin or None!r}, not the base venv")
spec = importlib.util.find_spec("nanobrain")
locs = [os.path.realpath(p) for p in (spec.submodule_search_locations or [])] if spec is not None else []
if locs != [os.path.realpath(os.path.join(wt, "src", "nanobrain"))]:
    fail(f"nanobrain resolves to {locs}, not only the worktree")
sys.argv = [target] + sys.argv[4:]
if target.endswith(".py"):
    runpy.run_path(target, run_name="__main__")
else:
    runpy.run_module(target, run_name="__main__", alter_sys=True)
'

summary_flag() {   # summary_flag KEY: True / False / missing
    "${PY}" -I -c 'import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]) is True)' \
        "${OUT}/summary.json" "$1" 2>/dev/null || echo missing
}

# ---------------------------------------------------------------- train (skipped once this script saw it done)
if ! seen train_done; then
    RESUME=false
    [[ "${AUTO_RESUME}" == "true" && -f "${OUT}/checkpoint-last.pth" ]] && RESUME=true
    event train_start ",\"resume\":${RESUME}"
    RC=0
    BRAINMARKS_SMRI_DATA="${JOBTMP}/data" run_phase nanobrain.train --config "${CFG}" --overrides \
        "output_dir=${OUT}" "name=$(basename "${OUT}")" "seed=${SEED}" "auto_resume=${AUTO_RESUME}" "${EXTRA_OV[@]}" || RC=$?
    DONE=$(summary_flag done)
    STOPPED=$(summary_flag stopped_by_signal)
    event train_end ",\"rc\":${RC},\"done\":$([[ ${DONE} == True ]] && echo true || echo false),\"stopped\":$([[ ${STOPPED} == True ]] && echo true || echo false)"
    if [[ "${RC}" != 0 ]]; then
        echo "nanobrain_job: training exited rc=${RC}"
        exit "${RC}"
    fi
    if [[ "${DONE}" != "True" ]]; then
        # stopped by a signal (preemption / scancel / time limit): nothing to evaluate; a requeue resumes it
        echo "nanobrain_job: training not done (see ${OUT}/summary.json); no eval"
        exit 75
    fi
    event train_done
fi

if [[ "${MODE}" == "smoke" ]]; then
    # the checkpoint loads the way brainmarks loads it (nanobrain.model.vit_mae_3d through create_model) and one
    # forward_embeddings pass on a synthetic volume gives finite global + dense embeddings
    cat > "${JOBTMP}/loadcheck.py" <<'PYEOF'
import importlib
import sys

import numpy as np
import torch

importlib.import_module("nanobrain.model")          # registers vit_mae_3d, like the eval's imports=[nanobrain.model]
from brainmarks_smri.models import create_model

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model("vit_mae_3d", checkpoint=sys.argv[1]).to(device).eval()
    grid = tuple(int(s) for s in model.grid_size)
    mask = torch.zeros(grid, dtype=torch.bool)
    box = tuple(slice(s // 4, s - s // 4) for s in grid)
    mask[box] = True
    gen = torch.Generator().manual_seed(0)
    values = torch.randn(int(mask.sum()), generator=gen)
    out = model.forward_embeddings([{"values": values, "mask": mask, "affine": np.eye(4)}], return_dense=True)[0]
    glob, dense = out["global_embedding"], out["dense_embedding"]
    assert glob.ndim == 1 and glob.numel() > 0, f"global embedding shape {tuple(glob.shape)}"
    assert bool(torch.isfinite(glob).all()) and bool(torch.isfinite(dense).all()), "non-finite embedding"
except Exception as exc:  # noqa: BLE001
    print(f"nanobrain_loadcheck: FAILED {type(exc).__name__}: {exc}", flush=True)
    raise
print(f"nanobrain_loadcheck: ok global={tuple(glob.shape)} dense={tuple(dense.shape)}", flush=True)
PYEOF
    RC=0
    run_phase "${JOBTMP}/loadcheck.py" "${OUT}/checkpoint-last.pth" || RC=$?
    rm -f "${OUT}/checkpoint-last.pth"                 # ~1.2 GB per smoke
    exit "${RC}"
fi

# ---------------------------------------------------------------- eval
# An eval dir before this script ever started the eval was written by the candidate's training code (it would make
# brainmarks skip those tasks as "done"): moved aside, recorded, no eval -- the adapter voids the run.
if ! seen eval_start && [[ -e "${OUT}/eval" ]]; then
    mv "${OUT}/eval" "${OUT}/eval.pre-eval.$(date +%s)"
    event eval_fabricated
    echo "nanobrain_job: REFUSED: ${OUT}/eval existed before the eval started (written by training code)"
    exit 0
fi
# Evaluate the last checkpoint (brainmarks-smri open). Finished tasks are skipped when requeued / continued.
event eval_start
RC=0
run_phase brainmarks_smri vit_mae_3d open --overrides \
    "imports=[nanobrain.model]" \
    "model_kwargs.checkpoint=${OUT}/checkpoint-last.pth" \
    "output_dir=${OUT}" \
    name=eval || RC=$?
event eval_end ",\"rc\":${RC}"
exit "${RC}"
