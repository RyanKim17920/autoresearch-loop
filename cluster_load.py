"""cluster_load.py — cheap, honest read of GPU headroom on the SLURM cluster.

WHY THIS EXISTS
    The train track's internal full-slot semaphore used to be a fixed 24. That
    is the ceiling the EXTERNAL launch gate enforces (job_cap.sh CAP /
    `safe_launch_gate.sh wait 24`), not a sensible steady-state load: when the
    cluster is busy with other users' work, 24 in-flight `main_1gpu` jobs just
    pile up PENDING and starve everyone (including us). The operator wants a
    small, adaptive number instead: 8 concurrent trainings when the cluster is
    free, 5 when it is contended.

THE SIGNAL — GPU HEADROOM AVAILABLE TO US
    `main_1gpu` and `pp_eval` each consume exactly one H100. So the resource
    that actually decides whether another one of our jobs will RUN (rather than
    queue) is *free GPUs on the GPU nodes*. We read it straight from SLURM's own
    accounting:

        free_gpus = sum(Gres) - sum(GresUsed)        over usable GPU nodes

    That number counts EVERY user's occupancy, which is the point — node-2/node-3/node-4
    /node-8 are routinely full of other people's jobs while showing free CPUs.

    We then add back the GPUs held by OUR OWN gated jobs (`main_1gpu` +
    `pp_eval`, the exact set job_cap.sh counts):

        headroom = free_gpus + our_gated_gpu_jobs

    This makes the signal measure EXTERNAL pressure only. Without it the metric
    is self-referential — every job we launch shrinks the headroom that decides
    how many jobs we may launch — so a cold start would read "free", ramp to 8,
    then read "contended" purely because of its own success.

WHY NOT PENDING-QUEUE DEPTH
    A queue-depth term was considered and rejected. On this cluster jobs pend
    for reasons unrelated to load (QOS, priority, per-user limits) — there are
    PENDING jobs from months ago sitting behind a bottom-tier QOS. Depth would
    therefore read "contended" permanently and the free-branch would be dead
    code. Idle-GPU headroom is the signal that actually predicts whether our
    next sbatch starts.

WHY NOT CPU IDLE COUNTS
    `sinfo`'s CPUS(A/I/O/T) shows node-5/node-6 with 64 idle CPUs while both nodes
    have all 8 GPUs allocated. Idle CPUs on a GPU node do not mean a GPU job
    can start. Using them would systematically over-report freedom.

UNKNOWN IS NOT FREE
    Every probe failure path (squeue/sinfo missing, non-zero exit, unparseable
    output, no GPU nodes found, unresolvable user) returns status="unknown".
    Callers must map unknown to the CONSERVATIVE value and say so out loud —
    never conflate "cannot tell" with "free". This mirrors the fail-closed
    convention of job_cap.sh / safe_launch_gate.sh.

No import-time side effects: nothing here runs a subprocess until called.
"""

from __future__ import annotations

import os
import subprocess

# Job names whose GPUs are "ours" for headroom purposes. Exactly the set
# job_cap.sh / safe_launch_gate.sh count — the jobs the internal semaphore and
# the external gate both govern. Side-runs (bestA_plain, qwen-serve-*, the
# segpath runs) are deliberately NOT in here: they are real external occupancy.
GATED_JOB_NAMES = ("main_1gpu", "pp_eval")

# Node states in which a node's GPUs are NOT schedulable for us. Anything else
# (idle / mixed / allocated, with or without a trailing '*' or '$') counts.
_UNUSABLE_STATE_MARKERS = (
    "down", "drain", "drng", "fail", "maint", "unknown", "reserved",
    "power", "invalid", "boot", "planned",
)

_SUBPROCESS_TIMEOUT = 30


class _ProbeFailure(Exception):
    """Internal: a SLURM probe could not produce a trustworthy answer."""


def _run(argv: list[str]) -> str:
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, check=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError) as e:
        raise _ProbeFailure(f"{argv[0]} failed: {e}") from e
    return proc.stdout


def _resolve_user() -> str:
    """USER -> LOGNAME -> passwd db. REFUSES rather than returning "".

    An empty -u filter makes squeue answer with every user's jobs (or, with
    other flag combinations, zero rows) — both are silent lies. Same rule as
    tools-v2/lib/common.resolve_slurm_user.
    """
    for var in ("USER", "LOGNAME"):
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
    try:
        import getpass
        val = (getpass.getuser() or "").strip()
    except Exception as e:  # noqa: BLE001
        raise _ProbeFailure(f"cannot resolve the SLURM user: {e}") from e
    if not val:
        raise _ProbeFailure("cannot resolve the SLURM user (empty)")
    return val


def parse_gres_gpu_count(field: str) -> int:
    """Count GPUs in one sinfo Gres/GresUsed field.

    Handles the shapes SLURM emits here:
        gpu:nvidia_h100_80gb_hbm3:8(S:0-1)
        gpu:nvidia_h100_80gb_hbm3:7(IDX:0,2-7)
        gpu:8
        (null)  /  ""  /  gpu:0
    Multiple comma-separated gres entries are summed; non-gpu entries ignored.

    NOTE the comma split is deliberately done on the *bracket-stripped* text —
    an IDX list like "(IDX:0,2-7)" contains commas that are not entry
    separators, and splitting first would shred it into bogus entries.
    """
    field = (field or "").strip()
    if not field or field == "(null)":
        return 0
    # Strip every parenthesised suffix first, so remaining commas separate
    # gres entries only.
    out, depth = [], 0
    for ch in field:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    total = 0
    for entry in "".join(out).split(","):
        entry = entry.strip()
        if not entry or not entry.startswith("gpu:"):
            continue
        tail = entry.rsplit(":", 1)[-1]
        try:
            total += int(tail)
        except ValueError:
            # "gpu" with no count, or a type name we cannot read as a number.
            # Unparseable means unknown; fail closed rather than count 0.
            raise _ProbeFailure(f"unparseable gres entry: {entry!r}") from None
    return total


def _node_usable(state: str) -> bool:
    s = (state or "").strip().lower().rstrip("*~#$@+")
    if not s:
        return False
    return not any(m in s for m in _UNUSABLE_STATE_MARKERS)


def _sinfo_gpu_totals() -> tuple[int, int, int]:
    """Return (total_gpus, used_gpus, n_nodes) over usable GPU nodes.

    Nodes are deduped by hostname: a node in two partitions (main* and n) is
    listed twice by `sinfo -N` and must not be double-counted.
    """
    raw = _run(["sinfo", "-h", "-N", "-O",
                "NodeHost:.40,Gres:.80,GresUsed:.80,StateLong:.30"])
    seen: dict[str, tuple[int, int]] = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        node, gres, used, state = parts[0], parts[1], parts[2], parts[3]
        if node in seen:
            continue
        total_g = parse_gres_gpu_count(gres)
        if total_g <= 0:
            continue  # not a GPU node (c-* login/CPU boxes)
        if not _node_usable(state):
            continue
        used_g = parse_gres_gpu_count(used)
        seen[node] = (total_g, min(used_g, total_g))
    if not seen:
        raise _ProbeFailure("sinfo listed no usable GPU nodes")
    total = sum(t for t, _ in seen.values())
    used = sum(u for _, u in seen.values())
    return total, used, len(seen)


def _our_gated_gpu_jobs() -> int:
    """Active (RUNNING|PENDING|CONFIGURING) jobs of ours named in
    GATED_JOB_NAMES. These are the GPUs the internal cap itself is issuing, so
    they are added back to the headroom to keep the signal external-only."""
    user = _resolve_user()
    raw = _run(["squeue", "-u", user, "-h",
                "-t", "RUNNING,PENDING,CONFIGURING", "-o", "%j"])
    return sum(1 for ln in raw.splitlines()
               if ln.strip() in GATED_JOB_NAMES)


def gpu_headroom(free_threshold: int) -> dict:
    """Classify cluster load for the train track.

    Returns a dict — ALWAYS, never raises:
        status      "free" | "contended" | "unknown"
        headroom    int | None   free GPUs + our own gated GPU jobs
        free_gpus   int | None   idle GPUs across usable GPU nodes
        total_gpus  int | None
        used_gpus   int | None
        our_jobs    int | None
        gpu_nodes   int | None
        reason      str          human-readable, always populated

    "free" iff headroom >= free_threshold, i.e. the cluster could actually
    absorb the whole free-branch cap right now. Anything we cannot determine is
    "unknown" — the caller is required to treat that as contended and log it.
    """
    unknown = {"status": "unknown", "headroom": None, "free_gpus": None,
               "total_gpus": None, "used_gpus": None, "our_jobs": None,
               "gpu_nodes": None}
    try:
        total, used, nodes = _sinfo_gpu_totals()
    except _ProbeFailure as e:
        return {**unknown, "reason": f"sinfo probe failed: {e}"}
    try:
        ours = _our_gated_gpu_jobs()
    except _ProbeFailure as e:
        return {**unknown, "reason": f"squeue probe failed: {e}"}

    free = max(0, total - used)
    headroom = free + ours
    status = "free" if headroom >= free_threshold else "contended"
    return {
        "status": status,
        "headroom": headroom,
        "free_gpus": free,
        "total_gpus": total,
        "used_gpus": used,
        "our_jobs": ours,
        "gpu_nodes": nodes,
        "reason": (f"{free} idle GPU(s) of {total} across {nodes} node(s) "
                   f"+ {ours} of our own gated job(s) = headroom {headroom} "
                   f"(free needs >= {free_threshold})"),
    }
