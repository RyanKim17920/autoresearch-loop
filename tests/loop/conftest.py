"""Fixtures for the loop tests. Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import os
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable

import pytest

from loop.spec import load_context
from loop.store import SqliteStore

REPO = Path(__file__).resolve().parents[2]

# tasks/nanopath_v2.yaml reads its repo's configs/main.yaml (forbid_config). Without a real nanopath checkout in
# NANOPATH_REPO, use the minimal stand-in under tests/fixtures (set before any spec is loaded).
if not (Path(os.environ.get("NANOPATH_REPO", "")) / "configs" / "main.yaml").is_file():
    os.environ["NANOPATH_REPO"] = str(REPO / "tests" / "fixtures" / "nanopath_repo")

# tasks/nanobrain.yaml requires NANOBRAIN_BASE_SHA (the root commit); the offline tests use a placeholder
os.environ.setdefault("NANOBRAIN_BASE_SHA", "0123456789abcdef0123456789abcdef01234567")

#: example operator-seeded programs (loop.ideation.operator_programs); the shipped task specs carry none
EXAMPLE_PROGRAMS = Path(__file__).parent / "fixtures" / "example_programs.yaml"


def example_programs() -> list:
    import yaml
    return yaml.safe_load(EXAMPLE_PROGRAMS.read_text())


@pytest.fixture
def fake_loop_spec() -> Path:
    return REPO / "tasks" / "fake_loop.yaml"


@pytest.fixture
def ctx(fake_loop_spec, tmp_path):
    return load_context(fake_loop_spec, state_dir=tmp_path / "state")


@pytest.fixture
def store(ctx, tmp_path):
    s = SqliteStore(tmp_path / "loop.db", ctx.task_id, weights=ctx.cfg.weights)
    yield s
    s.close()


class InlineExecutor:
    """A synchronous executor: the driver's LLM work runs inline (deterministic tests)."""

    def __init__(self, max_workers: int | None = None) -> None:
        pass

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - delivered through the future
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait: bool = True, **_: Any) -> None:
        pass


SLURM_BINS = ("sbatch", "scancel", "squeue", "sacct", "srun", "salloc", "scontrol")


@pytest.fixture(scope="session", autouse=True)
def _forbid_real_slurm(tmp_path_factory):
    """Hard guard: no test may ever reach the real SLURM. Fake slurm binaries shadow the real ones
    on PATH (inherited by CLI subprocesses) and LOOP_FORBID_SLURM makes the nanopath adapter's
    SubprocessRunner raise before exec. Adapter tests with their own fake runners are unaffected."""
    import os
    fake = tmp_path_factory.mktemp("fake_slurm_bin")
    for name in SLURM_BINS:
        f = fake / name
        f.write_text('#!/bin/sh\necho "REAL SLURM CALLED FROM TESTS: $0 $*" >&2\nexit 1\n')
        f.chmod(0o755)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PATH", f"{fake}{os.pathsep}{os.environ.get('PATH', '')}")
        mp.setenv("LOOP_FORBID_SLURM", "1")
        yield


def _unmeasure(text: str) -> tuple[str, int]:
    """noise.{seed_transfer,replicate,same_seed}_sd, loop.panel_sd, and every loop.axis_sd.* entry."""
    import re
    n = 0
    text, k = re.subn(r"^(  (?:same_seed_sd|seed_transfer_sd|replicate_sd|panel_sd)):[^\n]*$",
                      r"\1: UNMEASURED", text, flags=re.M)
    n += k
    m = re.search(r"^  axis_sd:\n((?:    \w+:[^\n]*\n)+)", text, flags=re.M)
    assert m, "axis_sd block not found"
    block = re.sub(r"^(    \w+):[^\n]*$", r"\1: UNMEASURED", m.group(1), flags=re.M)
    return text[:m.start(1)] + block + text[m.end(1):], n + block.count("UNMEASURED")


@pytest.fixture
def unmeasured_v2_spec(tmp_path) -> Path:
    """tmp copy of tasks/nanopath_v2.yaml with every measured noise.*_sd / loop.panel_sd /
    loop.axis_sd.* put back to UNMEASURED, so tests never depend on the live spec's state."""
    text = (REPO / "tasks" / "nanopath_v2.yaml").read_text()
    text, n = _unmeasure(text)
    assert n >= 3, n
    p = tmp_path / "nanopath_v2_unmeasured.yaml"
    p.write_text(text)
    return p


@pytest.fixture(autouse=True)
def _acquire_constants():
    """``acquire.configure`` (each tick) installs the operator keys and the target as module constants: restore them
    after every test so one test's config never leaks into another's acquisition."""
    from loop.core import acquire as A
    from loop.core import status as ST
    names = ("TARGET_MODE", "FRESH_SHRINK", "GAP_FLOOR_FRAC", "STACK_BEST_LEG_FRAC", "STACK_SUM_FRAC",
             "STACK_INTERCEPT", "UNTESTED_VAR_MULT", "IDLE_MAX_RESTARTS", "IDLE_MAX_RUN_FRAC", "TARGET",
             "GENERIC_FAMILIES", "GAP_LEVEL_MIN_TEST", "TARGET_ACTIVE", "IDEA_LANE_PRIOR",
             "PROGRAM_ARM_SHARE", "QUEUED_MAX_AGE_H")
    saved = {n: getattr(A, n) for n in names}
    st = (ST.MAX_SEED_SHARE, ST.SEED_GATE_MIN, ST.SHARE_MIN_N, ST.SHARE_K, ST.TRIM_MIN_N, ST.TRIM_FRAC)
    yield
    for n, v in saved.items():
        setattr(A, n, v)
    ST.MAX_SEED_SHARE, ST.SEED_GATE_MIN, ST.SHARE_MIN_N, ST.SHARE_K, ST.TRIM_MIN_N, ST.TRIM_FRAC = st


@pytest.fixture(autouse=True)
def _node_vector_cache():
    """``ideation.node_vector`` caches by node id for the process: a test on a copy of the live ledger (n0001 ...) must not
    leak its vectors into a test that stores another n0001."""
    from loop.core import ideation as ID
    ID._EMB.clear()
    yield
    ID._EMB.clear()


@pytest.fixture(autouse=True)
def _project_reset():
    """loop.project is a per-process registry (load_context installs a spec's loop.project block): every test starts
    and ends on the nanopath defaults, so one test's install never leaks into the next."""
    from loop import project as P
    P.reset()
    yield
    P.reset()
