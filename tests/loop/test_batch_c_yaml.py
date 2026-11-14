"""10-22 batch C4: the dead YAML sections (no live v3 reader) are gone from tasks/nanopath_v2.yaml; the cleaned spec loads with
the strict live loader (an unknown key would refuse to start) and the real Tick builds on it. The sections that LOOK dead but are
read (forbid_config at spec load, the lever_p the intake's sweep redirect reads, the ported components) stay."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

from loop import tick as TK
from loop.spec import load_context

from .test_set1_harness import _Held

V2 = Path(__file__).resolve().parents[2] / "tasks" / "nanopath_v2.yaml"
GONE = {"backfill": None, "bandit": None, "scout": None, "review": None, "moonshot_tune": None, "campaign": None, "legs": None,
        "slurm": ("scarce_gpus",), "seeds": ("audit_rotate", "confirm_rotate", "control_wait_h"),
        "promotion": ("head_p", "remint_hours"), "ideation": ("base_ladder_p", "max_points", "branch_depth"),
        "researcher": ("max_side", "max_stack"), "design": ("stall_closes", "max_knobs", "max_share")}
KEPT = {"design": ("lever_p",), "components": ("ablation_rate", "min_ablated"), "novelty": ("sweep_redirect", "major_ratio", "dead_extra"),
        "ideation": ("operator_programs", "operator_findings", "struct_tune_knobs"), "seeds": ("promote_per_node", "max_seeds_per_node"),
        "slurm": ("pools", "protect_jobs")}


@pytest.fixture(scope="module")
def raw():
    return yaml.safe_load(V2.read_text())


def test_the_dead_sections_and_fields_are_gone(raw):
    loop = raw["loop"]
    for sec, fields in GONE.items():
        if fields is None:
            assert sec not in loop, sec
        else:
            assert not set(fields) & set(loop[sec]), (sec, set(fields) & set(loop[sec]))
    assert not {"frontier_refresh_every", "no_improve_commits"} & set(raw["triggers"])
    assert not {"escalation_cap_per_cycle", "wave_fault_abort"} & set(raw["budget"]) and "champion_new_seed_evidence" not in raw["decision"]


def test_what_is_still_read_stays(raw):
    loop = raw["loop"]
    for sec, fields in KEPT.items():
        assert set(fields) <= set(loop[sec]), (sec, fields)
    assert loop["forbid_config"] and loop["init_compat"] and loop["labless"] and "operator_findings" in loop["ideation"]
    assert raw["caps"]["infra_retry"] == 8 and {"min_seeds", "verdict_sigmas", "screen"} <= set(raw["decision"])     # caps untouched
    assert raw["surface"]["protected_keys"]


def test_the_cleaned_spec_loads_with_the_strict_live_loader_and_the_real_tick_builds_on_it(tmp_path):
    ctx = load_context(V2, state_dir=tmp_path / "state", require_measured=False)
    assert ctx.cfg.design.lever_p == 0.8 and ctx.cfg.forbid_config == ("configs/main.yaml:probe",) and ctx.cfg.forbid_roots
    assert ctx.cfg.components.ablation_rate == 0.15 and ctx.cfg.promotion.p_promote == 0.6
    from loop.store import SqliteStore
    store = SqliteStore(tmp_path / "loop.db", ctx.task_id, weights=ctx.cfg.weights)
    tick = TK.Tick(ctx, store, _Held(), TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)
    assert " pools=" in tick.heartbeat(1, 0.5)
    tick.close()
    store.close()


def test_an_unknown_key_is_still_refused(tmp_path, raw):
    bad = dict(raw)
    bad["loop"] = {**raw["loop"], "backfill": {"enabled": True}, "labless_typo": 1}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="unknown key"):
        load_context(p, state_dir=tmp_path / "s", require_measured=False)
