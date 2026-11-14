"""Operator findings (10-21): the operator's manual experiments are rendered at the top of every research brief as
measured facts; earlier LLM sessions' prose (postmortems, handoffs) is labelled as hypotheses; the sampler-flatten
operator program no longer carries the arms the operator measured null."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from loop.agents import briefs as BR
from loop.core import data as D
from loop.core import ideation as ID
from loop.spec import IdeationCfg
from loop.store import SqliteStore

from ..test_tick import _ledger
from .conftest import TASK
from .test_ideation import README

ROOT = Path(__file__).resolve().parents[3]
FIND = ["DEAD: schedule clock = sample fraction: -.0091 +- .0005", "OPEN: site-level flattening"]


def _with(ctx, **ideation):
    base = ctx.cfg.ideation or IdeationCfg()
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(base, **ideation)))


@pytest.fixture
def fenv(ctx, tmp_path):
    db = _ledger(tmp_path / "ledger.db")
    ctx = _with(dataclasses.replace(ctx, task_id=TASK), operator_findings=tuple(FIND), diverge_explore=True)
    store = SqliteStore(db, TASK)
    readme = tmp_path / "README.md"
    readme.write_text(README)
    yield ctx, store, D.load(db, TASK), ID.Ideation(ctx, store, readme=readme)
    store.close()


def test_findings_block_is_empty_without_findings_and_lists_them_as_facts(ctx):
    assert BR.findings_block(_with(ctx, operator_findings=())) == ""
    got = BR.findings_block(_with(ctx, operator_findings=tuple(FIND)))
    assert got.startswith(BR.FINDINGS_HEADING) and all(f"- {x}" in got for x in FIND)
    assert "outrank every postmortem" in got and "Do not re-propose a DEAD / NULL" in got


def test_every_brief_kind_leads_with_the_findings_before_the_computed_tables(fenv):
    ctx, store, led, ide = fenv
    for kind in ("explore", "bold"):
        text = ide.brief_for("explore", led, kind=kind).text
        assert "## Operator's manual experiments" in text and FIND[0] in text, kind
    text = ide.brief_for("explore", led, kind="explore").text
    assert text.index("## Operator's manual experiments") < text.index("## Per-axis headroom")


def test_a_brief_without_findings_has_no_findings_section(fenv):
    ctx, store, led, ide = fenv
    ide.ctx = _with(ctx, operator_findings=())
    assert "Operator's manual experiments" not in ide.brief_for("explore", led, kind="explore").text
    assert "Operator's manual experiments" not in ide.brief_for("explore", led, kind="bold").text


def test_the_explore_head_asks_for_a_new_mechanism_and_calls_old_prose_a_guess():
    assert "not a dose of a term" in ID.KIND_TEXT["explore"] and "guesses" in ID.KIND_TEXT["explore"]


def test_postmortem_block_marks_prose_as_hypothesis_and_drops_the_steering_suggestion():
    pm = SimpleNamespace(node_id="n1", verdict="null", explanation="clean null, flat", suggested_next="pivot to readout")
    got = BR.postmortem_block([pm])
    assert got.startswith(BR.PROSE_NOTE) and "[n1 null] clean null, flat" in got and "pivot to readout" not in got
    assert BR.postmortem_block([]) == "(none)"


def test_the_task_spec_carries_the_manual_findings_and_the_sampler_program_lost_its_null_arms():
    raw = yaml.safe_load((ROOT / "tasks" / "nanopath_v2.yaml").read_text())
    ide = raw["loop"]["ideation"]
    from loop import spec as SP
    from tests.loop.conftest import example_programs
    cfg = SP.build(SP.IdeationCfg, {**ide, "operator_programs": example_programs()}, "loop.ideation")
    assert cfg.operator_findings == ()            # the public spec ships none (they are the operator's manual results)
    prog = next(p for p in cfg.operator_programs if p["id"] == "op-sampler-flatten")
    assert [a["id"] for a in prog["arms"]] == ["site_cap", "project_cap"]
    text = ID.program_text(prog)
    assert "aug_patient_cap_quantile {0.75" not in text and "patient-uniform sampling" not in text.split("Components:")[-1]
    assert "TCGA PROJECT" in text and "MEASURED EXHAUSTED" in text
