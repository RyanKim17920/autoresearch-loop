"""10-22: the operator focus leads every v3 brief (explore, bold, program, fnf branch) and the plateau review; it was
injected only by briefs.brief()/the analyst, which the v3 researcher never calls."""

from __future__ import annotations

import dataclasses

from loop.agents.briefs import FOCUS_HEADING
from loop.core import ideation as ID

from .test_c8b_bold import benv  # noqa: F401  (the fixture)

FOCUS = "Think from first principles; respect the hard constraints: no pathology init."


def _focused(ide):
    ide.ctx = dataclasses.replace(ide.ctx, cfg=dataclasses.replace(ide.ctx.cfg, operator_focus=FOCUS))


def test_focus_leads_explore_bold_and_program_briefs(benv):
    ctx, store, led, ide = benv
    _focused(ide)
    for kind in ("explore", "bold", "program", "literature"):
        b = ide.brief_for("explore", led, kind=kind)
        assert f"{FOCUS_HEADING}\n{FOCUS}" in b.text, kind
        assert b.text.index(FOCUS_HEADING) < 400, kind          # near the top, not buried
    assert ide.request(b, ide.view(led)).brief.count(FOCUS) == 1


def test_focus_leads_the_plateau_review_and_unset_adds_nothing(benv):
    ctx, store, led, ide = benv
    v = ide.view(led)
    p = ID.Plateau(True, 0.0, 100.0, 10.0, 2)
    _focused(ide)
    assert ide.review_prompt(v, p, {}).startswith(f"{FOCUS_HEADING}\n{FOCUS}")
    ide.ctx = dataclasses.replace(ide.ctx, cfg=dataclasses.replace(ide.ctx.cfg, operator_focus=""))
    assert FOCUS_HEADING not in ide.review_prompt(v, p, {})
    assert FOCUS_HEADING not in ide.brief_for("explore", led, kind="bold").text


def test_focus_leads_a_flushed_branch_brief(benv):
    ctx, store, led, ide = benv
    _focused(ide)
    v = ide.view(led)
    b = ide.branch_brief("explore", v, "explore", "A")
    assert f"{FOCUS_HEADING}\n{FOCUS}" in b.text
