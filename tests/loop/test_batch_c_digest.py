"""10-22 batch C2: the 3-hour digests stopped 10-07 (loop.digest unreachable from the v3 tick). Every meta report (the meta
cycle runs every ``core_meta.every_h`` of loop time) now writes a compact milestone digest to ``state/digests/``."""

from __future__ import annotations

import dataclasses
import time

from loop import digest as DG
from loop import tick as TK
from loop.core import data as CD

from .test_set1_harness import _Held

REP = {"ts": 1_790_000_000.0, "since": 1_789_989_200.0, "incumbent": "n9",
       "switches": ["n9 +0.0031 (n=5)"], "tests": {"closed": 48, "window": 2, "passed": 4, "pass_rate": 0.083, "mean_n": 5.75},
       "bar": {"best": "n3029 local mean .6826", "text": "bar=.6850", "beats": []}, "top_new": ["n9 PASS +0.0031 (n=5)"],
       "calibration": {"slope": 0.1, "coverage80": 0.8, "rho": 0.4}, "errors": {"alloc": "boom"}, "llm": {"failed": 2},
       "plateau": {"active": True}, "components": {"rows": [{"component": "web", "verdict": "harmful", "effect": -0.3, "se": 0.1, "n": 40}]}}


def test_render_names_the_new_incumbent_tests_best_posterior_top_results_and_alerts():
    text = DG.render_report(REP, prev_incumbent="n1248")
    assert "CHANGED n1248 -> n9" in text and "switches in window: n9 +0.0031" in text
    assert "closed 48 (2 in window), passed 4" in text and "n3029 local mean .6826" in text
    assert "- n9 PASS +0.0031 (n=5)" in text
    alerts = DG.report_alerts(REP)
    assert any("guard slope" in a for a in alerts) and any("not computed: alloc" in a for a in alerts)
    assert any("2 LLM session" in a for a in alerts) and "plateau ACTIVE" in alerts and any("HARMFUL" in a for a in alerts)
    assert len(text.splitlines()) < 30                                       # compact
    assert "- none" in DG.render_report({"ts": 1.0, "incumbent": "x"})


def test_write_report_files_and_remembers_the_incumbent(ctx, store, tmp_path):
    p = DG.write_report(tmp_path, store, REP)
    assert p.parent == tmp_path / "digests" and p.read_text() == (tmp_path / "digest.md").read_text()
    assert store.get_state(DG.K_DIGEST_V3)["incumbent"] == "n9"
    assert "CHANGED" not in DG.write_report(tmp_path, store, {**REP, "ts": REP["ts"] + 1}).read_text()   # no change now


def test_the_ticks_own_meta_cycle_writes_the_digest(ctx, store, tmp_path):
    c = dataclasses.replace(ctx, state_dir=str(tmp_path))
    tick = TK.Tick(c, store, _Held(), TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    tick.meta.cycle(led)
    got = list((tmp_path / "digests").glob("digest-*.md"))
    assert len(got) == 1 and (tmp_path / "digest.md").exists() and "# Loop digest" in got[0].read_text()
    assert store.events("digest")[-1]["v"] == 3
    tick.close()


def test_top_new_is_labelled_best_of_window_unadjusted():
    assert "best of window, unadjusted" in DG.render_report(REP)
