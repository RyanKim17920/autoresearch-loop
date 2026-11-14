"""loop.core.backtest: rank statistics and the time-ordered replay on the ledger fixture."""

from __future__ import annotations

import json

import pytest

from loop.__main__ import main as loop_main
from loop.core import backtest as B
from loop.core.data import HELD_OUT_STAGES, load

from .conftest import TASK


def test_ranks_spearman_slope():
    assert B.ranks([3.0, 1.0, 2.0, 2.0]) == [3.0, 0.0, 1.5, 1.5]
    assert B.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert B.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert B.slope([0, 1, 2], [1, 3, 5]) == pytest.approx(2.0)


def test_replay_uses_only_data_before_the_first_held_out_run(sample_db):
    led = load(sample_db, TASK)
    pts = B.replay(led)
    held = {r.node for r in led.runs if r.stage in HELD_OUT_STAGES and r.node != "root"}
    assert pts and {p.node for p in pts} <= held
    assert [p.ts for p in pts] == sorted(p.ts for p in pts)
    for p in pts:
        sel = {o.seed for o in led.obs if o.node == p.node and o.ts < p.ts}
        assert p.n_sel == len(sel) and p.n_fresh >= 1
        assert all(o.stage not in HELD_OUT_STAGES for o in led.obs if o.node == p.node and o.ts < p.ts)
        assert p.v > 0 and p.fresh_var > 0


def test_report_shape(sample_db):
    rep = B.report(B.replay(load(sample_db, TASK)))
    for key in ("a_composite", "a_panel"):
        assert {"n", "rho", "slope", "coverage80", "bias", "gate"} <= set(rep[key])
        assert -1.0 <= rep[key]["rho"] <= 1.0
    assert {"rho_raw_panel", "rho_raw_composite", "rho_prior_only"} <= set(rep["b"])
    assert B.report([]) == {"n": 0}


def test_cli_entry(sample_db, tmp_path, capsys):
    out = tmp_path / "points.jsonl"
    assert loop_main(["backtest", "--db", str(sample_db), "--task-id", TASK, "--points", str(out)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["a_composite"]["n"] == len(out.read_text().splitlines())


def test_checks_c_and_d(sample_db, capsys, monkeypatch):
    led = load(sample_db, TASK)
    noise, _ = B.fit_at(led, B._parents(led))
    c = B.check_c(led, noise)
    assert [r["node"] for r in c["rows"]] == [n for _, n, _ in led.audits if n in led.nodes]
    assert all(r["vs_parent"] in ("pass", "fail", "open", "extend") for r in c["rows"])
    monkeypatch.setattr(B, "simulations", lambda: [])
    assert loop_main(["backtest", "--db", str(sample_db), "--task-id", TASK, "--decide"]) == 0
    d = json.loads(capsys.readouterr().out)["d"]
    assert d["derived"][0] == "root" and [s["node"] for s in d["switches"]] == d["derived"][1:]
