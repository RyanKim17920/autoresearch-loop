"""loop.core.data: the read-only ledger view, pools, lanes, carriers."""

from __future__ import annotations

import hashlib
import json

import pytest

from loop.core import data as D

from .conftest import TASK


def test_lane_of():
    assert D.lane_of("design", "legs=L2 seed1=100005 design ...") == "factorial"
    assert D.lane_of("moonshot", "tune=t9 tune_of=n0100") == "factorial"
    assert D.lane_of("crossover", "xover=n1,n2") == "crossover"
    assert D.lane_of("exploit", "") == "exploit"
    assert D.lane_of("moonshot", "request=r brief=moonshot") == "explore"
    assert D.lane_of("revisit", "brief=failure") == "explore"
    assert D.lane_of("baseline", "") == "baseline"


def test_nearest_carrier():
    parents = {"a": "root", "b": "a", "c": "b", "root": None}
    assert D.nearest_carrier("c", parents, {"a"}) == "a"
    assert D.nearest_carrier("c", parents, {"b", "a"}) == "b"
    assert D.nearest_carrier("b", parents, set()) == "root"
    assert D.nearest_carrier("a", parents, {"a"}) == "root"          # a node is never its own carrier


def test_test_seeds():
    drawn = [("remint n0655", [100000, 100001]), ("legs L2", [100004, 100005]), ("confirm n0655", [100012]),
             ("audit n1706", [100048])]
    stages = [("audit", 201), ("audit", 202), ("screen", 11), ("confirm", 101), ("screen", 100004),
              ("parent", 100012), ("remint", 990000)]
    got = D.test_seeds(drawn, stages)
    assert got == {100000, 100001, 100012, 100048, 201, 202, 990000}
    assert 100004 not in got and 101 not in got            # factorial draws and the shared confirm pool: SELECT


def test_load_counts_pools_and_carriers(sample_db, sample):
    before = hashlib.sha256(sample_db.read_bytes()).hexdigest()
    led = D.load(sample_db, TASK)
    assert hashlib.sha256(sample_db.read_bytes()).hexdigest() == before        # read-only
    assert len(led.nodes) == len(sample["nodes"]) and len(led.runs) == len(sample["runs"])
    done = [r for r in sample["runs"] if r["status"] == "done"]
    assert len(led.obs) == len(done)
    o = {x.run_id: x for x in led.obs}
    assert o["n0011:confirm:101"].pool == D.SELECT                  # the shared confirm pool
    assert o["n1706:confirm:100036"].pool == D.TEST                 # a never-used confirm draw
    assert all(x.pool == D.TEST for x in led.obs if x.stage in ("audit", "remint"))
    assert all(x.pool == D.SELECT for x in led.obs if x.stage in ("screen", "promote"))
    # the panel is the official weighted sum of the six axes
    for x in led.obs:
        assert x.panel == pytest.approx(sum(w * x.axes[a] for a, w in D.OFFICIAL_WEIGHTS.items()), abs=1e-9)
    # carriers: an audited node's later children build on it; before its audit pass they build on root
    assert led.nodes["n1706"].carrier == "n0655"
    assert led.nodes["n0255"].carrier == "root"
    assert [c for _, c in led.carriers][:3] == ["root", "n0165", "n0255"]
    assert led.carrier_at("n1706", 0.0) == "root"


def test_before_is_a_time_cut(sample_db):
    led = D.load(sample_db, TASK)
    ts = sorted(x.ts for x in led.obs)[len(led.obs) // 2]
    cut = led.before(ts)
    assert len(cut.obs) < len(led.obs) and all(x.ts < ts for x in cut.obs)
    assert cut.nodes is led.nodes and cut.runs is led.runs


def test_task_filter(sample_db):
    assert D.load(sample_db, "some-other-task").runs == ()


def test_val_loss_q25(tmp_path):
    rows = [{"step": s, "total": 1.0} for s in range(1, 401)]
    rows += [{"step": 100, "val_total": 3.0}, {"step": 200, "val_total": 2.0}, {"step": 400, "val_total": 1.0}]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert D.val_loss_q25(tmp_path) == 3.0                          # step 100 = 25% of 400
    assert D.val_loss_q25(tmp_path / "missing") is None
