"""Ideation quality metrics and the brief-kind meta-bandit (loop/quality.py, loop/select/bandit.py).

Run: uv run pytest tests/loop -q --confcutdir=tests/loop
"""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from loop.quality import ideation_lessons, prediction_stats, quality_table
from loop.select.bandit import K_KIND_PULLS


def test_prediction_bias_sign_accuracy_and_correlation():
    pairs = [(0.005, 0.001), (0.004, 0.002), (0.006, -0.001), (0.002, 0.0005), (0.008, 0.003)]
    s = prediction_stats(pairs)
    assert s["n"] == 5 and s["sign_acc"] == pytest.approx(0.8)
    assert s["bias"] == pytest.approx(0.005 - 0.0011)
    assert s["optimism"] == pytest.approx(0.005 / 0.0011)            # "4.5x too optimistic"
    assert -1 <= s["pearson"] <= 1 and s["spearman"] == pytest.approx(0.3)
    assert "pearson" not in prediction_stats(pairs[:4])               # only once n >= 5
    assert math.isnan(prediction_stats([(0.01, -0.002)])["optimism"])  # measured mean <= 0
    assert prediction_stats([]) == {"n": 0}


def test_brief_kind_bandit_shifts_toward_a_planted_better_kind_with_a_floor(ctx, store):
    from loop.select import PortfolioSelector
    sel = PortfolioSelector(ctx, None, None, rng=random.Random(3))
    truth = {"literature": 0.1, "failure": 0.6}                       # credit per GPU-h
    noise, picks = random.Random(4), []
    for i in range(300):
        k = sel.pick_kind(store, ("literature", "failure"))
        picks.append(k)
        store.set_state(K_KIND_PULLS, store.get_state(K_KIND_PULLS, []) + [
            {"arm": k, "node_id": f"n{i}", "ts": float(i), "reward": truth[k] + noise.gauss(0, 0.1)}])
    late = Counter(picks[150:])
    assert late["failure"] / 150 > 0.75                               # learned the better kind
    assert late["literature"] / 150 >= 0.5 * ctx.cfg.bandit.floor     # ... without starving the other


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    from tests.loop.test_driver import Scripts, _make, _world
    ctx = _world(tmp_path_factory.mktemp("q"))
    driver, store, _ = _make(ctx, Scripts())
    for _ in range(60):
        driver.tick()
    return store, driver.world.fit(store)


def test_quality_table_and_lessons_on_the_sim(sim, capsys):
    store, snap = sim
    table, lessons = quality_table(store, snap), ideation_lessons(store, snap)
    with capsys.disabled():
        print("\n" + table + "\n" + lessons)
    assert " by brief" in table and " by arm" in table and " by backend" in table
    assert "literature" in table and "brief bandit credit/GPU-h" in table
    assert "your predictions: sign right" in lessons and "helped" in lessons
    assert len(lessons) < 2400                                         # ~600 tokens
    assert store.get_state(K_KIND_PULLS)                               # closed nodes credited their kind


def test_a_knob_exposure_carrier_is_not_counted_as_a_rejection(store):
    from loop.contracts import CARRIER, Node, NodeStatus, Strategy
    from loop.quality import _groups
    store.ensure_root("c0")
    for nid, why, req in (("n1", CARRIER, "expose-t1-1"), ("n2", "expose_carrier", "expose-t2-1"),
                          ("n3", "exact_dup", "req-3")):
        store.put_node(Node(id=nid, task_id=store.task_id, commit="c", parent_id="root", strategy=Strategy.DESIGN,
                            operator=None, family="f", mechanism_tag="m", diff_hash=nid,
                            status=NodeStatus.REJECTED, reject_reason=why, notes=f"request={req}"))
        store.log("research_started", request_id=req)
    nodes, failed = _groups(store)
    assert [n.id for n in nodes] == ["n3"] and not failed       # carriers (new and old name) are bookkeeping
