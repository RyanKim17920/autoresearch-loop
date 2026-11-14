"""Bounded self-tuning and the first-principles review (docs/loop/DESIGN.md §6.1).
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
import json
import random
import time

import pytest

from loop import selftune as T
from loop.agents import LLMAnalyst
from loop.agents.analyst import FP_TRIGGER
from loop.agents.llm import FakeLLMClient
from loop.contracts import ReviewReport, Run, RunStatus, Stage
from loop.driver import K_FP, K_LAST_CONFIRM, hb_line
from loop.spec import PoolCfg, ResearcherCfg, ReviewCfg, load_context
from tests.loop.conftest import InlineExecutor
from tests.loop.test_driver import FAKE, Scripts

FP_OUT = {"summary": "the schedule never finishes",
          "assumptions": [{"lens": "ml_problem", "assumption": "the LR schedule completes", "status": "broken",
                           "evidence": "flop_fraction 0.29 at the sample stop"},
                          {"lens": "measurement", "assumption": "2 seeds resolve 0.005", "status": "unknown",
                           "evidence": "panel sigma 0.0038"}],
          "meta_hypotheses": [{"claim": "schedules are truncated", "mechanism_tag": "trunc",
                               "levers": {"effective_lr": -1}, "predicted_sign": 1,
                               "falsification": "a sample-keyed schedule does not beat root",
                               "probes": ["key the cosine schedule to samples"]}],
          "moonshots": ["reallocate FLOPs to a second-stage high-res phase"],
          "assumption_probes": ["A/A control: shift the data-order offset by one"],
          "selftune": [{"knob": "researcher.max_concurrent", "value": 40, "reason": "slots idle"}],
          "proposals": [{"change": "raise max_seeds_per_node to 7", "rationale": "power"}]}


def _driver(tmp_path, llm=None, clock=None, review=None):
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    ctx = load_context(FAKE, state_dir=tmp_path / "state")
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, researcher=ResearcherCfg(max_concurrent=0),
        review=review or ReviewCfg(stall_evals=1000)))
    driver, store = build(ctx, llm=llm or Scripts().client(), adapter=SimAdapter(ctx),
                          executor=InlineExecutor, rng=random.Random(7))
    if clock is not None:
        driver.clock = clock
    driver.sleep = lambda s: None
    return ctx, driver, store


def _review(rid: str, trigger: str = "cadence", **kw) -> ReviewReport:
    return ReviewReport(rid, "fake-loop", trigger, "s", (), (), (), {}, 0, **kw)


def test_clamps_rounds_refuses_forbidden_and_keeps_the_floor_sum(ctx, store):
    items = [{"knob": "researcher.max_concurrent", "value": 40.4, "reason": "idle slots"},
             {"knob": "promotion.p_promote", "value": 0.1, "reason": "promote more"},
             {"knob": "review.stall_evals", "value": 7.6, "reason": "faster"},
             {"knob": "promotion.alpha_confirm", "value": 0.5, "reason": "easier"},
             {"knob": "seeds.max_seeds_per_node", "value": 9, "reason": "power"},
             {"knob": "bandit.floor", "value": 0.4, "reason": "explore"}]      # 5 x 0.4 > 0.9: refused
    acts, props = T.apply(store, ctx.cfg, items, source="rv-1", now=1.0)
    over = store.get_state(T.K_OVERRIDES)
    assert over["researcher.max_concurrent"]["value"] == 24 and over["promotion.p_promote"]["value"] == 0.2
    assert over["review.stall_evals"]["value"] == 8 and isinstance(over["review.stall_evals"]["value"], int)
    assert "bandit.floor" not in over and "promotion.alpha_confirm" not in over
    assert len(acts) == 3 and all("clamped" in a for a in acts[:2])
    assert len(props) == 2 and "alpha_confirm" in props[0] and "not self-tunable" in props[1]
    ev = [e["event"] for e in store.events()]
    assert ev.count("selftune") == 3 and ev.count("selftune_refused") == 3
    eff = T.effective(ctx.cfg, store)
    assert eff.researcher.max_concurrent == 24 and eff.promotion.p_promote == 0.2
    assert ctx.cfg.researcher.max_concurrent != 24                            # the spec is untouched
    assert eff.promotion.alpha_confirm == ctx.cfg.promotion.alpha_confirm


def test_moonshot_max_inflight_is_self_tunable_within_1_to_6(ctx, store):
    assert T.BOUNDS["bandit.moonshot_max_inflight"] == (1, 6) and T.BOUNDS["bandit.moonshot_floor"] == (0.05, 0.6)
    assert ctx.cfg.bandit.moonshot_max_inflight == 2
    T.apply(store, ctx.cfg, [{"knob": "bandit.moonshot_max_inflight", "value": 9, "reason": "more bold"}],
            source="rv-1", now=1.0)
    eff = T.effective(ctx.cfg, store)
    assert eff.bandit.moonshot_max_inflight == 6 and isinstance(eff.bandit.moonshot_max_inflight, int)
    T.apply(store, ctx.cfg, [{"knob": "bandit.moonshot_max_inflight", "value": 0.2, "reason": "fewer"}],
            source="rv-2", now=2.0)
    assert T.effective(ctx.cfg, store).bandit.moonshot_max_inflight == 1
    assert "bandit.moonshot_max_inflight: 2 (bounds 1..6)" in T.describe(ctx.cfg)


def test_operator_floors_beat_selftune_and_the_floor_sum_still_holds(ctx, store):
    """The operator's moonshot minimums: self-tuning can never go below them; the other arms'
    floors shrink proportionally so the five floors still sum to <= FLOOR_SUM_MAX."""
    from loop.select.bandit import Bandit
    from loop.contracts import BANDIT_ARMS, Strategy
    spec = dataclasses.replace(ctx.cfg, bandit=dataclasses.replace(
        ctx.cfg.bandit, floor=0.1, explore_floor=0.3, moonshot_floor=0.5, moonshot_floor_min=0.4,
        moonshot_max_inflight=4, moonshot_max_inflight_min=3))
    eff = T.effective(spec, store)                                    # 0.1*3 + 0.3 + 0.5 = 1.1 > 0.9
    assert eff.bandit.moonshot_floor == 0.5 and T.floor_sum(eff) == pytest.approx(T.FLOOR_SUM_MAX)
    assert eff.bandit.floor == pytest.approx(0.1 * 2 / 3) and eff.bandit.explore_floor == pytest.approx(0.2)
    acts, _ = T.apply(store, spec, [{"knob": "bandit.moonshot_floor", "value": 0.05, "reason": "fewer"},
                                    {"knob": "bandit.moonshot_max_inflight", "value": 1, "reason": "fewer"}],
                      source="rv-1", now=1.0)
    eff = T.effective(spec, store)
    assert eff.bandit.moonshot_floor == 0.4 and eff.bandit.moonshot_max_inflight == 3, acts
    assert T.floor_sum(eff) == pytest.approx(T.FLOOR_SUM_MAX)
    store.set_state(T.K_OVERRIDES, {"bandit.moonshot_floor": {"value": 0.0}})   # legacy (no spec value): stale
    assert T.effective(spec, store).bandit.moonshot_floor == 0.5
    T.apply(store, spec, [{"knob": "bandit.floor", "value": 0.2, "reason": "x"}], source="rv-2", now=2.0)
    assert "bandit.floor" not in store.get_state(T.K_OVERRIDES)        # raising a floor past the cap: refused
    T.apply(store, spec, [{"knob": "review.stall_evals", "value": 12, "reason": "x"}], source="rv-3", now=3.0)
    assert store.get_state(T.K_OVERRIDES)["review.stall_evals"]["value"] == 12   # others still tune
    assert "OPERATOR minimum 0.4" in T.describe(spec)
    b, counts = Bandit(T.effective(spec, store).bandit, random.Random(3)), {a: 0 for a in BANDIT_ARMS}
    for _ in range(4000):
        counts[b.choose(BANDIT_ARMS)] += 1
    assert counts[Strategy.MOONSHOT] / 4000 >= 0.38, counts             # its operator floor 0.4
    assert T.operator_floors(ctx.cfg) is ctx.cfg                         # no operator minimum: untouched


def test_an_override_tuned_against_an_old_spec_value_is_stale(ctx, store):
    """rv-3 tuned moonshot_floor 0.2 -> 0.3; the operator has since set the spec to 0.5 (minimum 0.4): the
    override is stale and ignored (effective 0.5, not 0.4). One recorded against today's spec still applies;
    a legacy override without its spec value is stale on an operator-floor knob only."""
    spec = dataclasses.replace(ctx.cfg, bandit=dataclasses.replace(
        ctx.cfg.bandit, floor=0.1, explore_floor=None, moonshot_floor=0.5, moonshot_floor_min=0.4))
    o = {"reason": "r", "source": "rv-3", "ts": 0.0}
    store.set_state(T.K_OVERRIDES, {"bandit.moonshot_floor": {**o, "value": 0.3, "spec": 0.2}})
    assert T.effective(spec, store).bandit.moonshot_floor == 0.5
    assert any("STALE, ignored: spec is now 0.5" in ln for ln in T.lines(store, spec))
    assert "moonshot 0.50->0.50" in T.floors_line(spec, store)
    store.set_state(T.K_OVERRIDES, {"bandit.moonshot_floor": {**o, "value": 0.45, "spec": 0.5}})
    assert T.effective(spec, store).bandit.moonshot_floor == 0.45
    assert "moonshot 0.50->0.45" in T.floors_line(spec, store)
    ev = spec.review.stall_evals
    store.set_state(T.K_OVERRIDES, {"bandit.moonshot_floor": {"value": 0.3}, "review.stall_evals": {"value": ev + 3}})
    eff = T.effective(spec, store)
    assert eff.bandit.moonshot_floor == 0.5 and eff.review.stall_evals == ev + 3
    store.set_state(T.K_OVERRIDES, {"review.stall_evals": {**o, "value": ev + 3, "spec": ev + 1}})
    assert T.effective(spec, store).review.stall_evals == ev          # general rule: every knob


def test_overrides_take_effect_in_the_driver_and_survive_a_restart(tmp_path):
    from loop.__main__ import build, main
    ctx, driver, store = _driver(tmp_path)
    driver.tick()
    acts = driver.persist_review(_review("rv-1", selftune=json.dumps([
        {"knob": "bandit.crossover_min_p", "value": 0.9, "reason": "too few parents"},
        {"knob": "seeds.block_children", "value": 3, "reason": "fresher controls"},
        {"knob": "caps.infra_retry", "value": 9, "reason": "flaky nodes"}])), [])
    assert any("crossover_min_p" in a for a in acts)
    assert driver.cfg.bandit.crossover_min_p == 0.9 == driver.selector.cfg.bandit.crossover_min_p
    assert driver.cfg.seeds.block_children == 3 and driver._spec_cfg.seeds.block_children != 3
    (rv,) = store.reviews()
    assert rv.proposals and "caps.infra_retry" in rv.proposals[0]           # refused -> operator proposal
    assert T.open_proposals(store) == [("rv-1#0", rv.proposals[0])]
    assert "proposals: 1 awaiting operator" in hb_line(ctx, store, "loop")
    driver.tick()                                                           # the next tick keeps them
    assert driver.cfg.bandit.crossover_min_p == 0.9
    fresh = load_context(FAKE, state_dir=tmp_path / "state")               # restart: same DB, new objects
    d2, s2 = build(fresh, llm=Scripts().client(), adapter=driver.adapter, executor=InlineExecutor)
    assert fresh.cfg.seeds.block_children == 3 and d2.selector.cfg.bandit.crossover_min_p == 0.9
    s2.close()
    assert main(["ack", "--task", str(FAKE), "--state-dir", str(tmp_path / "state"), "rv-1#0"]) == 0
    assert T.open_proposals(store) == []


def test_first_principles_trigger_every_third_review_or_stall_at_most_every_12h(tmp_path):
    now = [100_000.0]
    ctx, driver, store = _driver(tmp_path, clock=lambda: now[0])
    driver.tick()
    assert not driver._fp_due()
    for i in range(3):
        store.put_review(_review(f"rv-{i}"))
    assert driver._fp_due()
    store.set_state(K_FP, {"ts": now[0], "big": 3})
    store.put_review(_review("rv-3"))
    store.put_review(_review("rv-4"))
    store.put_review(_review("rv-5"))
    assert not driver._fp_due()                                            # 3 more reviews, but < 12 h
    now[0] += 12 * 3600
    assert driver._fp_due()
    store.set_state(K_FP, {"ts": now[0], "big": 6})
    now[0] += 13 * 3600
    assert not driver._fp_due()
    driver._n_evals = lambda: 42
    store.set_state(K_LAST_CONFIRM, 12)                                    # 30 closes since a confirmation
    assert driver._fp_due()
    store.set_state(K_LAST_CONFIRM, 13)
    assert not driver._fp_due()


def test_first_principles_review_runs_stores_proposals_and_selftunes(tmp_path):
    seen = []

    def review(**call):
        seen.append(call)
        return FP_OUT
    llm = FakeLLMClient({"review": review,
                         "researcher": Scripts().researcher})
    now = [1000.0]
    ctx, driver, store = _driver(tmp_path, llm=llm, clock=lambda: now[0],
                                 review=ReviewCfg(stall_evals=1000,
                                                  fp_stall_closes=0))
    for i in range(3):
        store.put_review(_review(f"rv-{i}"))
    acts = driver.tick() + driver.tick()
    assert any("big review started (first_principles)" in a for a in acts), acts
    fp = [r for r in store.reviews() if r.trigger == FP_TRIGGER]
    assert len(fp) == 1 and store.get_state(K_FP)["big"] == 3
    rv = fp[0]
    assert rv.findings[0].startswith("[ml_problem/broken] the LR schedule completes")
    briefs = [b for _, b in rv.probe_briefs]
    assert "[moonshot] reallocate FLOPs to a second-stage high-res phase" in briefs
    assert "A/A control: shift the data-order offset by one" in briefs
    assert rv.meta_hypothesis_ids and rv.proposals == ("raise max_seeds_per_node to 7 -- power",)
    assert driver.cfg.researcher.max_concurrent == 24                      # 40 clamped to the bound
    p = seen[0]["prompt"]
    for section in ("## Throughput and slots", "## Noise: per-axis", "## Run metrics", "researcher.max_concurrent"):
        assert section in p
    assert "{" + "throughput}" not in p and "{" + "selftune}" not in p
    assert not driver._fp_due()                                            # rate-limited now
    lines = T.lines(store)
    assert "selftune researcher.max_concurrent = 24" in lines[0] and "proposals 1 awaiting operator" in lines[1]


def test_big_review_selftune_is_schema_checked(ctx, store):
    from loop.agents.analyst import REVIEW_SCHEMA
    from loop.agents.llm import validate
    assert not validate({"summary": "s", "findings": [], "meta_hypotheses": [],
                         "selftune": [{"knob": "review.stall_evals", "value": 12, "reason": "r"}]}, REVIEW_SCHEMA)
    assert validate({"summary": "s", "findings": [], "meta_hypotheses": [],
                     "selftune": [{"knob": "review.stall_evals"}]}, REVIEW_SCHEMA)


def test_preemptible_slots_knob_clamps_and_sizes_the_pool(ctx, store):
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=dataclasses.replace(
        ctx.cfg.slurm, pools=(PoolCfg("primary", slots=8), PoolCfg("idle", slots=16, preemptible=True)))))
    assert T.BOUNDS["slurm.preemptible_slots"] == (4, 56) and T.BOUNDS["researcher.max_concurrent"] == (1, 24)
    T.apply(store, ctx.cfg, [{"knob": "slurm.preemptible_slots", "value": 500, "reason": "more"},
                             {"knob": "slurm.pools", "value": 3, "reason": "x"}], source="rv", now=1.0)
    eff = T.effective(ctx.cfg, store)
    assert [p.slots for p in eff.pools] == [8, 56] and eff.capacity == 64 and ctx.cfg.capacity == 24
    T.apply(store, ctx.cfg, [{"knob": "slurm.preemptible_slots", "value": -3, "reason": "less"}], source="rv", now=2.0)
    assert [p.slots for p in T.effective(ctx.cfg, store).pools] == [8, 4]


def test_preempt_telemetry_from_fake_runs(ctx, store):
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=dataclasses.replace(
        ctx.cfg.slurm, pools=(PoolCfg("primary", slots=8), PoolCfg("idle", slots=16, preemptible=True)))))
    now = time.time()
    assert T.preempt_line(ctx, store, now) == ("preempt 24h: 0 (0%), lost 0.0 vs useful 0.0 GPU-h (metric v2); "
                                               "preemptible now 0 running, 0 pending")
    for i, (st, gpu) in enumerate([(RunStatus.DONE, 2.0), (RunStatus.RUNNING, 0.0), (RunStatus.DONE, 1.0)]):
        store.put_run(Run(f"n{i}:screen:{i}", ctx.task_id, f"n{i}", i, Stage.SCREEN, st, gpu_h=gpu,
                          submitted_ts=now - 3600, pool="idle"))
    store.put_run(Run("p:screen:1", ctx.task_id, "p", 1, Stage.SCREEN, RunStatus.DONE, submitted_ts=now - 3600,
                      pool="primary"))
    for rid, gpu in (("n0:screen:0", 0.5), ("n0:screen:0", 0.25), ("n1:screen:1", 0.25)):
        store.log("run_preempted", run_id=rid, pool="idle", gpu_h=gpu)
    s = T.preempt_stats(ctx, store, now=now + 1)
    assert s["n"] == 3 and s["launches"] == 6 and s["rate"] == 0.5 and s["lost_gpu_h"] == 1.0
    assert s["requeues_per_done"] == 1.0 and s["useful_gpu_h"] == 3.0
    assert "lost 1.0 vs useful 3.0 GPU-h" in T.preempt_line(ctx, store, now=now + 1)
    assert "shrink it when preemption waste dominates" in T.preempt_block(ctx, store, now=1.0)
    # a requeue that resumed from latest.pt kept its progress: it lost at most one save interval, not 0.5 GPU-h
    store.log("run_preempted", run_id="n2:screen:2", pool="idle", gpu_h=0.5, resumed=True)
    s = T.preempt_stats(ctx, store, now=now + 1)
    every = ctx.cfg.slurm.resume_save_every * T.RESUME_STEP_S / 3600
    assert s["lost_gpu_h"] == pytest.approx(1.0 + every) and s["requeue_gpu_h"] == pytest.approx(1.5)
    store.update_run("n0:screen:0", resumed=1)       # events from before the flag: the run's last `resumed` ones
    assert T.preempt_stats(ctx, store, now=now + 1)["lost_gpu_h"] == pytest.approx(0.75 + 2 * every)
    # v2: a run that started preemptible and finished on primary is useful work (48/68 primary finishes live)
    store.put_run(Run("m:screen:9", ctx.task_id, "m", 9, Stage.SCREEN, RunStatus.DONE, gpu_h=4.0,
                      submitted_ts=now - 3600, pool="primary"))
    store.log("run_repooled", run_id="m:screen:9", frm="idle", to="primary")
    assert T.preempt_stats(ctx, store, now=now + 1)["useful_gpu_h"] == 7.0
    store.set_state(T.K_PENDING, ["n1:screen:1"])                    # the driver's last poll: SLURM holds it
    s = T.preempt_stats(ctx, store, now=now + 1)
    assert (s["running"], s["pending"]) == (0, 1) and "0 running, 1 pending" in T.preempt_line(ctx, store, now + 1)


def test_an_override_citing_an_older_metric_definition_expires(ctx, store):
    """rv-35 / rv-36 cut slurm.preemptible_slots 32 -> 24 -> 16 on the v1 lost GPU-h (108.6 / 124.9, no resume
    credit): an override without the current PREEMPT_METRIC_V is stale and the spec value is back."""
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=dataclasses.replace(
        ctx.cfg.slurm, pools=(PoolCfg("primary", slots=8), PoolCfg("idle", slots=32, preemptible=True)))))
    store.set_state(T.K_OVERRIDES, {"slurm.preemptible_slots": {"value": 16, "spec": 32, "reason": "lost 124.9",
                                                                "source": "rv-36", "ts": 1.0}})
    assert [p.slots for p in T.effective(ctx.cfg, store).pools] == [8, 32]
    assert any("STALE, ignored: tuned on metric v1, now v2" in line for line in T.lines(store, ctx.cfg))
    T.apply(store, ctx.cfg, [{"knob": "slurm.preemptible_slots", "value": 16, "reason": "v2 waste"}], source="rv", now=2.0)
    assert store.get_state(T.K_OVERRIDES)["slurm.preemptible_slots"]["metric_v"] == T.PREEMPT_METRIC_V
    assert [p.slots for p in T.effective(ctx.cfg, store).pools] == [8, 16]


def test_an_override_stored_under_older_wider_bounds_is_clamped_on_read(ctx, store):
    """A stored 99 (beyond the bound) takes effect as 56 with no manual state edit."""
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=dataclasses.replace(
        ctx.cfg.slurm, pools=(PoolCfg("primary", slots=8), PoolCfg("idle", slots=32, preemptible=True)))))
    store.set_state(T.K_OVERRIDES, {"slurm.preemptible_slots": {"value": 99, "spec": 32, "reason": "r", "metric_v": T.PREEMPT_METRIC_V,
                                                                "source": "rv", "ts": 1.0},
                                    "review.stall_evals": {"value": 2, "spec": ctx.cfg.review.stall_evals, "reason": "r",
                                                           "source": "rv", "ts": 1.0}})
    eff = T.effective(ctx.cfg, store)
    assert [p.slots for p in eff.pools] == [8, 56] and eff.capacity == 64 and eff.review.stall_evals == 5
    assert any("slurm.preemptible_slots = 56 " in line for line in T.lines(store))
