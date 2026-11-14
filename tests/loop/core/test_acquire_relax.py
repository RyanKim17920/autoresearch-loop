"""Pre-deploy review (10-13) MUST 1: GPU slots never sit idle while useful work exists. In a shadow-like state (a
gap no seed can reach, waves under a floor scaled off the idea rates, no QUEUED node) ``rank``'s relax pass still
launches the below-floor positive-value seeds / waves (preemptible only); the LLM session cap scales up while many
GPU slots are free."""

from __future__ import annotations

from types import SimpleNamespace

from loop.core import acquire as A
from loop.core.acquire import Option

S2 = 0.003 ** 2


def _shadow_like(gap: float = 0.0062):
    seeds = [Option("seed", f"n{i}", "explore", 0.0002 + 0.00001 * i, 0.0005 ** 2, S2, 1.0, 1) for i in range(44)]
    wave = Option("wave", "c", "factorial", 0.0004, 0.0006 ** 2, S2 / 16, 8.0, 8)
    idea = Option("idea", None, "explore", -0.002, 0.004 ** 2, S2 / 3, 3.5, 3, 0.8)
    dropped: list = []
    opts = A.target_aware([*seeds, wave, idea], gap, dropped)
    return opts, dropped


def test_all_options_below_the_floors_still_fill_free_slots(monkeypatch):
    """The pre-c5 rule (``acquire.fill_drop_below`` off): below-seed-floor seeds take every idle slot. v3-c5 drops
    them by default (tests/loop/test_c5_fill_slots.py)."""
    monkeypatch.setattr(A, "FILL_DROP_BELOW", False)
    opts, dropped = _shadow_like()
    assert {k for k, *_ in dropped} == {"seed", "wave"}                     # every GPU option is under a floor
    assert all(o.below for o in opts if o.kind != "idea")
    acts = A.rank(opts, free_slots=40, llm_free=2)
    gpu = [a for a in acts if a.kind != "idea"]
    assert gpu and sum(a.runs for a in gpu) == 40                          # every free slot is taken
    assert all(a.pre for a in gpu)                                         # ... preemptible only
    assert len({a.node for a in gpu}) == len(gpu)                          # one action per node
    rates = [a.rate for a in gpu]
    assert rates == sorted(rates, reverse=True)                            # best value / GPU-h first
    assert sum(a.kind == "idea" for a in acts) == 2


def test_relax_respects_hard_caps_and_can_be_switched_off(monkeypatch):
    opts, _ = _shadow_like()
    assert not [a for a in A.rank(opts, 40, 0, caps={"seed": 0, "wave": 0}) if a.kind != "idea"]
    monkeypatch.setattr(A, "RELAX_FILL", False)
    assert not [a for a in A.rank(opts, 40, 0) if a.kind != "idea"]       # the old behaviour: idle slots


def test_relax_never_preempts_the_ranked_selection(monkeypatch):
    monkeypatch.setattr(A, "FILL_DROP_BELOW", False)
    scr = Option("screen", "s", "explore", 0.004, 0.002 ** 2, S2 / 3, 3.0, 3)
    opts, _ = _shadow_like()
    acts = A.rank([scr, *opts], free_slots=5, llm_free=0)
    assert acts[0].kind == "screen" and not acts[0].pre                    # above the floors: normal pass, first
    assert sum(a.runs for a in acts) == 5 and all(a.pre for a in acts[1:])


def test_llm_sessions_scale_with_free_gpu_slots():
    from loop.runner import Research
    cfg = SimpleNamespace(researcher=SimpleNamespace(max_concurrent=4),
                          ideation=SimpleNamespace(llm_slots_max=8, llm_scale_free_slots=12))
    rs = Research.__new__(Research)
    rs.ctx, rs.gpu_free = SimpleNamespace(cfg=cfg), 0
    assert rs.limit() == 4
    rs.gpu_free = 12
    assert rs.limit() == 4                                                 # strictly more than the threshold
    rs.gpu_free = 30
    assert rs.limit() == 8
    cfg.researcher.max_concurrent = 10
    assert rs.limit() == 10                                                # never below the configured cap


def test_relax_waves_capped_at_relax_max_waves_and_open_waves_capped():
    """10-13: three relax waves (w002..w004) held 24 slots. The relax pass takes at most ``RELAX_MAX_WAVES`` (v3-c5:
    0, was 1) waves minus the open relax waves (``relax_caps``); ``budgets`` caps open waves at ``MAX_OPEN_WAVES``."""
    scr = Option("screen", "s", "explore", 0.004, 0.002 ** 2, S2 / 3, 3.0, 3)
    weak = [Option("wave", f"c{i}", "factorial", -0.004, 0.0005 ** 2, S2 / 16, 8.0, 8) for i in range(3)]
    assert not [a for a in A.rank([scr, *weak], free_slots=64, llm_free=0) if a.kind == "wave"]   # default 0
    acts = A.rank([scr, *weak], 64, 0, relax_caps={"wave": 1})
    assert [a.node for a in acts if a.kind == "wave"] == ["c0"] and all(a.pre for a in acts if a.kind == "wave")
    assert not [a for a in A.rank([scr, *weak], 64, 0, relax_caps={"wave": 0}) if a.kind == "wave"]   # one open
    assert len([a for a in A.rank([scr, *weak], 64, 0, relax_caps={"wave": 2}) if a.kind == "wave"]) == 2
    assert A.RELAX_MAX_WAVES == 0 and A.MAX_OPEN_WAVES == 3
    A.configure(SimpleNamespace(acquire=SimpleNamespace(
        target_mode="auto", fresh_shrink=0.55, gap_floor_frac=0.5, stack_best_leg_frac=0.74, stack_sum_frac=0.24,
        stack_intercept=0.0013, untested_var_mult=2.0, idle_max_restarts=2, idle_max_run_frac=1.5,
        relax_max_waves=2, max_open_waves=5)))
    try:
        assert A.RELAX_MAX_WAVES == 2 and A.MAX_OPEN_WAVES == 5
    finally:
        A.RELAX_MAX_WAVES, A.MAX_OPEN_WAVES = 0, 3


def test_budgets_cap_open_waves():
    from loop.core import data as D
    from loop.core import factorial as F
    nodes = {"root": D.Node("root", None, None, "baseline", "baseline", "baseline", "root", 0.0, "", "", "B"),
             **{f"w00{i}.p00": D.Node(f"w00{i}.p00", "root", None, "factorial", "design", "factorial", "root", 0.0,
                                      "", f"wave=w00{i} point=0 block=0 seed=7", "B") for i in range(1, 3)}}
    led = D.Ledger(nodes, (), (), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    assert len(F.open_waves(led)) == 2
    caps = A.budgets(led, {}, 10, SimpleNamespace(idea_runs=3))["caps"]
    assert caps["wave"] == A.MAX_OPEN_WAVES - 2 == 1


def test_a_gap_floored_wave_gets_no_relax_wave_that_tick():
    """w003 (10-13): the gap floor dropped carrier c this tick, yet the relax pass gave it a wave."""
    opts, dropped = _shadow_like()
    assert ("wave", "c") in {(k, n) for k, n, _ in dropped}
    assert not [a for a in A.rank(opts, 64, 0) if a.kind == "wave"]
