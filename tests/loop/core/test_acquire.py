"""loop.core.acquire: one value per GPU-h, the harm limit, the fill rule."""

from __future__ import annotations

import pytest

from loop.core import acquire as A
from loop.core.acquire import Option

S2 = 0.003 ** 2


def test_kg_and_ei():
    assert A.kg(0.0, 0.0, S2, 0.0) == 0.0                                   # nothing to learn
    assert A.kg(0.0, S2, S2, 0.0) > A.kg(0.0, S2, S2, 0.01)                  # far from the bar: little value
    assert A.ei(0.01, 0.0, S2, 0.0) == pytest.approx(0.01)
    assert A.ei(0.002, S2, S2, 0.0) > A.ei(-0.002, S2, S2, 0.0) > 0


def test_obvious_best_wins():
    good = Option("test", "good", "explore", 0.006, S2, S2 / 5, 10.0, 10)
    meh = Option("test", "meh", "explore", -0.003, S2, S2 / 5, 10.0, 10)
    acts = A.rank([meh, good], free_slots=10, llm_free=0)
    assert [a.node for a in acts] == ["good"]


def test_harm_blocked_action_is_skipped():
    harm = {"rob": (-0.012, 0.002 ** 2)}                                   # P(δ_rob < -.008) ≈ .98: catastrophic
    bad = Option("test", "bad", "explore", 0.008, S2, S2 / 5, 10.0, 10, axes=harm)
    ok = Option("test", "ok", "explore", 0.004, S2, S2 / 5, 10.0, 10, axes={"rob": (0.0, 0.001 ** 2)})
    assert [a.node for a in A.rank([bad, ok], free_slots=20, llm_free=0)] == ["ok"]
    assert A.harm_ok({"rob": (0.0, 0.001 ** 2)}) and not A.harm_ok(harm)
    assert A.harm_ok({"rob": (-0.006, 0.002 ** 2)})                       # a modest loss: the panel prices it


def test_ideas_take_llm_slots_runs_take_gpu_slots():
    idea = Option("idea", None, "explore", -0.004, 0.0015 ** 2, S2 / 3, 3.5, 3, p=0.8)   # negative lane mean
    fact = Option("idea", None, "factorial", -0.006, 0.0015 ** 2, S2 / 3, 3.5, 3, p=0.8)
    zero_seed = Option("seed", "w", "explore", -0.01, 0.0, S2, 1.0, 1)                     # nothing to learn: 0
    strong = Option("test", "s", "explore", 0.006, S2, S2 / 5, 10.0, 10)
    wave = Option("wave", "c", "factorial", 0.004, S2, S2 / 16, 16.0, 16)
    acts = A.rank([idea, fact, zero_seed, strong, wave], free_slots=10, llm_free=4)
    kinds = [a.kind for a in acts]
    assert kinds[:1] == ["test"] and "seed" not in kinds and "wave" not in kinds      # 10 slots: the test only
    assert [a.lane for a in acts if a.kind == "idea"] == ["explore", "factorial", "explore", "factorial"]
    assert kinds.count("idea") == 4 == len(A.rank([idea, fact], free_slots=0, llm_free=4))  # GPUs full: ideas start
    assert len(A.rank([idea], free_slots=64, llm_free=0)) == 0                             # no LLM slot: none


def test_heavy_tailed_idea_value_from_the_upper_tail():
    v, V = 0.0015 ** 2, S2 / 3
    assert A.ei_t(-0.004, v, V, 0.0) > 3 * A.ei(-0.004, v, V, 0.0) > 0                    # the t tail dominates
    assert A.ei_t(0.01, 0.0, V, 0.0) == pytest.approx(0.01)
    assert A.ei_t(0.0, v, V, 0.0) > A.ei_t(-0.002, v, V, 0.0) > A.ei_t(-0.004, v, V, 0.0) > 0
    z, s = 0.7, 1.0                                                                       # closed form vs quadrature
    h = 1e-3
    quad = sum(max(z + t, 0) * A._t3_pdf(t) * h for t in (-60 + h * i for i in range(int(120 / h))))
    assert A.ei_t(z, s * s, 0.0, 0.0) == pytest.approx(quad, rel=2e-3)


def test_one_action_per_node_and_slots_respected():
    seed = Option("seed", "c", "explore", 0.004, S2, S2, 1.0, 1)
    test = Option("test", "c", "explore", 0.004, S2, S2 / 5, 10.0, 10)
    acts = A.rank([seed, test], free_slots=5, llm_free=0)
    assert len(acts) == 1 and acts[0].runs <= 5


def test_ledger_options_on_the_sample(sample_db):
    from loop.core import data as D
    from loop.core import status as ST
    from loop.core.backtest import fit_at
    from loop.core.seeds import reconstruct
    led = D.load(sample_db, "nanopath-v2-rescore")
    blocks = reconstruct(led)
    noise, _ = fit_at(led, {k: n.parent for k, n in led.nodes.items()})
    st = ST.derive(led, blocks, noise)
    opts = A.ledger_options(led, blocks, st, ST.chain(led, blocks, noise)[-1])
    assert {o.kind for o in opts} >= {"idea"} and all(o.v >= 0 and o.gpu_h > 0 for o in opts)
    assert all(st[o.node] is ST.Status.SCREENING for o in opts if o.node)
    A.rank(opts, free_slots=64, llm_free=8)
