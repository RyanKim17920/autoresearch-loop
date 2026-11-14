"""loop.core.status: derive() is pure and idempotent; the chain switches only on a PASS against the incumbent."""

from __future__ import annotations

import random

from loop.core import data as D
from loop.core import status as ST
from loop.core.model import Noise
from loop.core.seeds import SeedBlock

from .conftest import TASK

SIG = 0.003
NOISE = Noise({t: SIG ** 2 for t in (*D.AXES, D.PANEL)}, {t: 0.4 for t in (*D.AXES, D.PANEL)})


def node(nid, parent=D.ROOT, status="screening"):
    return D.Node(nid, parent, None, "explore", "f", "explore", D.ROOT, 0.0, status, "")


def ledger(mu, plan, seed=0):
    """``plan``: (node, seed, stage, run status, ts) rows; done runs get y = mu + b[seed] + e."""
    rng = random.Random(seed)
    b: dict[int, float] = {}
    runs, obs = [], []
    for nid, s, stage, status, ts in plan:
        rid = f"{nid}:{stage}:{s}"
        runs.append(D.Run(rid, nid, s, stage, status, 1.0, ts, ts + 1, "", 0))
        if status == "done":
            y = mu[nid] + b.setdefault(s, rng.gauss(0, SIG * 0.4 ** 0.5)) + rng.gauss(0, SIG * 0.6 ** 0.5)
            obs.append(D.Obs(rid, nid, s, stage, D.SELECT, {a: y for a in D.AXES}, y, 1.0, ts + 1, 0))
    nodes = {n: node(n, None if n == D.ROOT else (D.ROOT if n != "c3" else "c1"),
                     "rejected" if n == "rej" else "screening") for n in mu}
    return D.Ledger(nodes, tuple(runs), tuple(obs), frozenset(), ((0.0, D.ROOT),))


def scenario():
    mu = {D.ROOT: 0.0, "c1": 0.012, "c2": 0.011, "c3": 0.0, "s": 0.0, "crash": 0.0, "rej": 0.0, "q": 0.0}
    t1, t2, t3 = (100000, 100001, 100002, 100003, 100004), (100005, 100006, 100007, 100008, 100009), (100010,)
    plan = [(D.ROOT, 11, "baseline", "done", 0.0), ("s", 11, "screen", "done", 1.0),
            ("crash", 11, "screen", "crash", 1.0), ("c3", 11, "screen", "done", 1.0)]
    plan += [(n, s, st, "done", 10.0) for s in t1 for n, st in (("c1", "test"), (D.ROOT, "test"))]
    plan += [(n, s, st, "done", 20.0) for s in t2 for n, st in (("c2", "test"), (D.ROOT, "test"))]
    plan += [("c3", s, "test", "done", 30.0) for s in t3]
    led = ledger(mu, plan)
    blocks = [SeedBlock("b1", D.TEST, t1, ("c1", D.ROOT), "test c1", 5.0, 15.0),
              SeedBlock("b2", D.TEST, t2, ("c2", D.ROOT), "test c2", 15.0, 25.0),
              SeedBlock("b3", D.TEST, t3, ("c3", "c1"), "test c3", 25.0, None)]
    return led, blocks


def test_chain_and_statuses():
    led, blocks = scenario()
    assert ST.chain(led, blocks, NOISE) == [D.ROOT, "c1"]
    st = ST.derive(led, blocks, NOISE)
    assert st["c1"] is ST.Status.INCUMBENT
    assert st["c2"] is ST.Status.PASSED          # passed vs root, shares no fresh seed with c1: no switch
    assert st[D.ROOT] is ST.Status.PASSED        # a former incumbent
    assert st["c3"] is ST.Status.TESTING
    assert st["s"] is ST.Status.SCREENING and st["crash"] is ST.Status.CRASHED
    assert st["rej"] is ST.Status.VOID and st["q"] is ST.Status.QUEUED


def test_derive_is_pure_and_idempotent():
    led, blocks = scenario()
    assert ST.derive(led, blocks, NOISE) == ST.derive(led, blocks, NOISE)
    assert blocks[2].closed_ts is None and len(led.obs) == len(scenario()[0].obs)


def test_a_pass_against_a_stale_incumbent_does_not_switch():
    led, blocks = scenario()
    judged = {j.node: j for j in ST.replay(led, blocks, NOISE)}
    assert judged["c1"].switched and judged["c2"].vs == D.ROOT and not judged["c2"].switched


def test_decide_open():
    led, blocks = scenario()
    [(b, c, v, _)] = ST.decide_open(led, blocks, NOISE, "c1")
    assert (b.id, c, v.n) == ("b3", "c3", 0)           # c1 has not run on the block's seed yet


def test_chain_on_the_ledger_sample(sample_db):
    from loop.core.backtest import fit_at
    from loop.core.seeds import reconstruct
    led = D.load(sample_db, TASK)
    noise, _ = fit_at(led, {k: n.parent for k, n in led.nodes.items()})
    blocks = reconstruct(led)
    chain = ST.chain(led, blocks, noise)
    assert chain[0] == D.ROOT and len(set(chain)) == len(chain)
    st = ST.derive(led, blocks, noise)
    assert set(st) == set(led.nodes) and st[chain[-1]] is ST.Status.INCUMBENT


def _frozen(block, recs):
    from dataclasses import replace
    return replace(block, verdicts=tuple(recs))


def _rec(c, vs, kind, mean, e=20.0, n=5):
    from loop.core import test as T
    return ST.record(c, vs, T.Verdict(T.Kind(kind), e, n, mean, "stub", (), 0, 0.002))


def test_frozen_verdicts_are_never_rejudged_with_refitted_noise():
    led, blocks = scenario()
    fz = ST.freeze(led, blocks, NOISE)
    assert set(fz) == {"b1", "b2"} and all(r["legacy"] for rs in fz.values() for r in rs)
    frozen = [_frozen(b, fz[b.id]) if b.id in fz else b for b in blocks]
    huge = Noise({t: (10 * SIG) ** 2 for t in (*D.AXES, D.PANEL)}, {t: 0.0 for t in (*D.AXES, D.PANEL)})
    assert ST.chain(led, blocks, huge) == [D.ROOT]                      # re-judged with a refit: c1 no longer passes
    assert ST.chain(led, frozen, huge) == ST.chain(led, frozen, None) == ST.chain(led, blocks, NOISE) == [D.ROOT, "c1"]
    assert ST.derive(led, frozen, None) == ST.derive(led, blocks, NOISE)
    assert ST.freeze(led, frozen, NOISE) == {}                          # frozen once


def test_a_frozen_live_verdict_ignores_later_shared_seeds():
    """A live block's verdict is its own seeds' at close: a FAIL stays a FAIL however many fresh seeds c later shares
    with the incumbent (the legacy replay pooled them into one e-process)."""
    led, blocks = scenario()
    b1 = _frozen(blocks[0], [_rec("c1", D.ROOT, "fail", 0.0005, e=0.8)])
    st = ST.derive(led, [b1, *blocks[1:]], NOISE)
    assert st["c1"] is ST.Status.FAILED and st["c2"] is ST.Status.INCUMBENT


def test_simultaneous_passes_switch_to_the_best_and_the_others_are_retestable():
    led, blocks = scenario()
    from dataclasses import replace
    b1 = _frozen(replace(blocks[0], closed_ts=30.0), [_rec("c1", D.ROOT, "pass", 0.010)])
    b2 = _frozen(replace(blocks[1], closed_ts=30.0), [_rec("c2", D.ROOT, "pass", 0.012)])
    judged = ST.replay(led, [b1, b2, blocks[2]], None)
    assert [(j.node, j.switched) for j in judged] == [("c1", False), ("c2", True)]
    assert ST.chain(led, [b1, b2, blocks[2]], None) == [D.ROOT, "c2"]
    st = ST.derive(led, [b1, b2, blocks[2]], None)
    assert st["c2"] is ST.Status.INCUMBENT and st["c1"] is ST.Status.PASSED
    assert ST.retest_candidates(led, [b1, b2, blocks[2]], None) == ["c1"]


def test_a_pass_against_a_former_incumbent_is_retestable_once_and_a_retest_closes_it():
    led, blocks = scenario()
    b1 = _frozen(blocks[0], [_rec("c1", D.ROOT, "pass", 0.010)])
    b2 = _frozen(blocks[1], [_rec("c2", D.ROOT, "pass", 0.012)])      # closed later, vs the former incumbent
    assert ST.chain(led, [b1, b2], None) == [D.ROOT, "c1"]
    assert ST.retest_candidates(led, [b1, b2], None) == ["c2"]
    retest = SeedBlock("b4", D.TEST, (100020,), ("c2", "c1"), "test c2", 40.0, 50.0,
                       (_rec("c2", "c1", "pass", 0.002, n=5),))
    assert ST.chain(led, [b1, b2, retest], None) == [D.ROOT, "c1", "c2"]
    assert ST.retest_candidates(led, [b1, b2, retest], None) == []
    open_retest = SeedBlock("b5", D.TEST, (100021,), ("c2", "c1"), "test c2", 40.0, None)
    assert ST.retest_candidates(led, [b1, b2, open_retest], None) == []   # testing: not offered again


def test_decide_open_never_extends_a_block_twice():
    from loop.core import test as T
    led, blocks = scenario()
    big = SeedBlock("b9", D.TEST, tuple(range(100030, 100038)), ("c3", "c1"), "test c3", 26.0, None)
    [(_, _, v, _)] = [x for x in ST.decide_open(led, [*blocks[:2], big], NOISE, "c1") if x[0].id == "b9"]
    assert v.kind is T.Kind.OPEN and v.more == 8                    # 8 planned, none in: waits, never EXTEND


def test_pair_fit_pools_every_clean_pair_by_default(monkeypatch):
    """S1 (10-13): σ_d from every clean pair of the epoch (near and far together), not the change class."""
    from loop.core import model as M
    led, _ = scenario()
    seen = []

    def fit(x, pairs):
        seen.append(sorted(pairs))
        return Noise({D.PANEL: SIG ** 2}, {D.PANEL: 0.4}, {D.PANEL: 30.0}, {D.PANEL: 2.0})
    np_, fp_ = M.pairs_by_change(led.nodes, {"c1"})
    pf = ST.pair_fit(led, "c3", set(), {"c1"}, True, fit)
    assert ST.SD_POOL == "pooled" and pf.label.startswith("pooled") and not pf.near
    assert seen[0] == sorted([*np_, *fp_])
    monkeypatch.setattr(ST, "SD_POOL", "class")
    seen.clear()
    pf = ST.pair_fit(led, "c3", set(), {"c1"}, True, fit)
    assert pf.label.startswith("near") and pf.near and seen[0] == sorted(np_)
