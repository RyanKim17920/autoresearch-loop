"""Target-aware acquisition (docs/loop-core-v3.md §8 "Target-aware acquisition"): the gap to the external target as
the bar of every value, the gap floor, sub-additive stacks with one credit per mechanism, the seed-artifact gate,
the widened prior of untested lanes / families, idle-pool churn and the meta report's window and fill bucket."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from types import SimpleNamespace

import pytest

from loop import spec as SP
from loop.core import acquire as A
from loop.core import data as D
from loop.core import ideation as ID
from loop.core import meta as MT
from loop.core import status as ST
from loop.core.acquire import Option
from loop.core.data import AXES, Obs

from .test_acquire_floors import _fill_world
from .test_epochs import _state

S2 = 0.003 ** 2
W = {a: 1 / len(AXES) for a in AXES}


def _node(nid, parent, *, lane="explore", family="f", base="B", second=None, port_of=""):
    return D.Node(nid, parent, second, "explore", family, lane, "root", 0.0, "", "", base, port_of)


def _obs(node, seed, y, base="B", gpu_h=1.0):
    return Obs(f"{node}:screen:{seed}", node, seed, "screen", "select", {a: y for a in AXES}, y, gpu_h, 0.0, 0, None,
               base)


# ---------------------------------------------------------------------------------------------- the target / gap
def test_configure_installs_the_operator_keys_and_the_target(ctx):
    cfg = dataclasses.replace(ctx.cfg, acquire=replace(ctx.cfg.acquire, fresh_shrink=0.5, idle_max_restarts=3),
                              status=SP.StatusGateCfg(max_seed_share=0.3),
                              labless=replace(ctx.cfg.labless, bar_score=0.68143, bar_node=""))
    A.configure(cfg)
    assert A.FRESH_SHRINK == 0.5 and A.IDLE_MAX_RESTARTS == 3 and ST.MAX_SEED_SHARE == 0.3
    assert A.TARGET == A.Target(0.68143, 0.0, "", 0.7) and A.target_on()     # legacy bar_score: the leader
    A.configure(dataclasses.replace(cfg, acquire=replace(cfg.acquire, target_mode="incumbent")))
    assert not A.target_on()                                       # the operator can switch it off
    A.configure(SimpleNamespace(acquire=replace(cfg.acquire, target_mode="auto"),
                                labless=SimpleNamespace(bar_score=0.68143, bar_margin=0.006, bar_node="n0655")))
    assert A.TARGET == A.Target(0.68143, 0.006, "n0655", 0.7) and A.target_on()   # bar_margin (back-compat key)
    A.configure(dataclasses.replace(cfg, labless=replace(ctx.cfg.labless, leader_median=0.6810443, bar_score=0.5,
                                                         promote_p=0.8, offset=0.001, min_test_seeds=5)))
    assert A.TARGET == A.Target(0.6810443, 0.004, "", 0.8, 0.001, 5)          # leader_median wins over bar_score
    A.configure(SimpleNamespace(labless=SimpleNamespace(bar_score=None, bar_node="")))
    assert A.TARGET is None and not A.target_on()                  # unset: the incumbent is the bar


def test_gap_is_the_target_minus_the_incumbent_on_our_infra():
    nodes = {"root": _node("root", None, lane="baseline", family="baseline"), "inc": _node("inc", "root")}
    obs = [_obs("root", s, 0.66) for s in (1, 2, 3)] + [_obs("inc", s, 0.675) for s in (1, 2, 3)]
    led = D.Ledger(nodes, (), tuple(obs), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    assert A.gap_of(led, "inc", A.Target(0.68143, 0.006)) == pytest.approx(0.68743 - 0.675)
    assert A.gap_of(led, "inc", A.Target(0.67, 0.0)) is None       # already past it: values over the incumbent
    assert A.gap_of(led, "inc") is None                            # no target configured
    # DIRECT (bar-direct): no root terms; bar_score wins over bar_node, a bare bar_node is its own mean level
    nodes2 = {**nodes, "r0": _node("r0", None, lane="baseline", family="baseline", base="A"),
              "bar": _node("bar", "r0", base="A")}
    obs2 = obs + [_obs("r0", s, 0.70, base="A") for s in (1, 2, 3)] + [_obs("bar", s, 0.72, base="A") for s in (1, 2, 3)]
    led2 = D.Ledger(nodes2, (), tuple(obs2), frozenset(), ((0.0, "root"),),
                    bases=((-1.0, "r0", "A"), (0.0, "root", "B")))
    assert A.gap_of(led2, "inc", A.Target(None, 0.006, "bar")) == pytest.approx(0.72 + 0.006 - 0.675)
    assert A.gap_of(led2, "inc", A.Target(0.68143, 0.006, "bar")) == pytest.approx(0.68743 - 0.675)


def test_gap_is_the_local_mean_needed_for_p_promote():
    """Labless rule (10-13): gap = the local mean a candidate needs for P(median of 3 >= leader + margin) >= p (pooled
    sigma, n = min_test seeds) - the incumbent level; p = .5 -> the target median itself; offset lowers it."""
    from loop.core import bar as BR
    nodes = {"root": _node("root", None, lane="baseline", family="baseline"), "inc": _node("inc", "root")}
    jit = (0.003, -0.003, 0.0)
    obs = [_obs("root", s, 0.66 + j) for s, j in zip((1, 2, 3), jit)] + \
        [_obs("inc", s, 0.675 + j) for s, j in zip((1, 2, 3), jit)]
    led = D.Ledger(nodes, (), tuple(obs), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    sig = BR.epoch_sigma(led)
    t = A.Target(0.6810443, 0.004, "", 0.7)
    need = BR.need_mean(sig, 4, 0.6850443, 0.7)
    assert sig > 0 and A.gap_of(led, "inc", t) == pytest.approx(need - 0.675) and need > 0.6850443 + 0.001
    assert A.gap_of(led, "inc", A.Target(0.6810443, 0.004, "", 0.5)) == pytest.approx(0.6850443 - 0.675)
    assert A.gap_of(led, "inc", A.Target(0.6810443, 0.004, "", 0.7, 0.002)) == pytest.approx(need - 0.002 - 0.675,
                                                                                                abs=1e-6)


def test_target_mode_values_every_kind_over_the_gap():
    t = Option("test", "c", "explore", 0.004, S2, S2 / 5, 10.0, 10)
    idea = Option("idea", None, "explore", -0.004, 0.0015 ** 2, S2 / 3, 3.5, 3)
    seed = Option("seed", "s", "explore", 0.003, S2, S2, 1.0, 1)
    assert A.target_aware([t], 0.008) == [t] and A.target_aware([t], 0.008)[0].h is None   # tests keep h = 0
    for o in (idea, seed):
        assert A.value(replace(o, h=0.008), 0.004, 0.0, 0.0) < A.value(o, 0.004, 0.0, 0.0)
        assert A.value(replace(o, h=-1.0), 0.004, 0.0, 0.0) == A.value(o, 0.004, 0.0, 0.0)   # never below the bar
    assert A.value(replace(seed, h=0.003), 0.003, 0.0, 0.0) > A.value(seed, 0.003, 0.0, 0.0)  # KG peaks AT the gap


def test_gap_floor_drops_hopeless_waves_and_ports_never_seed_extensions():
    gap = 0.01
    lo = Option("port", "p1", "crossover", 0.002, 0.001 ** 2, S2 / 3, 3.1, 3)       # .002 + 1.28 x .001 < .005
    hi = Option("port", "p2", "crossover", 0.012, 0.002 ** 2, S2 / 3, 3.1, 3)
    wave = Option("wave", "c", "factorial", 0.001, 0.001 ** 2, S2 / 8, 8.0, 8)
    seed = Option("seed", "s", "explore", 0.002, 0.001 ** 2, S2, 1.0, 1)          # ub .0033: under the wave floor
    hopeless = Option("seed", "s0", "explore", 0.001, 0.001 ** 2, S2, 1.0, 1)     # ub .0023 < seed floor .0025
    test = Option("test", "s", "explore", 0.001, 0.001 ** 2, S2 / 5, 10.0, 10)   # the entry rule judges tests
    idea = Option("idea", None, "explore", -0.004, 0.0015 ** 2, S2 / 3, 3.5, 3)
    dropped: list = []
    out = A.target_aware([lo, hi, wave, seed, hopeless, test, idea], gap, dropped)
    assert {(k, n) for k, n, _ in dropped} == {("port", "p1"), ("wave", "c"), ("seed", "s0")}
    assert all(ub < A.GAP_FLOOR_FRAC * gap for *_, ub in dropped)
    assert A.gap_floor_ub(seed) < A.GAP_FLOOR_FRAC * gap and A.SEED_FLOOR_FRAC == 0.25     # A3: the lower seed floor
    kept = [o for o in out if not o.below]                      # the hopeless wave / seed: relax pass only (10-13)
    assert [(o.kind, o.node) for o in kept] == [("port", "p2"), ("seed", "s"), ("test", "s"), ("idea", None)]
    assert [o.h for o in kept] == [gap, gap, None, gap]         # the test keeps h = 0 (exempt from floor and gap)
    assert {(o.kind, o.node) for o in out if o.below} == {("seed", "s0")}   # the gap-floored wave: dropped (10-13 w003)
    assert ("wave", "c") not in {(o.kind, o.node) for o in out}
    assert A.target_aware([lo, wave, seed, hopeless], None) == [lo, wave, seed, hopeless]   # incumbent mode
    # shrink ONCE: a posterior belief is not shrunk again (nor its sd); only a raw estimate's mean is
    mid = Option("port", "p3", "crossover", 0.0045, 0.0005 ** 2, S2 / 3, 3.1, 3)
    assert A.gap_floor_ub(mid) == pytest.approx(0.0045 + A.Z90 * 0.0005)
    assert A.gap_floor_ub(replace(mid, raw=True)) == pytest.approx(0.55 * 0.0045 + A.Z90 * 0.0005)
    assert [o.node for o in A.target_aware([mid, replace(mid, node="p4", raw=True)], gap)] == ["p3"]


def test_gap_uses_the_incumbents_test_seeds_and_target_mode_has_hysteresis():
    nodes = {"root": _node("root", None, lane="baseline", family="baseline"), "inc": _node("inc", "root")}
    sel = [_obs("inc", s, 0.68) for s in (1, 2, 3)]                 # selected on: reads high
    test = [replace(_obs("inc", s, 0.674 + 0.001 * (s % 2)), pool="test", stage="remint") for s in (101, 102, 103, 104)]
    led = D.Ledger(nodes, (), tuple(sel + test), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    g, se = A.gap_se(led, "inc", A.Target(0.68, 0.0))
    assert g == pytest.approx(0.68 - 0.6745) and 0 < se < 0.001     # the 4 TEST-pool seeds, not the selected ones
    few = D.Ledger(nodes, (), tuple(sel + test[:3]), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    assert A.gap_se(few, "inc", A.Target(0.68, 0.0))[0] == pytest.approx(0.68 - (0.68 * 3 + 0.675 * 2 + 0.674) / 6)
    assert A.target_active(0.002, 0.001, False) and not A.target_active(0.0005, 0.001, False)   # enter: gap > se
    assert A.target_active(-0.0005, 0.001, True) and not A.target_active(-0.002, 0.001, True)  # leave: gap < -se
    assert A.target_active(0.0001, 0.001, None) and not A.target_active(-0.0001, 0.001, None)
    A.TARGET_ACTIVE = False                                         # in the band, mode off: no gap
    assert A.gap_of(led, "inc", A.Target(0.6750, 0.0)) is None
    A.TARGET_ACTIVE = True
    assert A.gap_of(led, "inc", A.Target(0.6750, 0.0)) == pytest.approx(0.0005)


def test_core_persists_target_mode_and_logged_floor_drops(sample_db, monkeypatch):
    from loop import tick as TK
    led, blocks, noise = _state(sample_db)
    st, inc = ST.derive(led, blocks, noise), ST.chain(led, blocks, noise)[-1]
    post = A.Posterior(led, blocks)
    monkeypatch.setattr(A, "SEED_FLOOR_FRAC", 0.0)                 # the wave / port floor only (seeds: A3's test)
    state: dict = {}
    store = SimpleNamespace(get_state=lambda k, d=None: state.get(k, d), set_state=state.__setitem__,
                            events=lambda e=None: [])
    monkeypatch.setattr(A, "TARGET", A.Target(1.0, 0.0))               # far away: target mode
    real = A.ledger_options

    def opts(*a, dropped=None, **k):                                # two floor drops every tick
        dropped.extend([("port", "p1", 0.001), ("wave", "c1", 0.002)])
        return real(*a, dropped=dropped, **k)
    monkeypatch.setattr(A, "ledger_options", opts)
    core = TK.ModuleCore()
    core.configure(SimpleNamespace(), store=store)
    monkeypatch.setattr(A, "TARGET", A.Target(1.0, 0.0))               # (configure re-read an empty config)
    core.rank(led, blocks, st, post, 4, 0)
    assert state[core.K_TARGET] is True and any(n.startswith("target mode on") for n in core.notes)
    drops = [n for n in core.notes if n.startswith("gap_floor:")]
    assert len(drops) == 2 and state[core.K_FLOOR_SEEN] == [["port", "p1"], ["wave", "c1"]]
    again = TK.ModuleCore()                                          # a restart: nothing re-logged
    again.configure(SimpleNamespace(), store=store)
    monkeypatch.setattr(A, "TARGET", A.Target(1.0, 0.0))
    again.rank(led, blocks, st, post, 4, 0)
    assert not [n for n in again.notes if n.startswith(("gap_floor:", "target mode"))]


def test_ideation_installs_the_target_from_its_own_config(sample_db):
    led, blocks, _ = _state(sample_db)
    v = ID.view(led, blocks)
    cfg = SimpleNamespace(labless=SimpleNamespace(bar_score=0.7, bar_node="", bar_margin=0.006))
    assert A.TARGET is None
    ID.Ideation(SimpleNamespace(spec=None, cfg=cfg), store=None).view(led, v)
    assert A.TARGET == A.Target(0.7, 0.006, "", 0.7)


def test_one_mechanism_gets_one_credit():
    a = Option("port", "a", "crossover", 0.006, S2, S2 / 3, 3.1, 3, group="g")
    b = Option("port", "b", "crossover", 0.004, S2, S2 / 3, 3.1, 3, group="g")
    c = Option("port", "c", "crossover", 0.003, S2, S2 / 3, 3.1, 3, group="c")
    out = {o.node: o for o in A.target_aware([a, b, c], None)}
    assert out["a"].h is None and out["b"].h == 0.006 and out["c"].h is None
    val = {n: A.value(o, 0.0, 0.0, 0.0) for n, o in out.items()}
    assert val["b"] < val["c"] < val["a"]                          # the 2nd of a family must beat its leader


# ---------------------------------------------------------------------------------------------- stacks / mechanisms
def test_stacks_are_sub_additive_and_one_mechanism_counts_once():
    assert A.stack_mean([(0.006, "a")]) == 0.006                                     # one leg: the leg
    two = A.stack_mean([(0.006, "a"), (0.004, "b")])
    assert two == pytest.approx(max(0.74 * 0.006, 0.24 * 0.01 + 0.0013)) and two < 0.006 + 0.004
    assert A.stack_mean([(0.006, "a"), (0.004, "a")]) == 0.006     # one mechanism: its best leg, no 2nd, no discount
    assert A.stack_mean([(0.0002, "a"), (0.0002, "b")]) == pytest.approx(0.0004)      # never above the sum


def _mech_ledger():
    n = {"root": _node("root", None, lane="baseline", family="baseline"),
         "inc": _node("inc", "root", family="kde"),
         "a": _node("a", "root", family="eval_readout"), "b": _node("b", "root", family="eval_readout"),
         "c": _node("c", "inc", family="regularization"), "d": _node("d", "inc", family="data_curation"),
         "e": _node("e", "inc", family="kde"),
         "t1": _node("t1", "root", family="training"), "t2": _node("t2", "root", family="training"),
         "x": _node("x", "inc", lane="crossover", family="design", second="a")}
    return D.Ledger(n, (), (), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))


def test_legs_mechanisms_and_what_a_stack_adds():
    led = _mech_ledger()
    assert A.legs_of(led, "c") == {"c", "inc"} and A.legs_of(led, "x") == {"inc", "a"} and A.legs_of(led, "root") == set()
    assert A.same_mechanism(led, "a", "b")                          # one family
    assert A.same_mechanism(led, "c", "d") and not A.same_mechanism(led, "c", "d", ref="inc")   # only via inc
    assert A.adds_mechanism(led, "inc", "c") and not A.adds_mechanism(led, "inc", "e")          # e: inc's family
    assert not A.adds_mechanism(led, "c", "inc")                    # every leg of inc is already in c
    m = A.mechanisms(led, ["a", "b", "c", "d"], ref="inc")
    assert m["a"] == m["b"] and len({m["a"], m["c"], m["d"]}) == 3
    assert not A.same_mechanism(led, "t1", "t2")                    # a broad taxonomy family is not one mechanism
    assert {"training", "augmentation", "objective"} <= A.GENERIC_FAMILIES


def _stack_world(partner_m, parent_gain=0.004):
    """inc (gain ``parent_gain`` over root on 4 seeds) and a partner off the root with posterior own ``partner_m``."""
    n = {"root": _node("root", None, lane="baseline", family="baseline"),
         "inc": _node("inc", "root", family="kde"), "o": _node("o", "root", family="data_curation")}
    led = D.Ledger(n, (), (), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    by = {"root": [_obs("root", s, 0.65) for s in (1, 2, 3, 4)],
          "inc": [_obs("inc", s, 0.65 + parent_gain) for s in (1, 2, 3, 4)]}
    post = SimpleNamespace(est={"o": {}}, by=by, fit=lambda base, far=False: (None, W),
                           belief=lambda nid: (partner_m, S2))
    return led, post


def test_a_strong_partner_stacked_on_the_incumbent_is_worth_a_screen():
    led, post = _stack_world(0.009)
    sm = A.stack_over_parent(led, post, "inc", "o")
    sp, so = 0.55 * 0.004, 0.55 * 0.009                             # both legs on ONE scale
    assert sm == pytest.approx(max(0.74 * so, 0.24 * (sp + so) + 0.0013) - sp) and sm > 0
    old = max(0.74 * so, 0.24 * (0.004 + so) + 0.0013) - 0.004       # the unshrunk parent leg priced it below 0
    assert old < 0
    def sop(m):
        return A.stack_over_parent(*_stack_world(m), "inc", "o")
    assert sop(0.001) < 0                                           # a weak partner: the stack discount wins
    assert sop(0.009) > sop(0.006)
    led, post = _stack_world(0.009)
    post.fit = lambda base, far=False: (_ for _ in ()).throw(ValueError("no noise fit"))
    assert A.stack_over_parent(led, post, "inc", "o") is None       # no fit: no prior, never an exception


def test_the_crossover_screen_prior_is_the_stack_rule(sample_db, monkeypatch):
    led, blocks, noise = _state(sample_db)
    post = A.Posterior(led, blocks)
    inc = ST.chain(led, blocks, noise)[-1]
    cands = [n for n in post.est if n != inc and led.nodes[n].lane in ("explore", "exploit")
             and led.nodes[n].base == led.nodes[inc].base and post.belief(n)[0] > 0]
    assert cands
    for n in cands[:5]:
        sm = A.stack_over_parent(led, post, inc, n)
        assert sm is not None and sm <= A.FRESH_SHRINK * post.belief(n)[0] + 1e-12      # never more than the leg
    monkeypatch.setattr(A, "adds_mechanism", lambda *a, **k: False)                       # one mechanism
    for n in cands[:5]:                                     # one mechanism: .74 x the better leg, never more
        assert A.stack_over_parent(led, post, inc, n) <= A.FRESH_SHRINK * post.belief(n)[0] + 1e-12


# ---------------------------------------------------------------------------------------------- seed-artifact gate
def test_seed_artifact_gate_needs_leave_one_out_and_no_dominant_seed():
    par = [_obs("p", s, 0.65) for s in (1, 2, 3, 4)]
    even = [_obs("c", s, 0.65 + d) for s, d in zip((1, 2, 3, 4), (0.005, 0.006, 0.004, 0.005))]
    assert ST.seed_artifact(even, par, W) == (True, "")
    one = [_obs("c", s, 0.65 + d) for s, d in zip((1, 2, 3, 4), (0.0, 0.001, -0.001, 0.02))]
    ok, why = ST.seed_artifact(one, par, W)
    assert not ok and "leave-one-seed-out" in why                  # seed 4 alone makes the gain
    par5 = [_obs("p", s, 0.65) for s in (1, 2, 3, 4, 5)]
    big = [_obs("c", s, 0.65 + d) for s, d in zip((1, 2, 3, 4, 5), (0.001, 0.001, 0.001, 0.001, 0.008))]
    ok, why = ST.seed_artifact(big, par5, W)                       # v3 review K6: t (.008 - .001) / (.0034 x 1.118)
    assert not ok and "seed 5 is a one-seed outlier" in why and "carries 67%" in why
    big4 = [_obs("c", s, 0.65 + d) for s, d in zip((1, 2, 3, 4), (0.001, 0.001, 0.001, 0.006))]
    assert ST.seed_artifact(big4, par, W) == (True, "")             # t 1.27 at the default sd_d .0034
    assert not ST.seed_artifact(big4, par, W, sd_d=0.002)[0]        # t 2.17 at a tighter pooled sd_d
    assert ST.seed_artifact(big4, par, W, share_min_n=5, sd_d=0.002)[0]     # below share_min_n: not checked
    assert not ST.seed_artifact(even[:2], par, W)[0]                # 2 pairs: nothing to leave out


def test_seed_artifact_gate_passes_a_real_effect_at_three_seeds():
    """A true +.004 paired effect (sd .003 per seed) at n = 3 passes the gate >= 85% (the share check at n = 3 passed
    it 9.5%: it now waits for share_min_n seeds)."""
    import random
    rng, par, ok = random.Random(7), [_obs("p", s, 0.65) for s in (1, 2, 3)], 0
    for _ in range(4000):
        ch = [_obs("c", s, 0.65 + rng.gauss(0.004, 0.003)) for s in (1, 2, 3)]
        ok += ST.seed_artifact(ch, par, W)[0]
    assert ok / 4000 >= 0.85


def test_no_test_option_for_a_seed_artifact(sample_db, monkeypatch):
    led, blocks, noise = _state(sample_db)
    st, inc = ST.derive(led, blocks, noise), ST.chain(led, blocks, noise)[-1]
    post = A.Posterior(led, blocks)
    monkeypatch.setattr(ST, "seed_artifact", lambda *a, **k: (False, "seed 7 carries 90%"))
    opts = A.ledger_options(led, blocks, st, inc, post=post)
    assert not [o for o in opts if o.kind == "test"]
    assert A.entry_check(next(iter(n for n in post.est if n != inc)), post, inc, 4)[0] is False


# ---------------------------------------------------------------------------------------------- the ledger options
def test_ledger_options_in_target_mode(sample_db, monkeypatch):
    from loop.core import bar as BR
    led, blocks, noise = _state(sample_db)
    st, inc = ST.derive(led, blocks, noise), ST.chain(led, blocks, noise)[-1]
    post = A.Posterior(led, blocks)
    base = A.ledger_options(led, blocks, st, inc, post=post)
    assert all(o.h is None for o in base if not o.group)            # no target: v3 as designed
    lv = BR.level(led.in_epoch().by_node()[inc]).mean
    monkeypatch.setattr(A, "TARGET", A.Target(lv + 0.02, 0.0))
    gap = A.gap_of(led, inc)
    assert gap == pytest.approx(A.gap_se(led, inc)[0]) and gap > 0.01      # its TEST seeds when it has >= 4
    dropped: list = []
    opts = A.ledger_options(led, blocks, st, inc, post=post, dropped=dropped)
    assert opts and all(o.h is not None and o.h >= gap for o in opts if o.kind != "test")
    assert all(o.h is None for o in opts if o.kind == "test")
    assert {k for k, *_ in dropped} <= {"wave", "port", "seed"}
    assert all(A.gap_floor_ub(o) >= A.GAP_FLOOR_FRAC * gap for o in opts if o.kind in ("wave", "port")
               and not o.below)
    # seed extensions: only the lower seed floor (A3), never the wave / port one (under it: relax pass only)
    assert all(A.gap_floor_ub(o) >= A.SEED_FLOOR_FRAC * gap for o in opts if o.kind == "seed" and not o.below)
    assert {o.node for o in opts if o.kind == "seed" and not o.below} == {o.node for o in base if o.kind == "seed"
                                                          and A.gap_floor_ub(o) >= A.SEED_FLOOR_FRAC * gap}
    ideas = {o.lane: o.v for o in opts if o.kind == "idea"}
    plain = {o.lane: o.v for o in base if o.kind == "idea"}
    tested = {led.nodes[c].lane for b in blocks if b.pool == D.TEST for c in b.node_ids[:-1] if c in led.nodes}
    assert all(ideas[ln] == pytest.approx(plain[ln] * (1 if ln in tested else A.UNTESTED_VAR_MULT)) for ln in ideas)
    monkeypatch.setattr(A, "TARGET_MODE", "incumbent")
    assert all(o.h is None for o in A.ledger_options(led, blocks, st, inc, post=post) if not o.group)


def test_ideation_brief_shows_the_distance_and_the_partner_is_stack_aware(sample_db, monkeypatch):
    from loop.core import bar as BR
    led, blocks, _ = _state(sample_db)
    v = ID.view(led, blocks)
    assert ID.distance_text(v) == ""                                # target off
    lv = BR.level(led.in_epoch().by_node()[v.incumbent]).mean
    monkeypatch.setattr(A, "TARGET", A.Target(lv + 0.05, 0.0))
    text = ID.distance_text(v)
    assert text.startswith("## Distance to the target (computed)") and "OVER THE GAP" in text
    assert "EI over the gap" not in text and "mean own gain" not in text   # c4i: the rank stays, no number anchors
    gap, best, rows = ID.distance(v)
    assert gap > best and rows == sorted(rows, key=lambda r: -r.ei)
    assert all(r.v >= 0 for r in rows) and ("UNTESTED" in text) == any(r.tested == 0 for r in rows[:10])
    ide = ID.Ideation(SimpleNamespace(spec=None), store=None)
    partner = ide.stack_partner(v)
    anc, cur = set(), v.incumbent
    while cur is not None and cur not in anc:
        anc.add(cur)
        cur = v.led.nodes[cur].parent if cur in v.led.nodes else None
    eligible = [n for n, b in v.beliefs.items() if b.m > b.v ** 0.5 and b.n >= 2 and n not in anc
                and v.led.nodes[n].lane in ("explore", "exploit")]
    if partner is not None:                                         # the best STACK, not the best leg
        assert ID.Ideation.stack_value(v, partner) == max(ID.Ideation.stack_value(v, n) for n in eligible)
        monkeypatch.setattr(A, "adds_mechanism", lambda led, base, add, post=None: add != partner)
        monkeypatch.setattr(A, "stack_over_parent", lambda *a: None)
        assert ID.Ideation.stack_value(v, partner) == 0.0          # same mechanism as the incumbent: no credit


# ---------------------------------------------------------------------------------------------- idle fill / churn
def test_idle_fill_never_refills_a_churn_cancelled_seed(sample_db):
    led, blocks, st, inc, post = _fill_world(sample_db)
    acts = {a.node: a for a in A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs())}
    assert "kid2" in acts and "q1" in acts
    kid2 = {(r.node, r.seed) for r in led.runs if r.node == "kid2"}
    dead = dataclasses.replace(led, runs=tuple(dataclasses.replace(r, status="void") if r.node == "kid2" else r
                                               for r in led.runs))
    churned = kid2 | {("q1", 9)}
    again = {a.node: a for a in A.idle_fill(dead, blocks, st, inc, post, 200, set(), A.Costs(), churned=churned)}
    assert "q1" not in again                                        # a queued node the churn cancelled: not again
    if "kid2" in again:                                             # its cancelled seeds still use its budget
        assert again["kid2"].runs <= A.Costs().max_seeds - len(kid2)


def test_idle_fill_skips_long_runs(sample_db):
    led, blocks, st, inc, post = _fill_world(sample_db)
    acts = A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs())
    assert "kid2" in {a.node for a in acts}
    slow = dataclasses.replace(led, obs=tuple(dataclasses.replace(o, gpu_h=o.gpu_h * 10) if o.node == "kid2" else o
                                              for o in led.obs))
    assert "kid2" not in {a.node for a in A.idle_fill(slow, blocks, st, inc, post, 200, set(), A.Costs())}


def test_idle_fill_runs_preempted_too_often_move_to_a_free_primary_or_are_dropped(ctx, store):
    from loop.contracts import JobStatus, RunStatus, Stage
    from loop.runner import RunBook

    from ..test_driver import _pnode, _PoolAdapter, _pools_ctx
    ctx = _pools_ctx(ctx, primary=1, idle=3)
    adapter = _PoolAdapter()
    book = RunBook(ctx, store, adapter)
    n1, n2, n3 = _pnode(store, "n1"), _pnode(store, "n2"), _pnode(store, "n3")
    book.fill_only = {("n1", 1), ("n2", 1)}
    for n in (n1, n2, n3):
        book.launch(n, Stage.SCREEN, 1)
    assert {e["run_id"] for e in store.events("run_fill")} == {"n1:screen:1", "n2:screen:1"}
    assert RunBook(ctx, store, adapter).fill_runs() == {"n1:screen:1", "n2:screen:1"}   # survives a restart
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", restarts=1, requeued=1)
    book.poll()
    assert book.idle_churn() == []                                 # once is bad luck
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", restarts=2, requeued=1)
    book.poll()
    acts = book.idle_churn()
    assert adapter.moved == [("n1:screen:1", "primary")] and len(acts) == 2
    assert store.get_run("n2:screen:1").status is RunStatus.VOID    # primary full: dropped
    assert store.get_run("n3:screen:1").status is RunStatus.RUNNING  # not a fill run: the slurm repool rules
    assert book.idle_churn() == []


def test_idle_churn_never_moves_a_fill_run_onto_primary_while_high_runs_wait(ctx, store):
    from loop.contracts import JobStatus, RunStatus, Stage
    from loop.runner import RunBook

    from ..test_driver import _pnode, _PoolAdapter, _pools_ctx
    ctx = _pools_ctx(ctx, primary=1, idle=3)
    adapter = _PoolAdapter()
    book = RunBook(ctx, store, adapter)
    n1 = _pnode(store, "n1")
    book.fill_only = {("n1", 1)}
    book.launch(n1, Stage.SCREEN, 1)
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", restarts=2, requeued=1)
    book.poll()
    book.high_demand = 1                                           # a test's runs wait for the primary slot
    acts = book.idle_churn()
    assert adapter.moved == [] and len(acts) == 1 and "dropped" in acts[0]
    assert store.get_run("n1:screen:1").status is RunStatus.VOID
    ev = store.events("run_cancelled")
    assert ev and ev[-1]["reason"].startswith("idle fill preempted") and "HIGH" in ev[-1]["reason"]


def test_idle_churn_keeps_a_resumable_fill_run_far_through_training(ctx, store):
    """10-13: 16 of 18 dropped fill runs were ~98% done with latest.pt armed (a requeue resumes them): never drop a
    resumable run >= acquire.idle_keep_frac through training; one not armed, or early, is still dropped."""
    from loop.contracts import JobStatus, RunStatus, Stage
    from loop.runner import RunBook

    from ..test_driver import _pnode, _PoolAdapter, _pools_ctx
    ctx = _pools_ctx(ctx, primary=1, idle=4)
    adapter = _PoolAdapter()
    state = {"far:screen:1": {"armed": True, "resumed": 2, "fraction": 0.98},
             "early:screen:1": {"armed": True, "resumed": 1, "fraction": 0.2},
             "cold:screen:1": {"armed": False, "resumed": 0, "fraction": 0.9}}
    adapter.resume_state = lambda h: state[h.run_id]
    book = RunBook(ctx, store, adapter)
    nodes = [_pnode(store, n) for n in ("busy", "far", "early", "cold")]
    book.launch(nodes[0], Stage.CONFIRM, 1)                         # holds the only primary slot
    book.fill_only = {(n, 1) for n in ("far", "early", "cold")}
    for n in nodes[1:]:
        book.launch(n, Stage.SCREEN, 1)
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", restarts=2, requeued=1)
    book.poll()
    acts = book.idle_churn()
    assert sorted(a.split()[3] for a in acts) == ["cold:screen:1", "early:screen:1"]
    assert store.get_run("far:screen:1").status is RunStatus.RUNNING
    assert [e["run_id"] for e in store.events("idle_churn_kept")] == ["far:screen:1"]
    assert book.idle_churn() == [] and len(store.events("idle_churn_kept")) == 1     # logged once per restarts


# ---------------------------------------------------------------------------------------------- the meta report
def test_meta_fill_bucket_and_the_window_starts_at_the_current_epoch(ctx, tmp_path, monkeypatch):
    from loop.store import SqliteStore

    from ..test_tick import _ledger
    from .conftest import TASK
    runs = (D.Run("f:screen:1", "n", 1, "screen", "done", 2.0, 0.0, 5.0, "", 0),
            D.Run("s:screen:1", "n", 2, "screen", "done", 1.0, 0.0, 5.0, "", 0))
    led = D.Ledger({"n": _node("n", "root")}, runs, (), frozenset(), ((0.0, "root"),))
    g = MT.gpu_by_purpose(led, 0.0, fill={"f:screen:1"})
    assert g["fill"] == [2.0, 2.0] and "fill" in MT.PURPOSES
    db = _ledger(tmp_path / "ledger.db")
    store = SqliteStore(db, TASK)
    led = D.load(db, TASK)
    seen = []
    real = MT.report
    monkeypatch.setattr(MT, "report", lambda *a, **k: (seen.append(k["since"]), real(*a, **k))[1])
    meta = MT.Meta(dataclasses.replace(ctx, task_id=TASK, state_dir=str(tmp_path)), store, clock=lambda: 2e9,
                   locks=None)
    meta.cycle(led)
    assert seen[0] == led.bases[-1][0] > 0                          # no previous report: this epoch, not all history
    meta.cycle(led)
    assert seen[1] == 2e9                                           # then the previous report
    store.close()


def test_idle_fill_seed_floor_orders_never_drops(sample_db, monkeypatch):
    """A3 (10-13) revised by the pre-deploy review: the seed floor ORDERS idle fill (b), it never drops a node --
    idle GPU slots are free information. v3-c5 A2 reverses it by default (``acquire.fill_drop_below``: drop; the
    new rule: tests/loop/test_c5_fill_slots.py); this is the rule with the key off."""
    monkeypatch.setattr(A, "FILL_DROP_BELOW", False)
    led, blocks, st, inc, post = _fill_world(sample_db)
    from loop.core import model as Mo
    monkeypatch.setattr(Mo, "NEAR_MIN_DF", 0)                            # no far calibration: (b) gets the slots
    seeded = {a.node for a in A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs())
              if a.why.startswith(("idle (b)", "idle (d)"))}
    assert seeded
    monkeypatch.setattr(A, "target_on", lambda: True)
    monkeypatch.setattr(A, "gap_of", lambda *a, **k: 1.0)                 # a gap no node can reach
    acts = A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs())
    assert {a.node for a in acts if a.why.startswith(("idle (b)", "idle (d)"))} == seeded     # all still seeded (order only)
    monkeypatch.setattr(A, "SEED_FLOOR_FRAC", 0.0)                         # 0 = off
    assert {a.node for a in A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs())
            if a.why.startswith(("idle (b)", "idle (d)"))} == seeded
