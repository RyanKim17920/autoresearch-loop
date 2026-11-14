"""loop.core.status.local_reroot -- the LOCAL-INCUMBENT tier (operator 11-09 "reroot on free wins"): a TEST candidate
whose fresh pairs agree in sign and whose EB-shrunk paired gain over the incumbent (every paired seed, shrunk toward
the epoch's median gain) is >= min_gain with P(gain > 0) >= p_min re-roots; noise does not."""

from __future__ import annotations

import dataclasses
import math
from statistics import NormalDist

from loop.core import status as ST
from loop.core.data import AXES, Ledger, Node, Obs

BASE = "c" * 40
SCREEN = (1, 2, 3, 8, 2026)
FRESH = (100000, 100001, 100002, 100003, 100004)
#: the root's per-seed level (a seed effect the pairing cancels)
ROOT = {s: 0.5471 + 0.003 * math.sin(s) for s in (*SCREEN, *FRESH, 4, 5, 6, 7)}
#: the live 11-10 epoch's shape: 24 nodes paired with root on 1-5 screen seeds, true gains at the normal quantiles of
#: a prior whose fitted (robust) tau ~.003 and pooled sd_d ~.005 match the live epoch's (.0026 / .0049)
POP = [(-0.0016 + 0.0042 * NormalDist().inv_cdf((i + 0.5) / 24), (2, 3, 1, 2, 5, 2, 1, 3, 2, 4, 2, 1)[i % 12])
       for i in range(24)]
JIT = (0.006, -0.006, 0.003, -0.003, 0.0)


def _node(nid: str, parent: str | None) -> Node:
    return Node(nid, parent, None, "explore", "f", "explore", "root", 0.0, "", "", BASE, "")


def _obs(node: str, seed: int, y: float, stage: str) -> Obs:
    return Obs(f"{node}:{stage}:{seed}", node, seed, stage, "test" if stage in ("confirm", "remint") else "select",
               {a: y for a in AXES}, y, 1.0, 0.0, 0, None, BASE)


def ledger(cand: str, screen_d: dict[int, float], fresh_d: dict[int, float]) -> Ledger:
    nodes = {"root": _node("root", None), cand: _node(cand, "root")}
    obs = [_obs("root", s, y, "remint" if s in FRESH else "baseline") for s, y in ROOT.items()]
    obs += [_obs(cand, s, ROOT[s] + d, "screen") for s, d in screen_d.items()]
    obs += [_obs(cand, s, ROOT[s] + d, "confirm") for s, d in fresh_d.items()]
    for i, (g, k) in enumerate(POP):
        nid = f"p{i:02d}"
        nodes[nid] = _node(nid, "root")
        obs += [_obs(nid, s, ROOT[s] + g + JIT[(i + j) % 5], "screen") for j, s in enumerate((*SCREEN, 4, 5)[:k])]
    return Ledger(nodes, (), tuple(obs), frozenset(FRESH), ((0.0, "root"),), bases=((0.0, "root", BASE),))


#: n0012 (subject_tempered_sampling, 11-10): screens +.0014/+.0094/+.0042/+.0114/+.0039 vs root, fresh TEST
#: -.0023/+.0041/+.0106/+.0037 and a 5th pair
N0012_SCREEN = dict(zip(SCREEN, (0.0014, 0.0094, 0.0042, 0.0114, 0.0039)))
N0012_FRESH4 = dict(zip(FRESH, (-0.0023, 0.0041, 0.0106, 0.0037)))


def judge(screen, fresh, **kw):
    led = ledger("cand", screen, fresh)
    return ST.local_reroot(led, "cand", "root", FRESH, **kw)


def test_n0012_like_free_win_re_roots():
    """8 of 9 paired seeds positive, mean ~+.005: re-roots with a flat or mildly positive 5th fresh pair (gs2 alone
    would PASS only with a 5th pair >= ~+.009, else spend 3 more seeds per arm). How negative a 5th pair it survives
    depends on the epoch's prior and sd_d (live and this fixture: down to ~-.005)."""
    for d5 in (0.0010, 0.0, -0.004):
        r = judge(N0012_SCREEN, {**N0012_FRESH4, FRESH[4]: d5})
        assert r.ok, r.why
        assert r.n_fresh == 5 and r.n_pair == 10 and r.fresh_mean > 0
        assert 0.002 <= r.gain < 0.0055 and r.p >= 0.95       # shrunk below its raw ~+.005 mean, still clear
    r = judge(N0012_SCREEN, N0012_FRESH4)                      # 4 fresh pairs is enough by count
    assert r.ok and r.n_fresh == 4
    assert not judge(N0012_SCREEN, {**N0012_FRESH4, FRESH[4]: -0.012}).ok   # a bad 5th pair still stops it


def test_noise_candidate_with_mixed_signs_does_not_re_root():
    """A +.001 candidate with mixed signs (screens selected it at +.0013): P(gain > 0) far below .95."""
    r = judge({1: 0.004, 2: -0.003, 3: 0.003}, dict(zip(FRESH, (0.003, -0.002, 0.001, -0.0015, 0.0035))))
    assert not r.ok
    assert 0.0 < r.fresh_mean < 0.0015 and r.p < 0.95 and r.gain < 0.002
    assert "P(gain > 0)" in r.why and "min_gain" in r.why


def test_fresh_seeds_must_agree_in_sign_and_count():
    """A best-of-many screen at +.008 on 5 seeds whose fresh block is net negative never re-roots on the screens;
    nor does a block with only 3 fresh pairs in."""
    hot = dict(zip(SCREEN, (0.008, 0.009, 0.007, 0.010, 0.006)))
    r = judge(hot, dict(zip(FRESH, (0.002, -0.004, 0.001, -0.003, 0.0))))
    assert not r.ok and "fresh paired mean" in r.why
    r = judge(N0012_SCREEN, dict(list(N0012_FRESH4.items())[:3]))
    assert not r.ok and "3 fresh TEST pairs < 4" in r.why


def test_thresholds_are_the_configured_ones():
    fresh = {**N0012_FRESH4, FRESH[4]: 0.001}
    assert not judge(N0012_SCREEN, fresh, min_gain=0.006).ok              # min_gain binds
    assert not judge(N0012_SCREEN, fresh, p_min=0.99999).ok               # p_min binds
    assert not judge(N0012_SCREEN, fresh, min_fresh=6).ok                 # min_fresh binds
    assert judge(N0012_SCREEN, fresh, min_gain=0.0, p_min=0.9).ok


def test_unpaired_candidate_is_never_a_re_root():
    led = ledger("cand", N0012_SCREEN, N0012_FRESH4)
    r = ST.local_reroot(led, "nobody", "root", FRESH)
    assert not r.ok and r.n_fresh == 0
    assert not ST.local_reroot(led, "root", "root", FRESH).ok


def test_no_eb_prior_is_never_a_re_root():
    """Below eb_min_nodes paired nodes the paired view is flat (unshrunk): a best-of-many screen at face value."""
    r = judge(N0012_SCREEN, {**N0012_FRESH4, FRESH[4]: 0.001}, eb_min_nodes=1000)
    assert not r.ok and "no EB prior" in r.why


def test_hot_screens_cannot_carry_flat_fresh_seeds():
    """Review 11-10 MUST-FIX 1: screens +.008 and 5 fresh pairs averaging ~+.0002 (z ~.1) -- the screens alone would
    clear P .95 under a wide prior; the fresh seeds must carry min_gain at z >= 1 on their own."""
    hot = dict(zip(SCREEN, (0.008, 0.009, 0.007, 0.010, 0.006)))
    r = judge(hot, dict(zip(FRESH, (0.004, -0.003, 0.002, -0.003, 0.001))))
    assert not r.ok and "fresh" in r.why
    assert r.fresh_mean > 0                                   # sign agreement alone is not enough


def test_a_one_seed_fresh_outlier_is_a_seed_artifact():
    """Review 11-10 S6: the paired differences pass status.seed_artifact (leave-one-out, trimmed mean, outlier z) --
    one fresh pair carrying the whole gain is no free win even when the mean / z / shrunk-gain gates clear."""
    flat = dict(zip(SCREEN, (0.001, 0.0, 0.001, -0.001, 0.001)))
    r = judge(flat, dict(zip(FRESH, (0.0, 0.0005, -0.0005, 0.0, 0.040))))
    assert not r.ok and "seed artifact" in r.why


def _rec(node, vs, mean, **kw):
    return {"node": node, "vs": vs, "kind": "pass", "e": 30.0, "n": 5, "mean": mean, "reason": kw.pop("reason", "look 1"),
            "harm": [], "sd": 0.005, **kw}


def test_replay_ranks_a_gs2_pass_above_a_larger_local_one_closed_in_the_same_tick():
    """Review 11-10 S7: two passes vs the incumbent frozen in one tick -- the gs2 PASS switches even though the local
    one has the larger mean; the local one stays PASSED (re-testable)."""
    from loop.core.seeds import SeedBlock
    led = ledger("cand", N0012_SCREEN, dict(zip(FRESH, (0.004,) * 5)))
    led = dataclasses.replace(led, nodes={**led.nodes, "gs2": _node("gs2", "root")})
    blocks = [SeedBlock("test-a", "test", FRESH, ("cand", "root"), "test cand", 1.0, 10.0,
                        (_rec("cand", "root", 0.009, reason="local re-root: x", tier="local"),)),
              SeedBlock("test-b", "test", (7, 9), ("gs2", "root"), "test gs2", 2.0, 10.0,
                        (_rec("gs2", "root", 0.005),))]
    js = {j.node: j for j in ST.replay(led, blocks, None)}
    assert js["gs2"].switched and not js["cand"].switched
