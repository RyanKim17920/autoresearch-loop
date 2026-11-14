"""loop.core.test: the two-look group-sequential test (gs2, default) and the mixture SPRT (msprt): verdicts,
boundaries, type-I and power."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from loop.core import test as T

SD = 0.0035


@pytest.fixture
def msprt(monkeypatch):
    monkeypatch.setattr(T, "METHOD", "msprt")


def test_e_value_basics():
    assert T.e_value([], SD) == pytest.approx(1.0)
    assert T.e_value([0.004] * 3, SD) > T.e_value([0.004] * 2, SD) > 1.0
    assert T.e_value([-0.004] * 3, SD) < 1.0
    # a resumed pair (variance inflated) carries less evidence
    assert T.e_value([0.004] * 3, SD, infl=[2.0] * 3) < T.e_value([0.004] * 3, SD)


@pytest.mark.usefixtures("msprt")
def test_pass_needs_four_seeds():
    assert T.verdict([0.02, 0.02], SD).kind is T.Kind.OPEN
    v = T.verdict([0.006, 0.007, 0.006], SD)
    assert v.kind is T.Kind.OPEN and v.e >= 10                        # E >= 1/alpha at n = 3: still OPEN
    v = T.verdict([0.006, 0.007, 0.006, 0.006], SD)
    assert v.kind is T.Kind.PASS and v.n == T.MIN_PASS_N == 4


@pytest.mark.usefixtures("msprt")
def test_fail_when_unreachable_or_at_cap():
    assert T.verdict([-0.01, -0.01, -0.008], SD).kind is T.Kind.FAIL
    assert T.verdict([0.001] * 8, SD).kind is T.Kind.FAIL
    assert "cap" in T.verdict([0.001] * 8, SD).reason


@pytest.mark.usefixtures("msprt")
def test_extend_only_promising_and_with_a_slot():
    d = [0.003, 0.001, 0.002, 0.0025, 0.0005]                      # E between 1 and 10 at n = 5
    v = T.verdict(d, SD)
    assert 1 <= v.e < 10 and v.kind is T.Kind.EXTEND and v.more == 3
    assert T.verdict(d, SD, extended_open=2).kind is T.Kind.OPEN        # no extension slot: wait
    assert T.verdict(d, SD, prior=(-0.01, 1e-6)).kind is T.Kind.FAIL    # shrunk mean <= 0
    assert T.verdict([0.0, -0.002, 0.001, -0.001, 0.0], SD).kind is T.Kind.FAIL   # E < 1
    assert T.verdict([*d, 0.002], SD, planned=8).kind is T.Kind.OPEN    # extended, still running
    assert T.verdict([*d, 0.002], SD, planned=6).kind is T.Kind.FAIL    # extended once, all in, no pass


@pytest.mark.usefixtures("msprt")
def test_harm_on_a_guard_axis_blocks_the_pass():
    d = [0.006, 0.007, 0.006, 0.006, 0.007]
    rob = ([-0.006] * 5, 0.004, [1.0] * 5)
    assert T.verdict(d, SD).kind is T.Kind.PASS
    v = T.verdict(d, SD, {"rob": rob})
    assert v.harm == ("rob",) and v.kind is T.Kind.FAIL
    assert T.verdict(d[:3], SD, {"rob": (rob[0][:3], 0.004, [1.0] * 3)}).kind is T.Kind.OPEN


@pytest.mark.usefixtures("msprt")
def test_type_one_error_on_10k_null_sims():
    r = T.simulate(0.0, 0.0031, 0.38, n_sims=10_000, seed=7)
    assert r["pass"] <= T.ALPHA


@pytest.mark.usefixtures("msprt")
def test_power_rises_with_gain_and_rho():
    p = [T.simulate(g, 0.0031, 0.38, n_sims=800, seed=3)["pass"] for g in (0.002, 0.005)]
    assert p[0] < p[1] and p[1] > 0.9
    assert T.simulate(0.003, 0.0031, 0.8, n_sims=800, seed=3)["pass"] > T.simulate(0.003, 0.0031, 0.38,
                                                                                  n_sims=800, seed=3)["pass"]


@pytest.mark.usefixtures("msprt")
def test_never_extended_twice_and_in_flight_seeds_count():
    """The 10-12 sim grew blocks to 26-47 seeds: n = 5 done of 8 planned re-triggered EXTEND every tick."""
    d = [0.003, 0.001, 0.002, 0.0025, 0.0005]
    assert T.verdict(d, SD, planned=5).kind is T.Kind.EXTEND
    v = T.verdict(d, SD, planned=8)                                    # its 3 extension seeds still running
    assert v.kind is T.Kind.OPEN and v.more == 3
    assert T.verdict([*d, 0.001, 0.0015, 0.001], SD, planned=8).kind is T.Kind.FAIL
    assert T.verdict([0.006, 0.007, 0.006, 0.006], SD, planned=8).kind is T.Kind.PASS   # a pass never waits


def test_sd_used_inflates_by_the_fit_dfs():
    assert T.sd_used(SD, None) == SD and T.sd_used(SD, float("inf")) == SD
    assert T.var_inflation(4) == pytest.approx(2.41, abs=0.05)
    assert T.var_inflation(20) == pytest.approx(1.37, abs=0.02)
    assert T.var_inflation(100) == pytest.approx(1.14, abs=0.01)
    assert T.sd_used(SD, 4) > T.sd_used(SD, 20) > T.sd_used(SD, 400) > SD


@pytest.mark.usefixtures("msprt")
def test_type_one_with_an_estimated_and_misfit_sd():
    """The reviewers' failure: a near sd fitted on 4 dfs (one pair) -> type-I far above alpha. With the df inflation
    a 20-df fit stays near alpha even when the pair is 1.25x noisier than its class."""
    assert T.simulate_fit(0.0, 0.00123, 20, 1.0, n_sims=3000, seed=2)["pass"] <= T.ALPHA
    assert T.simulate_fit(0.0, 0.00123, 20, 1.25, n_sims=3000, seed=2)["pass"] <= 1.5 * T.ALPHA
    assert T.simulate_fit(0.0035, 0.00123, 20, 1.0, n_sims=800, seed=2)["pass"] > 0.8


def test_test_entry_ok():
    ok, why = T.test_entry_ok(0.002, 1e-6, 0.0035)                     # m_sel > MDE/2 = .0016
    assert ok and "MDE/2" in why
    assert not T.test_entry_ok(0.0005, 4e-6, 0.0035)[0]                # small and uncertain
    assert T.test_entry_ok(0.0005, 1e-8, 0.0035)[0]                     # small but P(d > 0) ~ 1


# -- gs2 ----------------------------------------------------------------------------------------------------------
SD_OBS, DF_OBS = 0.0033, 71        # the live pooled decision noise (10-13)


def test_gs_bounds_spend_alpha_over_two_looks():
    c1, c2 = T.gs_bounds(0.1, 5 / 8, None, "obf", 0.0)
    assert c1 == pytest.approx(1.781, abs=0.002) and c2 == pytest.approx(1.336, abs=0.002)
    assert T.spent(0.1, 5 / 8, "obf") == pytest.approx(0.0375, abs=5e-4)
    # no futility -> a higher final boundary; an estimated σ (t, 71 dfs) -> wider than normal, 20 dfs wider still
    assert T.gs_bounds(0.1, 5 / 8, None, "obf", float("-inf"))[1] > c2
    t71, t20 = T.gs_bounds(0.1, 5 / 8, 71, "obf", 0.0), T.gs_bounds(0.1, 5 / 8, 20, "obf", 0.0)
    assert c1 < t71[0] < t20[0] and c2 < t71[1] < t20[1]
    p1, p2 = T.gs_bounds(0.1, 5 / 8, None, "pocock", 0.0)
    assert p1 < c1 and p2 > c2                                         # Pocock spends more early


def test_gs2_looks_at_five_and_eight(monkeypatch):
    monkeypatch.setattr(T, "SD_PRIOR_DF", 0.0)                         # the pooled σ_d alone (the blend: below)
    sd = SD_OBS
    assert T.verdict([0.01] * 4, sd).kind is T.Kind.OPEN               # no look before K = 5 seeds
    v = T.verdict([0.004] * 5, sd)                                     # Z = 2.71 >= c1
    assert v.kind is T.Kind.PASS and v.z == pytest.approx(0.004 * 5 ** 0.5 / sd) and "look 1" in v.reason
    assert T.verdict([0.004, -0.001, -0.002, -0.001, -0.001], sd).kind is T.Kind.FAIL     # Z < 0: futility
    mid = [0.002, 0.003, 0.0, 0.002, 0.001]                            # 0 <= Z = 1.08 < c1
    v = T.verdict(mid, sd)
    assert v.kind is T.Kind.EXTEND and v.more == 3
    assert T.verdict(mid, sd, extended_open=T.MAX_EXTENDED).kind is T.Kind.OPEN        # no slot: wait
    assert T.verdict(mid, sd, planned=8).kind is T.Kind.OPEN           # its extension in flight
    assert T.verdict([*mid, 0.002], sd, planned=8).kind is T.Kind.OPEN
    v = T.verdict([*mid, 0.002, 0.003, 0.002], sd, planned=8)          # Z = 1.71 >= c2
    assert v.kind is T.Kind.PASS and "look 2" in v.reason
    v = T.verdict([*mid, 0.0, 0.001, -0.001], sd, planned=8)
    assert v.kind is T.Kind.FAIL and "look 2" in v.reason
    assert T.verdict([-0.01] * 3, sd).kind is T.Kind.FAIL              # cannot reach c2 even at +2 sd


def test_gs2_blends_the_pooled_sd_with_the_tests_own_variance():
    """The coordinator's 10-13 follow-up: a candidate noisier than the pool widens its own σ_d."""
    noisy = [0.009, -0.004, 0.008, -0.003, 0.007]                      # own sd .0062 vs the pooled .0033
    assert T.blend_sd(noisy, SD_OBS, 0.0) == SD_OBS and T.blend_sd(noisy[:1], SD_OBS, 4.0) == SD_OBS
    b = T.blend_sd(noisy, SD_OBS, 4.0)
    own2 = sum((x - 0.0034) ** 2 for x in noisy) / 4
    assert b ** 2 == pytest.approx((4 * SD_OBS ** 2 + 4 * own2) / 8)
    assert T.verdict_gs2(noisy, SD_OBS, prior_df=0.0).kind is T.Kind.PASS      # the pooled σ_d alone: Z 2.30
    v = T.verdict_gs2(noisy, SD_OBS, prior_df=4.0)
    assert v.kind is T.Kind.EXTEND and v.sd == pytest.approx(b)                 # blended: Z 1.47 < c1 (t, 8 dfs)
    c1, c2 = T.gs_bounds(0.1, 5 / 8, 4 + 7, "obf", 0.0, 4 + 4)
    assert c1 > T.gs_bounds(0.1, 5 / 8, None)[0] and c2 > T.gs_bounds(0.1, 5 / 8, None)[1]


def test_gs2_blend_type_one_at_a_misfit_pair():
    """σ_d 1.25x the pool's: type-I <= .12 with the blend (the pooled σ_d alone: ~.16); power +.003 >= .75."""
    assert T.SD_PRIOR_DF == 4.0
    assert T.simulate_fit(0.0, SD_OBS, DF_OBS, 1.25, n_sims=20_000, seed=21, method="gs2")["pass"] <= 0.12
    assert T.simulate_fit(0.003, SD_OBS, DF_OBS, 1.0, n_sims=4000, seed=22, method="gs2")["pass"] >= 0.75


def test_gs2_harm_guard_and_prior():
    d = [0.004] * 5
    rob = ([-0.006] * 5, 0.004, [1.0] * 5)
    v = T.verdict(d, SD_OBS, {"rob": rob})
    assert v.kind is T.Kind.FAIL and v.harm == ("rob",)
    mid = [0.002, 0.003, 0.0, 0.002, 0.001]
    assert T.verdict(mid, SD_OBS, prior=(-0.01, 1e-6)).kind is T.Kind.FAIL     # shrunk mean <= 0
    # a resumed pair weighs 1/f in Z
    assert T.verdict(d, SD_OBS, infl=[3.0] * 5).z < T.verdict(d, SD_OBS).z


def test_gs2_type_one_is_alpha_on_20k_null_sims():
    """S1 (10-13): the msprt as run passed .013 of nulls; gs2 at the observed pooled noise spends alpha."""
    r = T.simulate_fit(0.0, SD_OBS, DF_OBS, 1.0, n_sims=20_000, seed=11, method="gs2")
    assert abs(r["pass"] - T.ALPHA) <= 0.02
    assert T.simulate(0.0, SD_OBS, 0.0, n_sims=20_000, seed=12, method="gs2")["pass"] == pytest.approx(T.ALPHA,
                                                                                                         abs=0.02)


def test_gs2_power_at_the_observed_noise():
    p3 = T.simulate_fit(0.003, SD_OBS, DF_OBS, 1.0, n_sims=4000, seed=13, method="gs2")
    assert p3["pass"] >= 0.7
    assert T.simulate_fit(0.005, SD_OBS, DF_OBS, 1.0, n_sims=2000, seed=14, method="gs2")["pass"] >= 0.97
    # and it is far more powerful than the msprt it replaces at the same noise
    assert p3["pass"] > T.simulate_fit(0.003, SD_OBS, DF_OBS, 1.0, n_sims=2000, seed=13, method="msprt")["pass"] + 0.2


def test_sd_for_drops_the_ucb_past_ucb_df_max(monkeypatch):
    assert T.sd_for(SD, 71, "gs2") == SD
    assert T.sd_for(SD, 20, "gs2") == T.sd_used(SD, 20) > SD
    assert T.sd_for(SD, 71, "msprt") == T.sd_used(SD, 71)
    monkeypatch.setattr(T, "UCB_DF_MAX", 100.0)
    assert T.sd_for(SD, 71, "gs2") == T.sd_used(SD, 71)


def test_configure_selects_the_method(monkeypatch):
    from loop.spec import TestCfg
    for k in ("METHOD", "SPENDING", "FUTILITY_Z", "UCB_DF_MAX"):
        monkeypatch.setattr(T, k, getattr(T, k))
    T.configure(SimpleNamespace(test=TestCfg(method="msprt", spending="pocock", futility_z=-0.5, ucb_df_max=30)))
    assert (T.METHOD, T.SPENDING, T.FUTILITY_Z, T.UCB_DF_MAX) == ("msprt", "pocock", -0.5, 30.0)
    assert T.verdict([0.006, 0.007, 0.006, 0.006], SD).kind is T.Kind.PASS        # msprt passes at n = 4
    with pytest.raises(ValueError):
        T.configure(SimpleNamespace(test=TestCfg(method="bogus")))
    T.configure(SimpleNamespace())                                                  # no section: unchanged
    assert T.METHOD == "msprt"


def test_meta_type1_guard_follows_the_configured_bound(monkeypatch):
    from loop.core import meta as MT
    assert MT.guard_bounds()["type1_sim"] == (0.0, T.TYPE1_GUARD_MAX) == (0.0, 0.2)
    rep = MT.MetaReport(ts=0.0, since=0.0, epoch="e")
    rep.tests = {"type1_sim": 0.17}
    assert not rep.out_of_bounds()
    monkeypatch.setattr(T, "TYPE1_GUARD_MAX", 0.15)
    assert rep.out_of_bounds() and "type1_sim" in rep.out_of_bounds()[0]
