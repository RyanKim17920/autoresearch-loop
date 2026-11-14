"""Fix batch B (10-12) of loop.core.factorial: the retro wave trimmed to <= 4 knobs at |z| >= 3, the two
highest-prior knobs kept off the block contrast, the far-noise se without residual dfs, relative flag knobs, and the
bank's used marks."""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from loop.core import factorial as F
from loop.core import model as M

from .test_factorial import NOISE, RHO, SIGMA, simulate


def _bk(key, m=0.001, z=0.0, source="retro"):
    return F.BankKnob(F.Knob(key, 0.1, 0.2), m, 1e-6, source, "", z)


def test_retro_trim_keeps_at_most_four_knobs_measured_at_z_3():
    bank = [_bk("a", z=3.6), _bk("b", z=2.0), _bk("c", z=4.8), _bk("d", z=3.3), _bk("e", z=3.1), _bk("f", z=3.0)]
    assert [b.knob.key for b in F.retro_trim(bank)] == ["c", "a", "d", "e"]
    assert F.retro_trim([_bk("b", z=2.9)]) == []
    assert F.BankKnob.from_json(bank[0].to_json()) == bank[0]
    assert F.BankKnob.from_json({k: v for k, v in bank[0].to_json().items() if k != "z"}).z == 0.0   # an old seed


def test_retro_bank_records_the_pooled_z_it_selected_on():
    eff = [F.RetroEffect("d1", "inc", F.Knob("a.x", 0.1, 0.2), 0.004, 0.001, 16),
           F.RetroEffect("d2", "inc", F.Knob("a.x", 0.1, 0.2), 0.003, 0.001, 16)]
    [b], _ = F.retro_bank(eff, ("inc",), {"a.x": 0.1}, tau2=2.5e-6)
    assert b.z == pytest.approx(0.0035 / math.sqrt(0.5e-6))


def test_used_knobs_are_never_offered_again():
    bank = [_bk("a"), _bk("b")]
    assert F.unused(bank, [bank[0].signature()]) == [bank[1]]
    other = F.BankKnob(F.Knob("a", 0.2, 0.4), 0.0, 1e-6, "next")     # the next range of a: a new signature
    assert F.unused([other], [bank[0].signature()]) == [other]


@pytest.mark.parametrize("k", [5, 6, 7])
def test_the_two_highest_prior_knobs_interaction_is_not_the_block(k):
    bank = [_bk(f"k{i}", m=0.001 * (i + 1)) for i in range(k)]                 # the best is last
    ks = F.arrange(bank)
    assert sorted(b.knob.key for b in ks) == sorted(b.knob.key for b in bank)
    rows, blocks = F.matrix(k)
    i1, i2 = (next(j for j, b in enumerate(ks) if b.knob.key == f"k{t}") for t in (k - 1, k - 2))
    sign = [1 if b else -1 for b in blocks]
    assert sum(r[i1] * r[i2] * s for r, s in zip(rows, sign)) == 0                 # orthogonal to the block
    assert abs(sum(r[0] * r[1] * s for r, s in zip(rows, sign))) == len(rows)       # (the old A, B: confounded)
    assert F.arrange(bank[:4]) == bank[:4]                                          # 8 points: one block


def test_without_four_residual_dfs_the_far_noise_sets_the_se():
    knobs = [F.Knob(f"k{j}", 0, 1) for j in range(4)]
    small = F.design("s", "c", knobs, [1])                                     # 8 points, 4 mains: 3 dfs
    ys = simulate(random.Random(2), small, [0.0] * 4)
    near = F.estimate(small, ys, NOISE, tau2=1.0)[0].se
    far = M.Noise({M.COMPOSITE: SIGMA ** 2}, {M.COMPOSITE: 0.30})
    got = F.estimate(small, ys, NOISE, tau2=1.0, far=far)[0].se
    assert near == pytest.approx(math.sqrt(4 * (1 - RHO) * SIGMA ** 2 / 8))
    assert got == pytest.approx(math.sqrt(4 * 0.70 * SIGMA ** 2 / 8)) and got > near
    big = F.design("b", "c", [F.Knob(f"k{j}", 0, 1) for j in range(5)], [1, 2])     # 9 dfs: its own residuals
    yb = simulate(random.Random(3), big, [0.0] * 5)
    assert F.estimate(big, yb, NOISE, tau2=1.0, far=far)[0].se == F.estimate(big, yb, NOISE, tau2=1.0)[0].se


def test_a_relative_flag_knob_moves_the_carriers_value_by_its_factor(tmp_path):
    reg = {"clip_dominated": [{"key": "dino.clip_grad", "factor": 3, "scale": "log"}]}
    assert F.flagged(["clip_dominated"], reg, {"dino.clip_grad": 3.0}) == [F.Knob("dino.clip_grad", 3.0, 9.0, "log")]
    assert F.flagged(["clip_dominated"], reg, {}) == [] == F.flagged(["clip_dominated"], reg)
    from loop.spec import load_context
    repo = Path(__file__).resolve().parents[3]
    live = load_context(repo / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path / "s",
                        require_measured=False).cfg.factorial.flag_knobs
    assert F.flagged(["clip_dominated"], live, {"dino.clip_grad": 3.0}) == [F.Knob("dino.clip_grad", 3.0, 9.0, "log")]
    assert "schedule_not_annealed" not in live                       # no cooldown / horizon key exists (YAML note)
