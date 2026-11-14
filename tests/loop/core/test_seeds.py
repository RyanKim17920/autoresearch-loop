"""loop.core.seeds: pools, never-used draws, enforcement, historical reconstruction, the additive table."""

from __future__ import annotations

import pytest

from loop.core import data as D
from loop.core import seeds as S
from loop.core.seeds import SeedBlock, SeedPoolViolation

from .conftest import TASK


def blk(id_, pool, seeds, nodes, ts, closed=None, purpose="test"):
    return SeedBlock(id_, pool, tuple(seeds), tuple(nodes), purpose, ts, closed)


def test_draw_test_never_used():
    blocks = [blk("a", D.TEST, [100000, 100001], ["c", "i"], 1.0)]
    got = S.draw_test(3, used=[100002, 11, 12], blocks=blocks)
    assert got == (100003, 100004, 100005)
    assert S.draw(2, used=[], fresh_base=500) == (500, 501)


def test_check_submit_raises_outside_block():
    blocks = [blk("t", D.TEST, [100000, 100001], ["c", "inc"], 1.0)]
    S.check_submit("c", 100000, blocks)
    S.check_submit("inc", 100001, blocks)
    S.check_submit("anyone", 11, blocks)                     # a seed in no block: SELECT
    with pytest.raises(SeedPoolViolation):
        S.check_submit("other", 100000, blocks)


def test_pass_releases_seeds_to_select():
    """On a PASS the test seeds are re-issued as the new incumbent's SELECT (CRN) block: children may use them."""
    blocks = [blk("t", D.TEST, [100000], ["c", "inc"], 1.0, closed=2.0),
              blk("r", D.SELECT, [100000], ["c"], 3.0, purpose="remint c")]
    S.check_submit("child_of_c", 100000, blocks)
    with pytest.raises(SeedPoolViolation):
        S.check_submit("child_of_c", 100000, blocks, ts=2.5)      # as of before the re-issue
    assert S.pool_of(100000, blocks) == D.SELECT and S.pool_of(100000, blocks, 2.5) == D.TEST


def test_screening_hides_open_test_blocks():
    obs = [D.Obs(f"r{s}", "c", s, "confirm", D.TEST, {a: 0.0 for a in D.AXES}, 0.0, 1.0, 0.0, 0) for s in (1, 2)]
    blocks = [blk("open", D.TEST, [1], ["c", "i"], 0.0), blk("closed", D.TEST, [2], ["c", "i"], 0.0, closed=1.0)]
    assert [o.seed for o in S.screening_obs(obs, blocks)] == [2]


def test_reconstruct_sample(sample_db):
    led = D.load(sample_db, TASK)
    blocks = S.reconstruct(led)
    assert {s for b in blocks if b.pool == D.TEST for s in b.seeds} == set(led.test_seeds)
    assert S.violations(led, blocks) == []
    legs = [b for b in blocks if b.purpose.startswith("legs")]
    assert legs and all(b.pool == D.SELECT for b in legs)
    audit = {b.purpose: b for b in blocks}
    if "audit n0655" in audit:
        assert audit["audit n0655"].node_ids == ("n0655", D.ROOT)


def test_store_table_and_migrate(tmp_path, sample_db):
    from loop.store import SqliteStore
    led = D.load(sample_db, TASK)
    store = SqliteStore(tmp_path / "s.db", TASK)
    try:
        n = S.migrate(store, led)
        assert n == len(S.reconstruct(led)) and S.migrate(store, led) == 0     # idempotent
        got = store.seed_blocks()
        assert [b.id for b in got] == [b.id for b in S.reconstruct(led)]
        store.put_seed_block(blk("new", D.TEST, [100990], ["c", "i"], 9e9))
        store.update_seed_block("new", seeds=(100990, 100991), closed_ts=9e9 + 1)
        assert store.seed_blocks()[-1] == blk("new", D.TEST, [100990, 100991], ["c", "i"], 9e9, 9e9 + 1)
    finally:
        store.close()
