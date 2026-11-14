"""The LOCAL-INCUMBENT tier wired into ``Tick._decide`` (``test.local_reroot``): a TEST look gs2 leaves undecided
becomes a PASS frozen with tier "local" that switches the incumbent through the usual machinery (remint SELECT block,
replayed switch, provenance events); a "no" leaves gs2 in charge (EXTEND), a harmed axis or a look not yet in is
never judged, and the tier switched off judges nothing (on by default since 11-10)."""

from __future__ import annotations

import dataclasses

from loop import tick as TK
from loop.core import data as CD
from loop.core import status as ST

from .test_tick_core import _module_tick, _Verdicts


def _off(ctx):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, test=dataclasses.replace(ctx.cfg.test, local_reroot=False)))


def _on(ctx, **kw):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, test=dataclasses.replace(ctx.cfg.test, local_reroot=True, **kw)))


class _Look(_Verdicts):
    """A stub gs2 verdict for the opened block: ``kind`` with ``n`` paired seeds in and ``harm``."""

    def __init__(self, kind, n=5, harm=()):
        super().__init__([kind])
        self.kind_, self.n, self.harm_, self.more_ = kind, n, harm, None

    def decide(self, data, blocks, noise, incumbent):
        from loop.core.test import Kind, Verdict
        opened = sorted((b for b in blocks if b.pool == CD.TEST and b.open and b.id.startswith("test-")),
                        key=lambda b: b.id)
        return [(b, b.node_ids[0], Verdict(Kind(self.kind_), 3.0, self.n, 0.004, "look 1: c1=2.05 > Z=1.69 >= 0",
                                           tuple(self.harm_), self.more_ if self.more_ is not None else (3 if self.kind_ == "extend" else 0), 0.0055, 1.69))
                for b in opened[:1]]


def _fake(ok, calls):
    def f(led, c, i, seeds, **kw):
        calls.append((c, i, tuple(seeds), kw))
        return ST.LocalReroot(ok, 0.996 if ok else 0.61, 0.0041 if ok else 0.0003, 0.0016, 10, 5,
                              0.004 if ok else 0.0008, "stub why" if ok else "stub no")
    return f


def test_free_win_at_look_one_re_roots_without_extending(ctx, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ST, "local_reroot", _fake(True, calls))
    tick, store, t = _module_tick(_on(ctx), tmp_path, _Look("extend"))
    tick.step()
    [b] = [b for b in store.seed_blocks() if b.id.startswith("test-")]
    cand, ctrl = b.node_ids
    t[0] += 60
    rep = tick.step()
    after = {x.id: x for x in store.seed_blocks()}
    closed = after[b.id]
    assert not closed.open and closed.seeds == b.seeds                       # closed at look 1, never extended
    [rec] = closed.verdicts
    assert rec["kind"] == "pass" and rec["tier"] == "local" and rec["vs"] == ctrl and rec["gs2_kind"] == "extend"
    assert rec["local_p"] == 0.996 and rec["local_n_fresh"] == 5 and rec["reason"].startswith("local re-root")
    assert calls and calls[0][:3] == (cand, ctrl, b.seeds)
    assert calls[0][3]["p_min"] == 0.95 and calls[0][3]["min_fresh"] == 4 and calls[0][3]["min_gain"] == 0.002
    [remint] = [x for x in after.values() if x.id.startswith("remint-")]    # the usual switch: its seeds = CRN block
    assert remint.pool == CD.SELECT and remint.node_ids == (cand,) and remint.seeds == b.seeds
    [ev] = store.events("local_reroot")
    assert ev["node_id"] == cand and ev["previous"] == ctrl and ev["block"] == b.id and ev["tier"] == "local"
    assert [e.get("tier") for e in store.events("test_verdict")] == ["local"]
    assert any("LOCAL RE-ROOT" in x and cand in x for x in rep.lines)
    led = CD.load(tick.db_path, TASK_OF(tick))
    [j] = [j for j in ST.replay(led, store.seed_blocks(), None) if j.block == b.id]
    assert j.switched and j.node == cand                                     # replay: the incumbent switches
    assert ST.derive(led, store.seed_blocks(), None)[cand] is ST.Status.INCUMBENT
    tick.close()
    store.close()


def TASK_OF(tick):
    return tick.ctx.task_id


def test_no_free_win_leaves_gs2_in_charge_and_says_why_once(ctx, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ST, "local_reroot", _fake(False, calls))
    core = _Look("open")                     # "promising; waiting for an extension slot": a look, judged every tick
    tick, store, t = _module_tick(_on(ctx), tmp_path, core)
    tick.step()
    [b] = [b for b in store.seed_blocks() if b.id.startswith("test-")]
    for _ in range(2):
        t[0] += 60
        tick.step()
    after = {x.id: x for x in store.seed_blocks()}[b.id]
    assert after.open and after.verdicts is None and len(calls) == 1    # ONE decision per look (no optional stopping)
    assert len(store.events("local_reroot_no")) == 1 and not store.events("local_reroot")
    assert store.get_state(TK.Tick.K_LOCAL) == {b.id: {"5": "no"}}
    assert not [x for x in store.seed_blocks() if x.id.startswith("remint-")]
    tick.close()
    store.close()


def test_extend_after_a_no_still_extends(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "local_reroot", _fake(False, []))
    tick, store, t = _module_tick(_on(ctx), tmp_path, _Look("extend"))
    tick.step()
    [b] = [b for b in store.seed_blocks() if b.id.startswith("test-")]
    t[0] += 60
    tick.step()
    after = {x.id: x for x in store.seed_blocks()}[b.id]
    assert after.open and len(after.seeds) == 8
    tick.close()
    store.close()


def test_harm_an_unfinished_look_a_pass_or_the_tier_off_are_never_judged(ctx, tmp_path, monkeypatch):
    for core, cx in ((_Look("fail", harm=("seg",)), _on(ctx)),     # a harmed axis vetoes
                     (_Look("open", n=4), _on(ctx)),               # seeds still running: not a look
                     (_Look("pass"), _on(ctx)),                    # gs2 passed it already
                     (_Look("extend"), _off(ctx))):                # the tier switched off
        calls = []
        monkeypatch.setattr(ST, "local_reroot", _fake(True, calls))
        sub = tmp_path / f"t{len(list(tmp_path.iterdir()))}"
        sub.mkdir()
        tick, store, t = _module_tick(cx, sub, core)
        tick.step()
        t[0] += 60
        tick.step()
        assert not calls and not store.events("local_reroot")
        tick.close()
        store.close()


def test_a_look_one_fail_is_never_overridden(ctx, tmp_path, monkeypatch):
    """futility / "cannot reach c2" at look 1 is binding (priced into c2): never a local re-root."""
    calls = []
    monkeypatch.setattr(ST, "local_reroot", _fake(True, calls))
    tick, store, t = _module_tick(_on(ctx), tmp_path, _Look("fail"))
    tick.step()
    t[0] += 60
    tick.step()
    assert not calls and not store.events("local_reroot")
    tick.close()
    store.close()


def test_an_error_in_the_tier_is_a_logged_no(ctx, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(ST, "local_reroot", boom)
    tick, store, t = _module_tick(_on(ctx), tmp_path, _Look("extend"))
    tick.step()
    [b] = [b for b in store.seed_blocks() if b.id.startswith("test-")]
    t[0] += 60
    tick.step()
    assert store.events("local_reroot_error") and len({x.id: x for x in store.seed_blocks()}[b.id].seeds) == 8
    tick.close()
    store.close()


def test_a_look_two_fail_without_harm_can_still_re_root(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "local_reroot", _fake(True, []))
    tick, store, t = _module_tick(_on(ctx), tmp_path, _Look("fail", n=8))
    tick.step()
    [b] = [b for b in store.seed_blocks() if b.id.startswith("test-")]
    t[0] += 60
    tick.step()
    [rec] = {x.id: x for x in store.seed_blocks()}[b.id].verdicts
    assert rec["kind"] == "pass" and rec["tier"] == "local" and rec["gs2_kind"] == "fail"
    assert TK.incumbent_of(tick.statuses) in (b.node_ids[0], b.node_ids[-1])
    tick.close()
    store.close()


def test_an_extended_block_never_judged_at_look_one_gets_its_look_one_once(ctx, tmp_path, monkeypatch):
    """Deployed mid-test (11-10: n0012 EXTENDed by the old code at 09:11): look 1 is judged once, late, on its 5
    seeds while the 3 extension seeds still run; then not again until look 2."""
    calls = []
    monkeypatch.setattr(ST, "local_reroot", _fake(False, calls))
    tick, store, t = _module_tick(_on(ctx), tmp_path, _Look("extend"))
    tick.step()
    [b] = [b for b in store.seed_blocks() if b.id.startswith("test-")]
    store.update_seed_block(b.id, seeds=(*b.seeds, 900001, 900002, 900003))   # extended while the tier was off
    tick._blocks = None
    tick.core.kind_, tick.core.more_ = "open", 3              # gs2: "extended, running" with 3 more
    for _ in range(2):
        t[0] += 60
        tick.step()
    assert len(calls) == 1 and store.get_state(TK.Tick.K_LOCAL) == {b.id: {"5": "no"}}
    tick.close()
    store.close()


def test_an_error_anywhere_in_the_tier_is_guarded_and_gs2_still_extends(ctx, tmp_path, monkeypatch):
    """Review 11-10 S8: an exception OUTSIDE the scorer call (state read, logging, ...) never stops the tick --
    ``_decide`` guards the tier with default None, so gs2's EXTEND goes ahead."""
    def boom(self, *a, **k):
        raise RuntimeError("state store gone")
    monkeypatch.setattr(TK.Tick, "_local_reroot", boom)
    tick, store, t = _module_tick(_on(ctx), tmp_path, _Look("extend"))
    tick.step()
    [b] = [b for b in store.seed_blocks() if b.id.startswith("test-")]
    t[0] += 60
    tick.step()
    after = {x.id: x for x in store.seed_blocks()}[b.id]
    assert after.open and len(after.seeds) == 8 and after.verdicts is None
    tick.close()
    store.close()


def test_a_frozen_local_pass_stays_frozen_when_the_rule_changes(ctx, tmp_path, monkeypatch):
    """Replay is deterministic: a local PASS frozen into its block (n0012, 11-10) is never re-judged -- a later rule
    that would say "no" changes neither the frozen verdict, nor the replayed switch, nor the incumbent."""
    monkeypatch.setattr(ST, "local_reroot", _fake(True, []))
    tick, store, t = _module_tick(_on(ctx), tmp_path, _Look("extend"))
    tick.step()
    [b] = [b for b in store.seed_blocks() if b.id.startswith("test-")]
    t[0] += 60
    tick.step()
    frozen = {x.id: x for x in store.seed_blocks()}[b.id].verdicts
    calls = []
    monkeypatch.setattr(ST, "local_reroot", _fake(False, calls))      # the rule tightens after the freeze
    for _ in range(2):
        t[0] += 60
        tick.step()
    blocks = store.seed_blocks()
    assert {x.id: x for x in blocks}[b.id].verdicts == frozen and not calls
    led = CD.load(tick.db_path, TASK_OF(tick))
    [j] = [j for j in ST.replay(led, blocks, None) if j.block == b.id]
    assert j.switched and j.node == b.node_ids[0]
    assert ST.derive(led, blocks, None)[b.node_ids[0]] is ST.Status.INCUMBENT
    tick.close()
    store.close()
