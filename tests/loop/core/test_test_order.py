"""11-10 (nanobrain 16:38Z): the TEST slot went to a re-root port (paired gain -.0008 on 3 seeds) over n0103 (+.0013 on
4, P(promote) .84) on a .0001 composite-EI tie, and its unstarted block then held n0103's test off (one partial test at
a time). Tests are now ordered by the promotion bar's paired belief, and an unstarted block gives way to a clearly
stronger candidate."""
from __future__ import annotations

from types import SimpleNamespace

from loop import tick as TK
from loop.contracts import RunStatus
from loop.core import acquire as A
from loop.core import data as D

S2 = 0.003 ** 2


def _test(node: str, m: float) -> A.Option:
    return A.Option("test", node, "explore", m, 0.002 ** 2, S2 / 5, 10.0, 10)


def test_test_order_fills_the_test_places_best_paired_gain_first():
    port, n0103 = _test("port", 0.00457), _test("n0103", 0.00446)          # the live composite-EI tie
    assert [a.node for a in A.rank([port, n0103], 4, 0) if a.kind == "test"] == ["port"]
    got = A.rank([port, n0103], 4, 0, test_order={"port": -0.0008, "n0103": 0.0013})
    assert [(a.kind, a.node, a.runs) for a in got if a.kind == "test"] == [("test", "n0103", 4)]
    whole = A.rank([port, n0103], 20, 0, test_order={"port": -0.0008, "n0103": 0.0013})
    assert [a.node for a in whole if a.kind == "test"] == ["n0103", "port"]     # both fit: both open, best first
    seat = A.rank([port, n0103], 4, 0, partial_test=frozenset({"n0103"}), test_order={"port": 0.1, "n0103": 0.0})
    assert [(a.node, a.runs) for a in seat if a.kind == "test"] == [("n0103", 4)]   # the seat is the challenger's


def _core(gains: dict) -> TK.ModuleCore:
    core = TK.ModuleCore()
    core._gains = gains
    return core


def _block(bid: str, holder: str, seeds=(100008, 100009)) -> SimpleNamespace:
    return SimpleNamespace(id=bid, pool=D.TEST, open=True, seeds=tuple(seeds), node_ids=(holder, "inc"))


def test_an_unstarted_block_gives_way_only_to_a_clearly_stronger_candidate():
    data = SimpleNamespace(runs=[SimpleNamespace(seed=1), SimpleNamespace(seed=2)])
    opts = [_test("n0103", 0.0044)]
    gains = {"port": (-0.0004, 0.00304 ** 2, 3), "n0103": (0.0037, 0.00299 ** 2, 5)}
    order = {"n0103": 0.0037}
    blk = _block("b1", "port")
    assert _core(gains)._preemptible(data, [blk], opts, order) == {"b1": ("port", "n0103", False)}   # P ~ .83
    started = SimpleNamespace(runs=[SimpleNamespace(seed=100009)])            # a run on its seeds: it keeps running,
    assert _core(gains)._preemptible(started, [blk], opts, order) == {"b1": ("port", "n0103", True)}   # n0103 beside
    two = [blk, _block("b2", "other", seeds=(100010,))]
    both = SimpleNamespace(runs=[SimpleNamespace(seed=100009), SimpleNamespace(seed=100010)])
    assert _core({**gains, "other": gains["port"]})._preemptible(both, two, opts, order) == {}   # 2 started: no 3rd
    close = {**gains, "n0103": (0.0010, 0.00299 ** 2, 5)}                     # not clearly stronger
    assert _core(close)._preemptible(data, [blk], opts, {"n0103": 0.0010}) == {}
    neg = {**gains, "n0103": (-0.0001, 1e-12, 5)}                             # a challenger with no expected gain
    assert _core(neg)._preemptible(data, [_block("b1", "worse")], opts, {"n0103": -0.0001}) == {}
    unjudged = {"n0103": gains["n0103"]}                                       # the holder does not pair: keeps it
    assert _core(unjudged)._preemptible(data, [blk], opts, order) == {}
    assert _core(gains)._preemptible(data, [], opts, order) == {}             # nothing open: nothing to withdraw


def test_preempt_withdraws_the_block_only_when_the_challenger_test_was_placed():
    logged, updated = [], []
    blk = _block("b1", "port")
    core = SimpleNamespace(preempt={"b1": ("port", "n0103", False)}, _gains={"port": (-0.0004, 0, 3), "n0103": (0.0037, 0, 5)})
    me = SimpleNamespace(core=core, blocks=lambda: [blk], _blocks=[blk],
                         store=SimpleNamespace(log=lambda ev, **f: logged.append((ev, f))),
                         _update_block=lambda bid, acts, note, **f: updated.append((bid, f)) or acts.append(note))
    acts: list[str] = []
    TK.Tick._preempt_tests(me, [A.Action("screen", "x", "explore", 1.0, 1.0)], 5.0, acts)
    assert updated == [] and logged == [] and core.preempt == {}              # not placed: the block keeps its seat
    core.preempt = {"b1": ("port", "n0103", True)}                            # started: never withdrawn
    TK.Tick._preempt_tests(me, [A.Action("test", "n0103", "explore", 1.0, 1.0, 4)], 5.0, acts)
    assert updated == [] and "opens beside started block b1" in acts[-1]
    core.preempt = {"b1": ("port", "n0103", False)}
    TK.Tick._preempt_tests(me, [A.Action("test", "n0103", "explore", 1.0, 1.0, 4)], 5.0, acts)
    assert updated == [("b1", {"closed_ts": 5.0, "verdicts": ()})] and me._blocks is None
    assert logged[0][0] == "test_preempted" and logged[0][1]["by"] == "n0103" and "withdrawn unstarted" in acts[-1]


def test_a_withdrawn_block_judges_nothing_and_frees_its_candidate():
    from loop.core import seeds as S
    from loop.core import status as ST
    blk = S.SeedBlock("b1", D.TEST, (100008,), ("port", "inc"), "test port", 1.0, 2.0, verdicts=())
    assert ST.replay(SimpleNamespace(), [blk], None) == []


def test_submit_line_says_submitted_not_running():
    assert TK._submitted(SimpleNamespace(status=RunStatus.RUNNING, job_id="586674")) == "submitted (job 586674)"
    assert TK._submitted(SimpleNamespace(status=RunStatus.RUNNING, job_id="")) == "submitted"
    assert TK._submitted(SimpleNamespace(status=RunStatus.VOID, job_id="")) == "void"


def test_non_finite_paired_gains_never_order_or_preempt(monkeypatch):
    core = TK.ModuleCore()
    nan = float("nan")
    monkeypatch.setattr(core, "_paired", lambda data, inc: {"a": (nan, 1e-6, 1), "b": (0.002, nan, 1),
                                                            "c": (0.001, 1e-6, 3)})
    order = core._test_order(None, [_test("a", 0.004), _test("b", 0.003), _test("c", 0.002)], "inc")
    assert order == {"a": 0.004, "b": 0.003, "c": 0.001} and set(core._gains) == {"c"}   # unpaired: own belief
    data = SimpleNamespace(runs=[])
    core._gains = {**core._gains, "h": (-0.004, 1e-6, 3)}
    # "a" / "b" (unjudgeable) outrank the best paired challenger "c": no clear challenger, the block stays
    assert core._preemptible(data, [_block("b1", "h")], [], order) == {}
