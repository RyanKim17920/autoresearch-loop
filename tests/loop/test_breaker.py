"""loop.breaker (11-08): the LLM-spend circuit breaker -- K sessions in a row producing nothing, or N sessions on a
fresh history without a passing smoke, stop new research sessions until a redeploy or an operator clear."""

from __future__ import annotations

from types import SimpleNamespace

from loop import runner as R
from loop.breaker import CLEAR_FILE, K_BREAKER, Breaker


class Store:
    def __init__(self, nodes=()):
        self.st, self.events, self._nodes = {}, [], list(nodes)

    def get_state(self, k):
        return self.st.get(k)

    def set_state(self, k, v):
        self.st[k] = v

    def log(self, event, **f):
        self.events.append((event, f))

    def nodes(self):
        return self._nodes


def _node(i, status):
    return SimpleNamespace(id=i, status=SimpleNamespace(value=status))


def _ctx(tmp_path, **rc):
    return SimpleNamespace(state_dir=str(tmp_path), cfg=SimpleNamespace(researcher=SimpleNamespace(**rc)))


FAIL = SimpleNamespace(status="failed", error="empty_diff: the agent changed nothing")


def _nothing(b, n, acts):
    return [b.record(FAIL, "research req-v3-1 produced nothing (failed)", acts) for _ in range(n)]


def test_k_consecutive_sessions_producing_nothing_trip_it_and_name_the_reason(tmp_path):
    st = Store([_node("root", "BASELINE"), _node("n0001", "QUEUED")])        # a working history
    rev = ["abc"]
    b = Breaker(_ctx(tmp_path, breaker_consecutive=6, breaker_fresh_sessions=30), st, code_rev=lambda: rev[0])
    acts: list[str] = []
    assert not any(_nothing(b, 5, acts)) and not b.tripped()
    assert b.record(SimpleNamespace(status="ok", error=""), "queued n0002 (x/y) from req-v3-2", acts) == ""
    assert st.st[K_BREAKER]["consecutive"] == 0                             # a node resets the streak
    assert not any(_nothing(b, 5, acts))
    assert b.record(FAIL, "research req-v3-3 produced nothing (failed)", acts)
    assert "6 consecutive" in b.tripped() and "failed: empty_diff x6" in b.tripped()
    ev = [f for e, f in st.events if e == "ALERT_research_breaker"]
    assert len(ev) == 1 and ev[0]["dominant"] == "failed: empty_diff"
    assert any(a.startswith("ALERT research breaker tripped") for a in acts)
    # held across a plain restart (same revision), released by a redeploy (another revision)
    b2 = Breaker(_ctx(tmp_path, breaker_consecutive=6, breaker_fresh_sessions=30), st, code_rev=lambda: "abc")
    assert b2.check(acts) and b2.tripped()
    b3 = Breaker(_ctx(tmp_path, breaker_consecutive=6, breaker_fresh_sessions=30), st, code_rev=lambda: "def")
    assert b3.check(acts) == "" and not b3.tripped()
    assert any(e == "research_breaker_cleared" for e, _ in st.events)


def test_outages_and_reissues_count_nothing(tmp_path):
    st = Store([_node("n0001", "QUEUED")])
    b = Breaker(_ctx(tmp_path, breaker_consecutive=2, breaker_fresh_sessions=0), st)
    acts: list[str] = []
    for _ in range(5):
        b.record(SimpleNamespace(status="backend_unavailable", error=""), "research x produced nothing", acts)
        b.record(SimpleNamespace(status="restart", error=""), "", acts)
        b.record(FAIL, "research req-v3-9: smoke got no slot in time; re-issued (1/2)", acts)
    assert not b.tripped()


def test_a_fresh_history_without_a_passing_smoke_trips_and_the_operator_clears_it(tmp_path):
    st = Store([_node("root", "BASELINE"), _node("n0001", "REJECTED")])     # every node rejected: fresh
    ctx = _ctx(tmp_path, breaker_consecutive=6, breaker_fresh_sessions=4)
    b = Breaker(ctx, st, code_rev=lambda: "abc")
    acts: list[str] = []
    rej = SimpleNamespace(status="ok", error="")
    for i in range(3):                                                      # nodes, all rejected at smoke
        assert not b.record(rej, f"rejected n000{i + 2} from req-v3-{i}: smoke_failed", acts)
    assert b.record(rej, "rejected n0009 from req-v3-9: smoke_failed", acts)
    assert "fresh history" in b.tripped() and "rejected: smoke_failed x4" in b.tripped()
    (tmp_path / CLEAR_FILE).write_text("")
    assert b.check(acts) == "" and not (tmp_path / CLEAR_FILE).exists() and not b.tripped()


def test_a_history_with_an_admitted_node_has_no_fresh_rule(tmp_path):
    """nanopath parity: a working ledger (admitted nodes) only ever meets the consecutive rule; rejected nodes never
    trip it."""
    st = Store([_node("root", "BASELINE"), _node("n0001", "SCREENING"), _node("n0002", "REJECTED")])
    b = Breaker(_ctx(tmp_path, breaker_consecutive=6, breaker_fresh_sessions=4), st)
    acts: list[str] = []
    for i in range(50):
        b.record(SimpleNamespace(status="ok", error=""), f"rejected n{i} from r{i}: below_noise", acts)
    assert not b.tripped() and acts == []


def test_zero_disables_and_a_tripped_breaker_holds_research_capacity(tmp_path):
    st = Store()
    b = Breaker(_ctx(tmp_path, breaker_consecutive=0, breaker_fresh_sessions=0), st)
    acts: list[str] = []
    assert not any(_nothing(b, 100, acts))
    ctx = SimpleNamespace(cfg=SimpleNamespace(researcher=SimpleNamespace(max_concurrent=4), ideation=None))
    res = R.Research(ctx, Store(), None, object(), None)
    res.smoke_capacity = lambda: ("open", "")
    assert res.capacity() == 4
    res.breaker = "6 consecutive research sessions produced nothing"
    assert res.capacity() == 0 and res.free() == 0


# ---------------------------------------------------------------------------------------------- review fixes (11-08)
def test_the_revision_is_read_at_process_start_not_at_trip_time(tmp_path):
    revs = iter(["old", "new"])
    b = Breaker(_ctx(tmp_path, breaker_consecutive=2, breaker_fresh_sessions=0), Store(), code_rev=lambda: next(revs))
    _nothing(b, 2, [])                                  # trips AFTER a redeploy checked out "new"
    assert b.store.get_state(K_BREAKER)["rev"] == "old"
    assert b.check([]) == b.tripped()                   # same process: still held until a restart on "new"


def test_a_node_after_a_trip_releases_the_breaker(tmp_path):
    st = Store()
    b = Breaker(_ctx(tmp_path, breaker_consecutive=2, breaker_fresh_sessions=0), st)
    acts: list[str] = []
    _nothing(b, 2, acts)
    assert b.tripped()
    b.record(SimpleNamespace(status="ok", error=""), "rejected n0007 from root: smoke_failed", acts)
    assert b.tripped() == "" and any(e == "research_breaker_cleared" for e, _ in st.events)
    assert any("cleared" in a for a in acts)


def test_an_intake_that_raises_counts_as_a_failed_session(tmp_path):
    from loop import tick as TK
    recorded = []
    t = object.__new__(TK.Tick)
    t.store, t.phase, t.selfcheck_hold = Store(), "intake", ""
    t.research = SimpleNamespace(breaker="", reissue_infra=None, reissue_starved=None)
    t.breaker = SimpleNamespace(record=lambda res, out, extra: recorded.append((res, out)), tripped=lambda: "",
                                store=t.store)

    def boom(req, res):
        raise RuntimeError("db locked")
    t.intake = boom
    out = t._intake(SimpleNamespace(id="req-1"), SimpleNamespace(status="ok", error="", candidate=None))
    assert out.startswith("ERROR intake") and recorded and recorded[0][0].status == "failed"
    assert "intake_error" in recorded[0][0].error


def test_the_smoke_nan_scan_sees_infinity():
    from loop.adapters.nanobrain import nonfinite
    assert nonfinite("loss: Infinity") and nonfinite("grad: -inf") and nonfinite("loss: NaN")
    assert not nonfinite("vol/s: nan (nan)  tflop/s: Infinity (inf)  loss: 0.53") and not nonfinite("information inferred")
