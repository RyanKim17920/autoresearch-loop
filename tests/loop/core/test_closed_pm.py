"""10-22: a COMPUTED postmortem (no LLM) for every closed node: typed outcome, per-axis deltas, engagement, tuning
points; labelled computed in the briefs and the history tool."""

from __future__ import annotations

from types import SimpleNamespace

from loop.agents import briefs as BR
from loop.contracts import NodeStatus
from loop.core import closed_pm as CPM
from loop.core import ideation as ID

from .test_c8b_bold import benv  # noqa: F401  (the fixture)
from .test_fnf import env  # noqa: F401  (the fixture)

AX = {"cls": (0.02, 0.005, 4), "seg": (-0.03, 0.006, 4), "slide": (0.0, 0.005, 4)}


def kw(**o):
    base = dict(m=-0.004, sd=0.004, engaged=None, axes={}, tuned=0, structural=False)
    return {**base, **o}


def test_classify_types():
    c = lambda status="screened_out", **o: CPM.classify(status, **kw(**o))[0]   # noqa: E731
    assert c(m=-0.02, sd=0.004, engaged="engaged") == "harmful_engaged"
    assert c(m=-0.02, sd=0.004) == "harmful"
    assert c(engaged="did not engage") == "no_engage" and c(engaged="reversed") == "no_engage"
    kind, trade = CPM.classify("screened_out", **kw(m=0.001, axes=AX))
    assert kind == "axis_tradeoff" and "cls" in trade and "seg" in trade
    assert c(m=-0.001, structural=True, engaged="engaged") == "under_tuned"
    assert c(m=-0.001, structural=True, tuned=3, engaged="engaged") == "noise"
    assert c(m=0.001) == "noise" and c(m=0.02, sd=0.004) == "positive_unconfirmed"
    assert c("crashed") == "crashed" and c("void") == "void" and c("confirmed") == "confirmed"
    assert c("test_fail") == "test_fail" and c(m=None) == "unmeasured"


def test_write_one_row_per_closed_node_labelled_computed_and_idempotent(benv):
    ctx, store, led, ide = benv
    v = ide.view(led)
    closed = {n.id for n in store.nodes() if n.status in CPM.CLOSED and n.id in led.nodes}
    assert closed, "the fixture ledger has closed nodes"
    before = {pm.node_id for pm in store.postmortems()}
    acts = CPM.write(ctx, store, v, ide._engagements(led), 123.0, limit=500)
    rows = [pm for pm in store.postmortems() if CPM.is_computed(pm)]
    assert acts and {pm.node_id for pm in rows} == closed - before
    for pm in rows:
        assert pm.explanation.startswith("[") and "Tuning points tried" in pm.explanation
        assert pm.verdict in ("loss", "null", "crash", "void", "win") and pm.suggested_next
    assert CPM.write(ctx, store, v, ide._engagements(led), 124.0, limit=500) == []     # nothing is written twice
    n = len(store.postmortems())
    assert len(store.postmortems()) == n


def test_briefs_and_history_show_computed_rows_as_facts(benv):
    ctx, store, led, ide = benv
    v = ide.view(led)
    CPM.write(ctx, store, v, ide._engagements(led), 1.0, limit=500)
    pms = [pm for pm in store.postmortems() if CPM.is_computed(pm)]
    assert pms
    txt = BR.postmortem_block(pms)
    assert BR.COMPUTED_NOTE in txt and BR.PROSE_NOTE not in txt and pms[-1].node_id in txt
    old = SimpleNamespace(node_id="nX", verdict="null", explanation="a story", mechanism_reasoning="because",
                          suggested_next="x")
    mixed = BR.postmortem_block([*pms, old])
    assert BR.PROSE_NOTE in mixed and BR.COMPUTED_NOTE in mixed
    from loop.agents.history import Historian
    h = Historian.__new__(Historian)
    h.store = store
    out = h._postmortem(pms[0].node_id)
    assert BR.COMPUTED_NOTE in out and BR.PROSE_NOTE not in out


def test_ideate_writes_the_computed_postmortems_each_tick(env):
    from ..conftest import InlineExecutor
    from .test_fnf import _Research
    ctx, store, led, ide = env
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)
    tick = SimpleNamespace(research=_Research(0), pool=InlineExecutor(), dry=False)
    n0 = len(store.postmortems())
    acts = ide.ideate(tick, [], led, {})
    rows = [pm for pm in store.postmortems() if CPM.is_computed(pm)]
    assert len(store.postmortems()) > n0 and rows and any(a.startswith("computed postmortems") for a in acts)
    assert all(store.get_node(pm.node_id).status in CPM.CLOSED for pm in rows)


def test_a_structural_node_with_too_few_tuned_points_is_under_tuned_before_any_causal_type():
    """S9: harmful_engaged / test_fail hints told the next session 'do not retry' a node that never got its ladder."""
    c = lambda status="screened_out", **o: CPM.classify(status, **kw(**o))[0]   # noqa: E731
    assert c(m=-0.02, sd=0.004, engaged="engaged", structural=True, tuned=1) == "under_tuned"
    assert c("test_fail", m=0.004, sd=0.004, structural=True, tuned=2) == "under_tuned"
    assert c("test_fail", m=0.004, sd=0.004, structural=True, tuned=3) == "test_fail"          # laddered: the old types
    assert c(m=-0.02, sd=0.004, engaged="engaged", structural=True, tuned=3) == "harmful_engaged"
    assert c("crashed", structural=True) == "crashed" and c("confirmed", structural=True, tuned=0) == "confirmed"
    for k in ("harmful_engaged", "no_engage", "test_fail", "under_tuned"):                     # causal hints say 'possible'
        assert "possibl" in CPM.HINTS[k]
    assert "do not retry" not in CPM.HINTS["harmful_engaged"]


def test_a_retro_port_is_not_a_tuned_point_of_the_old_node():
    mk = lambda nid, notes: SimpleNamespace(notes=notes)                                       # noqa: E731
    led = SimpleNamespace(nodes={"old": mk("old", ""), "port": mk("port", "ladder=old struct_retro=1"),
                                 "kid": mk("kid", "ladder=old struct_tune=backbone cell=lr:x"),
                                 "tune": mk("tune", "tune_of=old")})
    assert CPM.children_of(led) == {"old": {"kid", "tune"}}
