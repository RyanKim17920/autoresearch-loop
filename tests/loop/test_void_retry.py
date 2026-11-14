"""A node VOID only by infra voids keeps its bounded infra retries ("a void is not dead"): Tick._retries."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from loop.contracts import NodeStatus, RunStatus, VoidType
from tests.loop.test_adapters import run_of


def _void_node_tick(runs, stored="void", owned=True, notes="", result=RunStatus.RUNNING):
    from loop.tick import K_NODE_SYNC
    tried, updates, logs = [], [], []
    state = {K_NODE_SYNC: {"n7": stored}} if owned else {}

    def retry(run, node):
        tried.append(run.id)
        return dataclasses.replace(run, status=result)
    node = SimpleNamespace(id="n7", status=SimpleNamespace(value=stored), notes=notes, updated_ts=123.0)
    store = SimpleNamespace(
        runs=lambda status=None, node_id=None: [r for r in runs if node_id in (None, r.node_id)],
        get_node=lambda nid: node, get_state=lambda k: state.get(k),
        set_state=lambda k, v: state.__setitem__(k, v),
        update_node=lambda nid, **f: updates.append((nid, f)), log=lambda ev, **f: logs.append((ev, f)))
    fake = SimpleNamespace(store=store, book=SimpleNamespace(infra_retry=3, free=lambda: 4,
                                                            pool=lambda *a: "primary", retry=retry))
    return fake, SimpleNamespace(tried=tried, updates=updates, logs=logs, state=state)


def _run_retries(fake):
    from loop.tick import Tick
    fake._infra_void_only = lambda nid, st: Tick._infra_void_only(fake, nid, st)
    fake._reopen = lambda node: Tick._reopen(fake, node)
    acts = []
    return Tick._retries(fake, {"n7": "VOID"}, acts), acts


def _infra(seed, attempt=0):
    return dataclasses.replace(run_of("n7", seed), status=RunStatus.VOID, void_type=VoidType.INFRA, attempt=attempt)


def test_a_node_void_only_by_infra_voids_is_retried_under_the_cap_and_reopened():
    """Live 11-11 n0168: its one screen voided on submit_failed; the node derived VOID (CLOSED) and _retries never
    re-submitted it although its infra retries were all left. A void is not dead: it is retried, and its stored
    status goes back to screening at once (sync-owned, updated_ts kept) so stored-status readers see it live."""
    from loop.tick import K_NODE_SYNC
    r1, r2 = _infra(1), _infra(2)
    fake, rec = _void_node_tick([r1, r2])
    out, acts = _run_retries(fake)
    assert out == [r1.id, r2.id] and rec.tried == [r1.id, r2.id] and acts[0] == f"retry {r1.id} -> running"
    assert rec.updates == [("n7", {"status": NodeStatus.SCREENING, "updated_ts": 123.0})]     # reopened once
    assert rec.state[K_NODE_SYNC]["n7"] == "screening"
    assert rec.logs == [("node_transition", {"node_id": "n7", "frm": "void", "to": "screening",
                                             "reason": "infra retry re-submitted"})]


def test_a_retry_that_voids_again_does_not_reopen_the_node():
    fake, rec = _void_node_tick([_infra(1)], result=RunStatus.VOID)
    assert _run_retries(fake)[0] == [_infra(1).id] and rec.updates == [] and rec.logs == []


def test_a_node_void_only_by_infra_voids_stays_void_at_the_cap_or_when_closed_by_an_act():
    fake, rec = _void_node_tick([_infra(1, attempt=3)])
    assert _run_retries(fake)[0] == [] and rec.tried == [] and rec.updates == []   # cap respected: stays VOID
    cancelled = dataclasses.replace(run_of("n7", 2), status=RunStatus.VOID, void_type=VoidType.CANCELLED)
    fake, rec = _void_node_tick([_infra(1), cancelled])                            # not ONLY infra voids
    assert _run_retries(fake)[0] == [] and rec.tried == []
    fake, rec = _void_node_tick([_infra(1)], stored="rejected", owned=False)      # an explicit act closed it
    assert _run_retries(fake)[0] == [] and rec.tried == []


@pytest.mark.parametrize("notes", ["request=r1 rescue_closed=1", "port_of=n0091 port_failed port_error"])
def test_a_node_closed_on_purpose_is_never_revived(notes):
    """rescue_closed=1 (a rescue point that never ran) and port_failed (a failed graft) close a node on purpose."""
    fake, rec = _void_node_tick([_infra(1)], notes=notes)
    assert _run_retries(fake)[0] == [] and rec.tried == [] and rec.updates == []
