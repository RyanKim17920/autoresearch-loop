"""The data bank in the v3 loop (fix batch B, 10-12): ideation's intake requests a finish's dataset_spec, the core
holds a QUEUED node until its dataset is ready (voids it when the build failed / nobody asked), and briefs plus the
researcher's v3 context pack list the registered datasets from their manifests."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from loop import databank as DB
from loop import tick as TK
from loop.agents import researcher as RS
from loop.contracts import NodeStatus, ResearchResult
from loop.core import acquire as A
from loop.core import ideation as ID
from loop.core.status import Status

from .core.test_ideation import _cand, env  # noqa: F401  (the fixture)
from .test_databank import SPEC, _ctx, _ds, _node


def test_queued_nodes_wait_for_their_dataset_and_are_voided_when_it_cannot_come(ctx, store, tmp_path, monkeypatch):
    root = tmp_path / "bank"
    c = _ctx(ctx, root)
    _ds(root, "ready1", "ready")
    _ds(root, "busy", "extracting")
    store.ensure_root("r0")
    _node(store, "n0001", {"data.dataset_dir": f"{root}/ready1"})
    _node(store, "n0002", {"data.dataset_dir": f"{root}/busy"})
    _node(store, "n0003", {"data.dataset_dir": f"{root}/ghost"})
    core, acts = TK.ModuleCore(), []
    st = {"n0001": Status.QUEUED, "n0002": Status.QUEUED, "n0003": Status.QUEUED}
    held = core._held(SimpleNamespace(ctx=c, store=store), st, acts)
    assert held == {"n0002", "n0003"}
    assert store.get_node("n0003").status is NodeStatus.VOID and "never requested" in store.get_node("n0003").reject_reason
    assert store.get_node("n0002").status is not NodeStatus.VOID and any(a.startswith("void n0003") for a in acts)
    assert core._held(SimpleNamespace(ctx=ctx, store=store), st, []) == frozenset()     # the bank off: nothing held
    core.held = held
    opts = [A.Option("screen", n, "explore", 0.0, 1e-6, 1e-6, 3.0, 3) for n in ("n0001", "n0002")]
    monkeypatch.setattr(A, "ledger_options", lambda *a, **k: opts)
    post = SimpleNamespace(led=SimpleNamespace(runs=(), nodes={}, epoch=""))
    data = SimpleNamespace(runs=(), nodes={}, epoch="")
    got = core.rank(data, [], st, post, 64, 0)
    assert [a.node for a in got] == ["n0001"]                                          # n0002 waits


def test_intake_requests_the_finish_dataset_spec(env, tmp_path):  # noqa: F811
    ctx, store, led, ide = env
    ide.ctx = _ctx(ctx, tmp_path / "bank")
    req = ide.request(ID.Brief("explore", "explore", ("root",), "b"))
    res = ResearchResult("ok", _cand("train on 448 px tiles at 0.5 mpp", commit="cD"))
    out = ide.intake(req, dataclasses.replace(res, plan={"dataset_spec": SPEC}))
    nid = out.split()[1]
    assert out.startswith("queued") and "databank:" in out and "queued for a build" in out
    [(name, r)] = DB.book(store)["requests"].items()
    assert r["by"] == [nid] and r["state"] == "waiting" and name.startswith("ds448_")
    none = ide.intake(req, ResearchResult("failed", None, "x", plan={"dataset_spec": {**SPEC, "seed": 1}}))
    assert "databank:" in none and any(req.id in r["by"] for r in DB.book(store)["requests"].values())


def test_briefs_and_the_v3_context_pack_list_the_registered_datasets(env, tmp_path):  # noqa: F811
    ctx, store, led, ide = env
    root = tmp_path / "bank"
    for name in ("mixed_448", "mpp050_448"):
        _ds(root, name, "ready")
    _ds(root, "mixed_448_pilot", "ready")
    ide.ctx = c = _ctx(ctx, root)
    tick = SimpleNamespace(research=SimpleNamespace(free=lambda: 0, researcher=SimpleNamespace(llm=None)),
                           pool=None, dry=True, core=SimpleNamespace(harm={}))
    acts = ide.ideate(tick, [], led, {})
    assert sum(a.startswith("databank: registered") for a in acts) == 2              # read from the manifests
    b = ide.brief_for("explore", led, ide.view(led))
    assert "## Data bank" in b.text and f"{root}/mixed_448: 4,000,000 tiles of 448 px" in b.text
    assert "pilot" not in b.text.split("## Data bank")[1].split("##")[0]
    assert "dataset_spec" in b.text
    pack = RS._bank_block(c, store)
    assert f"{root}/mpp050_448" in pack and "dataset_spec" in pack and RS._bank_block(ctx, store) == ""
