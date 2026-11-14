"""c3j pre-deploy review fixes: program dose-ladder guards + per-node bookkeeping, run_unrun re-root + retry."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from loop import databank as DB
from loop.contracts import NodeStatus, ROOT_NODE_ID
from loop.core import data as D
from loop.core import factorial as F
from loop.core import program as PG
from loop.core.status import Status

from .core.test_ideation_c3i import _store_node
from .test_databank import _ctx, _ds


def _ladder_env(store, cfg):
    store.ensure_root("r0")
    _store_node(store, "p1", ROOT_NODE_ID, "multi_part", NodeStatus.SCREENING, family="objective")
    nodes = {"p1": D.Node("p1", ROOT_NODE_ID, None, "moonshot", "objective", "explore", ROOT_NODE_ID, 1.0,
                          "screening", "request=r1 brief=program", base="B")}
    data = SimpleNamespace(nodes=nodes, port=lambda n, b=None: n, epoch="B")
    store.set_state("ideation:p1", {"knobs": [{"key": "loss.w", "lo": 0.25, "hi": 4.0, "scale": "log"}]})
    post = SimpleNamespace(by={}, est={"p1": {}}, epoch="B", prior=None, belief=lambda n: (0.002, 4e-6))
    built = []
    adapter = SimpleNamespace(config_rel="config.yaml", config_at=lambda c: cfg,
                              dry_build=lambda commit, levels, knobs, diffs, name: built.append(levels) or
                              (f"sha-{len(built)}", ""))
    return data, post, adapter, {"p1": Status.SCREENING}


def test_ladder_skips_a_knob_absent_from_the_config(store, ctx):
    data, post, adapter, st = _ladder_env(store, {})
    acts = PG.ladders(ctx, store, adapter, data, post, st, 10.0)
    assert store.get_state(PG.K_LADDERS)["p1"].startswith("-") and "absent" in acts[0]
    assert not [n for n in store.nodes() if "ladder=p1" in n.notes]


def test_ladder_skips_a_protected_knob(store, ctx, monkeypatch):
    import loop.extrapolate as EX
    data, post, adapter, st = _ladder_env(store, {"loss": {"w": 1.0}})
    monkeypatch.setattr(EX, "_allowed", lambda c, k: False)
    acts = PG.ladders(ctx, store, adapter, data, post, st, 10.0)
    assert store.get_state(PG.K_LADDERS)["p1"].startswith("-") and "protected" in acts[0]
    assert not [n for n in store.nodes() if "ladder=p1" in n.notes]


def test_ladder_bookkeeping_is_saved_per_node_so_a_mid_loop_failure_never_duplicates(store, ctx, monkeypatch):
    data, post, adapter, st = _ladder_env(store, {"loss": {"w": 1.0}})
    real, calls = F.build_point, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(F, "build_point", flaky)
    with pytest.raises(RuntimeError):
        PG.ladders(ctx, store, adapter, data, post, st, 10.0)
    booked = store.get_state(PG.K_LADDERS)["p1"]
    assert len(booked) == 1 and store.get_node(booked[0]) is not None
    n_before = len(store.nodes())
    assert PG.ladders(ctx, store, adapter, data, post, st, 20.0) == []     # booked: not rebuilt
    assert len(store.nodes()) == n_before


def _bank(ctx, store, tmp_path, names=(("a", 30),)):
    root = tmp_path / "bank"
    c = _ctx(ctx, root)
    for name, gb in names:
        d = _ds(root, name, "ready", px=224, spec={"seed": ord(name)})
        m = json.loads((d / "manifest.json").read_text())
        (d / "manifest.json").write_text(json.dumps({**m, "bytes": gb * 1e9}))
    DB.register(c, store)
    store.ensure_root("r0")
    _store_node(store, "n0001", ROOT_NODE_ID, "inc", NodeStatus.SCREENING)
    return root, c


def _adapter(built):
    return SimpleNamespace(config_rel="config.yaml", config_at=lambda commit: {"data": {"dataset_dir": "/x"}},
                           dry_build=lambda commit, levels, knobs, diffs, name: built.append(levels) or
                           (f"sha-{len(built)}", ""))


def test_reroot_skips_stranded_nodes_with_done_runs_and_moves_all_others(ctx, store, tmp_path):
    root, c = _bank(ctx, store, tmp_path)
    cfg = {"data.dataset_dir": str(root / "a")}
    for i, nid in enumerate(("n0002", "n0003", "n0004")):
        (d := tmp_path / f"{nid}.diff").write_text("diff --git a/config.yaml b/config.yaml\n+  x: 1\n")
        _store_node(store, nid, ROOT_NODE_ID, f"old{i}", cfg={**cfg, "train.seed": i}, diff=str(d))
    nodes = {n: D.Node(n, None, None, "explore", "f", "explore", ROOT_NODE_ID, 1.0, "queued", "",
                       base="B" if n == "n0001" else "A") for n in ("n0001", "n0002", "n0003", "n0004")}
    runs = (D.Run("r1", "n0004", 1, "screen", "done", 1.0, 0.0, 0.0, "", 0),)
    led = SimpleNamespace(epoch="B", nodes=nodes, port=lambda n, b=None: n, runs=runs)
    DB.run_unrun(c, store, _adapter([]), led, "n0001", 100.0)
    assert store.get_node("n0002").status is NodeStatus.VOID and store.get_node("n0003").status is NodeStatus.VOID
    assert store.get_node("n0004").status is NodeStatus.QUEUED                # has a done run: left alone
    assert str(store.get_state(DB.K_RUNS)["B"]["a"]).startswith("n")


def test_a_void_run_with_no_done_runs_retries_up_to_max_retries(ctx, store, tmp_path):
    root, c = _bank(ctx, store, tmp_path)
    nodes = {"n0001": D.Node("n0001", None, None, "explore", "f", "explore", ROOT_NODE_ID, 1.0, "screening", "",
                             base="B")}
    led = SimpleNamespace(epoch="B", nodes=nodes, port=lambda n, b=None: n, runs=())
    ad, ids = _adapter([]), []
    for t in range(3):                                                         # max_retries=2: 1 run + 2 retries
        DB.run_unrun(c, store, ad, led, "n0001", 100.0 + t)
        nid = store.get_state(DB.K_RUNS)["B"]["a"]
        ids.append(nid)
        nodes[nid] = D.Node(nid, None, None, "explore", "f", "explore", "n0001", 1.0, "void", "", base="B")
        store.update_node(nid, status=NodeStatus.VOID)
    assert len(set(ids)) == 3 and store.get_state(DB.K_RUNS)["B#retries"] == {"a": 2}
    assert DB.run_unrun(c, store, ad, led, "n0001", 200.0) == []              # budget spent: stays dropped
    assert store.get_state(DB.K_RUNS)["B"]["a"] == ids[-1]


def test_a_void_run_with_done_runs_is_not_retried(ctx, store, tmp_path):
    root, c = _bank(ctx, store, tmp_path)
    nodes = {"n0001": D.Node("n0001", None, None, "explore", "f", "explore", ROOT_NODE_ID, 1.0, "screening", "",
                             base="B")}
    led = SimpleNamespace(epoch="B", nodes=nodes, port=lambda n, b=None: n, runs=())
    DB.run_unrun(c, store, _adapter([]), led, "n0001", 100.0)
    nid = store.get_state(DB.K_RUNS)["B"]["a"]
    nodes[nid] = D.Node(nid, None, None, "explore", "f", "explore", "n0001", 1.0, "void", "", base="B")
    store.update_node(nid, status=NodeStatus.VOID)
    led.runs = (D.Run("r1", nid, 1, "screen", "done", 1.0, 0.0, 0.0, "", 0),)
    assert DB.run_unrun(c, store, _adapter([]), led, "n0001", 150.0) == []
