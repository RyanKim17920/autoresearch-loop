"""Ports (docs/loop-core-v3.md §8 "Base epochs", cutover step 12): an old-epoch node's change grafted onto the current
base by the adapter (``graft_port`` / ``merge_config``), a ``port_of=<old>`` node on the anchor's counterpart; a conflict
is a REJECTED port node (never offered again) plus one researcher port session; the old node is never touched."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from loop import cutover as CO
from loop import tick as TK
from loop.adapters.sim import SimAdapter
from loop.contracts import Node, NodeStatus, Run, RunStatus, Stage, Strategy
from loop.core import data as CD
from loop.store import SqliteStore


def _commit(sim, parent, name, cfg=None, files=None):
    wt = sim.prepare_worktree(parent_commit=parent, name=name)
    if cfg:
        p = Path(wt.path) / "config.yaml"
        c = yaml.safe_load(p.read_text())
        for key, value in cfg.items():
            sec, leaf = key.split(".")
            c[sec][leaf] = value
        p.write_text(yaml.safe_dump(c))
    for rel, text in (files or {}).items():
        (Path(wt.path) / rel).write_text(text)
    return sim.commit(wt, message=name)


def _node(store, nid, commit, parent, ts, notes="", status=NodeStatus.SCREENING, strategy=Strategy.EXPLORE):
    store.put_node(Node(id=nid, task_id=store.task_id, commit=commit, parent_id=parent, strategy=strategy,
                        operator=None, family="f", mechanism_tag=nid, diff_hash=nid, status=status, created_ts=ts,
                        updated_ts=ts, notes=notes))


def _done(store, nid, seed, y=0.67):
    store.put_run(Run(id=Run.make_id(nid, Stage.SCREEN, seed), task_id=store.task_id, node_id=nid, seed=seed,
                      stage=Stage.SCREEN, status=RunStatus.DONE, scores={k: y for k in CD.PROBE_KEYS.values()},
                      aggregate=y, submitted_ts=10.0, gpu_h=1.0))


def _two_epochs(ctx, tmp_path):
    """Old base: root -> inc (lr 5e-4) -> x (aug .3), z (lr 2e-4); root -> y (wd .01). New base root2 (a code change)
    -> inc@ (the incumbent's port, lr re-tuned to 7e-4 on the new base)."""
    sim = SimAdapter(ctx)
    store = SqliteStore(tmp_path / "loop.db", ctx.task_id, weights=ctx.cfg.weights)
    r = sim.root_commit()
    store.ensure_root(r)
    store.update_node("root", created_ts=1.0)
    inc = _commit(sim, r, "inc", {"train.lr": 5e-4})
    _node(store, "inc", inc, "root", 2.0)
    _node(store, "x", _commit(sim, inc, "x", {"aug.strength": 0.3}), "inc", 3.0)
    _node(store, "z", _commit(sim, inc, "z", {"train.lr": 2e-4}), "inc", 3.5)
    _node(store, "y", _commit(sim, r, "y", {"train.weight_decay": 0.01}), "root", 4.0)
    r2 = _commit(sim, r, "root2", files={"train.py": "def train(cfg):\n    return 'crn'\n"})
    _node(store, "root2", r2, None, 10.0, status=NodeStatus.BASELINE, strategy=Strategy.BASELINE)
    _node(store, "inc@new", _commit(sim, r2, "incp", {"train.lr": 7e-4}), "root2", 11.0, notes="port_of=inc")
    for s in (700001, 700002, 700003):
        _done(store, "root2", s)
        _done(store, "inc@new", s, 0.68)
    return sim, store


def _cfg(sim, sha):
    return yaml.safe_load(sim._object(sha)["files"]["config.yaml"])


def test_port_grafts_the_change_onto_the_anchors_counterpart_and_tags_it(ctx, tmp_path):
    sim, store = _two_epochs(ctx, tmp_path)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    base = led.epoch
    assert CO.anchor(led, "x") == ("inc", "inc@new") and CO.anchor(led, "y") == ("root", "root2")
    before = store.get_node("x")
    p = CO.port_node(store, sim, led, "x", 20.0)
    assert p.ok and p.node == f"x@{base[:7]}" and p.parent == "inc@new" and p.anchor == "inc"
    node = store.get_node(p.node)
    assert node.status is NodeStatus.QUEUED and node.parent_id == "inc@new" and "port_of=x" in node.notes
    cfg = _cfg(sim, node.commit)
    assert cfg["aug"]["strength"] == 0.3 and cfg["train"]["lr"] == 7e-4          # x's change on the port's config
    assert "crn" in sim._object(node.commit)["files"]["train.py"]                  # on the new base's code
    assert store.get_node("x") == before                                           # the old node is never touched
    q = CO.port_node(store, sim, led, "y", 21.0)
    assert q.ok and q.parent == "root2" and _cfg(sim, store.get_node(q.node).commit)["train"]["weight_decay"] == 0.01
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert led.port("x") == p.node and led.nodes[p.node].port_of == "x" and led.nodes[p.node].base == base
    assert CO.port_node(store, sim, led, "x", 22.0).how == "exists"               # idempotent
    assert [e["node_id"] for e in store.events("port")] == [p.node, q.node]
    store.close()


def test_a_conflicting_port_is_a_rejected_node_and_queues_one_researcher_session(ctx, tmp_path):
    sim, store = _two_epochs(ctx, tmp_path)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    tick = TK.Tick(ctx, store, sim, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=lambda: 30.0)
    lines = tick.port_graft("z", led)                 # z moved lr 5e-4 -> 2e-4; the port moved it to 7e-4
    pid = CO.port_id("z", led.epoch)
    assert lines[0].startswith("port z onto inc@new: conflict: config.yaml:train.lr") and "queued" in lines[1]
    node = store.get_node(pid)
    assert node.status is NodeStatus.REJECTED and node.reject_reason.startswith("port conflict")
    assert [e["node_id"] for e in store.events("port_conflict")] == ["z"]
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert led.port("z") == pid                       # "ported" (as a failure): no port option again
    assert tick.port_graft("z", led) == []
    started = []
    tick.research.start = lambda req, *a, **k: started.append(req)
    tick.research.free = lambda: 1
    acts: list[str] = []
    tick._port_sessions(acts)
    tick._port_sessions(acts)                         # ONE session per node
    [req] = started
    assert req.id == "req-v3-port-z" and req.parent_ids == ("inc@new",) and req.brief_kind == "port"
    assert "PORT z onto inc@new" in req.brief and "train.lr" in req.brief
    tick.close()
    store.close()


def test_researcher_port_session_result_becomes_the_port(ctx, tmp_path):
    from loop.core.ideation import PORT_PREFIX
    assert PORT_PREFIX == "req-v3-port-"
    sim, store = _two_epochs(ctx, tmp_path)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    CO.port_node(store, sim, led, "z", 20.0)          # rejected
    _node(store, "n0100", _commit(sim, store.get_node("inc@new").commit, "zz", {"train.lr": 3e-4}), "inc@new", 40.0,
          notes="request=req-v3-port-z brief=port port_of=z", status=NodeStatus.QUEUED)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert led.port("z") == "n0100"                    # the latest port supersedes the rejected graft
    store.close()


def test_a_port_action_launches_the_port_and_its_screening_runs_on_the_parents_seeds(ctx, tmp_path):
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    tick = TK.Tick(ctx, store, sim, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=lambda: 30.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    acts: list[str] = []
    launched = tick._launch([TK.Action("port", "x", "explore", 3.1, 1.0, 3)], led, {}, 30.0, acts)
    pid = CO.port_id("x", led.epoch)
    assert any(a.startswith(f"port x -> {pid} on inc@new") for a in acts)
    assert sorted(launched) == [f"{pid}:screen:{s}" for s in (700001, 700002, 700003)]   # CRN with the incumbent port
    tick.close()
    store.close()


NANOPATH = Path("/home/user/nanopath/base")
REAL = {"root": "6c142b9b4f373de6f6ede5ebecc06d2744e5da94", "n0655": "8746fbda4b6694c19903596980e0fcbf375854d3",
        "n1181": "7d84885cff44b1a72097059a7d705040cc001860", "n1431": "4dde654830ee5b90a473a85fc948b9d98d150bb2",
        "root-065b87f": "065b87f57b9d1ee1fafe8df9fcdc2b0e4b2b5d8f",
        "n0655@065b87f": "6a6c1efd2486e324f8d3b9dcd67cb2c86a3ad577"}


def _has_commits() -> bool:
    import subprocess
    return NANOPATH.is_dir() and all(subprocess.run(["git", "-C", str(NANOPATH), "cat-file", "-e", f"{c}^{{commit}}"],
                                                    capture_output=True).returncode == 0 for c in REAL.values())


@__import__("pytest").mark.skipif(not _has_commits(), reason="needs the nanopath repo with the crn-knobs commits")
def test_real_repo_port_in_a_temp_clone(tmp_path):
    """The nanopath adapter's graft on a --shared clone of the real repo (no ref of the real repo moves): n1181's
    change applies onto the crn-knobs n0655 port, n1431's conflicts in train.py (a REJECTED port node)."""
    import subprocess
    from loop.adapters.nanopath_v2 import NanopathV2Adapter
    from loop.spec import load_context
    # v3 review O7: only the refs this test could move -- the live loop's own loop/* branches (research sessions)
    # change under it while the test runs, and those are not the test's
    refs = lambda: "\n".join(l for l in subprocess.run(  # noqa: E731
        ["git", "-C", str(NANOPATH), "for-each-ref"], capture_output=True, text=True).stdout.splitlines()
        if not l.split("\t")[-1].startswith("refs/heads/loop/"))
    before = refs()
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(NANOPATH), str(clone)], check=True)
    ctx = load_context(Path(__file__).resolve().parents[2] / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path / "st")
    ad = NanopathV2Adapter(ctx, repo=str(clone), runs_root=str(tmp_path / "runs"), worktrees_root=str(tmp_path / "wt"))
    store = SqliteStore(tmp_path / "loop.db", ctx.task_id)
    for i, (nid, parent) in enumerate((("root", None), ("n0655", "root"), ("n1181", "n0655"), ("n1431", "n0655"),
                                       ("root-065b87f", None))):
        _node(store, nid, REAL[nid], parent, float(i), status=NodeStatus.BASELINE if parent is None else
              NodeStatus.SCREENING)
    _node(store, "n0655@065b87f", REAL["n0655@065b87f"], "root-065b87f", 10.0, notes="port_of=n0655")
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    ok = CO.port_node(store, ad, led, "n1181", 20.0)
    assert ok.ok and ok.parent == "n0655@065b87f" and ok.anchor == "n0655"
    cfg, onto = ad.config_at(ok.commit), ad.config_at(REAL["n0655@065b87f"])
    assert {k: v for k, v in cfg.items() if onto.get(k) != v} == {"dino.jepa_rank_fb_band": 0.75}
    bad = CO.port_node(store, ad, led, "n1431", 21.0)
    assert bad.how.startswith("conflict") and "train.py" in bad.how
    assert store.get_node(bad.node).status is NodeStatus.REJECTED
    assert refs() == before                            # the real repo: no branch, no tag moved
    store.close()
