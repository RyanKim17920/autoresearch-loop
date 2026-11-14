"""Fix-and-continue: GPU OOM classification, the numerics-neutral OOM ladder (nanopath: allocator env;
the sim plants a two-step ladder) before any verdict, the deterministic DEBUG request for a crashed idea node,
the retro pass on deploy, and a designed batch's heaviest-point memory smoke.
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import yaml

from loop import design as D
from loop.contracts import (ROOT_NODE_ID, Node, NodeStatus, Operator, Run, RunStatus, Stage, Strategy,
                            VoidType, is_oom, next_mitigation, oom_note, with_oom_note)
from loop.driver import K_DEBUG
from tests.loop.test_adapters import (LAUNCHER, SlurmRunner, base_repo, finish_job, git, make_np,  # noqa: F401
                                     np_ctx, run_of, sbatch_ok)
from tests.loop.test_driver import Scripts, _make, _world

LIVE_WARN = ("[W929 21:18:07.346296875 CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on "
             "device 0 while trying to allocate 1312817152 bytes (free: 336003072, total: 85009367040).")
LIVE_ERR = ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB. GPU 0 has a total "
            "capacity of 79.17 GiB ... try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
LADDER = ("expandable_segments", "activation_checkpointing")


def test_oom_is_classified_from_the_log_tail():
    assert is_oom(LIVE_WARN) and is_oom(LIVE_ERR) and is_oom("RuntimeError: CUDA error: out of memory")
    assert not is_oom("ValueError: shapes (3,) and (4,)") and not is_oom("")
    assert next_mitigation(LADDER) == LADDER[0] and next_mitigation(LADDER, LADDER[0]) == LADDER[1]
    assert next_mitigation(LADDER, "", LADDER[1]) == "" and next_mitigation((), "") == ""
    n = Node(id="n", task_id="t", commit="c", parent_id=None, strategy=Strategy.EXPLORE, operator=None,
             family="f", mechanism_tag="m", diff_hash="h", status=NodeStatus.SCREENING, notes="request=r1 oom=a")
    assert oom_note(dataclasses.replace(n, notes=with_oom_note(n.notes, "b"))) == "b"
    assert with_oom_note(n.notes, "b") == "request=r1 oom=b"


# ------------------------------------------------------------------------------ nanopath adapter
def test_nanopath_oom_is_a_crash_even_with_a_cuda_error_line(np_ctx, base_repo):
    runner = SlurmRunner(sbatch_ok())
    adapter, _ = make_np(np_ctx, base_repo, runner=runner)
    from types import SimpleNamespace
    handle = adapter.submit(run=run_of("n1", 11), commit=git(base_repo, "rev-parse", "HEAD"))
    Path(handle.output_dir).mkdir(parents=True)
    l = SimpleNamespace(adapter=adapter, runner=runner, handle=handle)
    st = finish_job(l, state="FAILED", summary=None, err="RuntimeError: CUDA error: out of memory\n")
    assert st.state is RunStatus.CRASH and is_oom(st.log_tail)


def test_nanopath_ladder_is_allocator_env_only(np_ctx, base_repo):
    """Activation checkpointing is NOT on the ladder: the FLOP counter counts the recompute and train.py
    keys its schedules on train_flops / max_train_flops, so it would change the numerics."""
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner(sbatch_ok()))
    cfg = yaml.safe_load((base_repo / "configs/main.yaml").read_text())
    cfg["train"]["activation_checkpointing"] = False
    (base_repo / "configs/main.yaml").write_text(yaml.safe_dump(cfg))
    git(base_repo, "commit", "-qam", "ckpt key")
    head = git(base_repo, "rev-parse", "HEAD")
    assert adapter.oom_ladder(head) == ("expandable_segments",)
    adapter.submit(run=dataclasses.replace(run_of("n1", 12, attempt=1), mitigation="expandable_segments"),
                   commit=head)
    call = next(c for c in runner.calls if c["argv"][0] == LAUNCHER)
    assert call["argv"][1] == "configs/main.yaml" and call["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    rec = json.loads((Path(np_ctx.state_dir) / "receipts/n1/screen_s12_a1.json").read_text())
    assert rec["env"]["PYTORCH_CUDA_ALLOC_CONF"]                              # a free retry keeps it
    runner.calls.clear()
    adapter.submit(run=run_of("n1", 13), commit=head)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in next(c for c in runner.calls if c["argv"][0] == LAUNCHER)["env"]


# ------------------------------------------------------------------------------ driver
def _events(store, name):
    return [e for e in store.events(name)]


def _tick_until(driver, cond, n=40):
    for _ in range(n):
        driver.tick()
        if cond():
            return True
    return False


def test_oom_retries_with_the_env_then_checkpointing_before_any_verdict(tmp_path):
    driver, store, adapter = _make(_world(tmp_path, review_every=1000), Scripts())
    adapter.inject_fault("n0001:screen:11", "oom", times=None, fixed_by="activation_checkpointing")
    assert _tick_until(driver, lambda: (r := store.get_run("n0001:screen:11")) is not None
                       and r.status is RunStatus.DONE)
    r = store.get_run("n0001:screen:11")
    assert r.mitigation == "activation_checkpointing" and r.attempt == 2
    assert [e["mitigation"] for e in _events(store, "run_oom_retry")] == list(LADDER)
    n = store.get_node("n0001")
    assert "oom=activation_checkpointing" in n.notes and n.status is not NodeStatus.CRASHED
    _tick_until(driver, lambda: bool(store.runs(node_id="n0001", stage=Stage.PROMOTE)), n=20)
    for p in store.runs(node_id="n0001", stage=Stage.PROMOTE):     # later runs start at the step
        assert p.mitigation == "activation_checkpointing"


def test_ladder_spent_the_crash_closes_and_a_debug_is_queued_at_once(tmp_path):
    driver, store, adapter = _make(_world(tmp_path, review_every=1000), Scripts())
    adapter.inject_fault("n0001:screen:11", "oom", times=None)          # nothing fixes it
    assert _tick_until(driver, lambda: store.get_node("n0001") is not None
                       and store.get_node("n0001").status is NodeStatus.CRASHED)
    assert store.get_run("n0001:screen:11").mitigation == "activation_checkpointing"
    driver.tick()
    started = [e for e in store.events("research_started") if e.get("deterministic_debug")]
    assert [e["parents"] for e in started] == [["n0001"]]
    child = next(n for n in store.nodes() if n.operator is Operator.DEBUG)
    assert child.second_parent_id == "n0001" and child.debug_depth == 1 and child.parent_id == ROOT_NODE_ID
    assert child.strategy is Strategy.REVISIT
    assert (store.get_state(K_DEBUG) or {}).get("n0001")
    for _ in range(3):
        driver.tick()                                                   # once per crashed node
    assert len([e for e in store.events("research_started") if e.get("deterministic_debug")]) == 1


def test_debug_prompt_carries_the_crash_tail_and_the_mitigation_tried(tmp_path):
    driver, store, adapter = _make(_world(tmp_path, review_every=1000), Scripts())
    adapter.inject_fault("n0001:screen:11", "oom", times=None)
    _tick_until(driver, lambda: (n := store.get_node("n0001")) is not None and n.status is NodeStatus.CRASHED)
    from loop.contracts import CandidateRequest
    req = CandidateRequest(id="r", task_id=store.task_id, strategy=Strategy.REVISIT, operator=Operator.DEBUG,
                           island="main", parent_ids=("n0001",), brief="b", brief_kind="failure")
    prompt = driver.researcher._prompt(req, store, driver.snapshot, [store.get_node("n0001")], "c", 1, "n/a")
    assert "CUDA out of memory" in prompt and "activation_checkpointing" in prompt and "parent diff" in prompt


def _plant_crash(store, adapter, nid, tail, *, depth=0, strategy=Strategy.MOONSHOT, operator=Operator.DRAFT):
    root = store.get_node(ROOT_NODE_ID)
    store.put_node(Node(id=nid, task_id=store.task_id, commit=root.commit, parent_id=ROOT_NODE_ID,
                        strategy=strategy, operator=operator, family="optimization", mechanism_tag="x",
                        diff_hash=f"h{nid}", status=NodeStatus.CRASHED, debug_depth=depth, created_ts=1.0,
                        notes="request=old brief=moonshot"))
    for seed in (11, 12):                                              # both screen seeds crashed
        store.put_run(Run(id=Run.make_id(nid, Stage.SCREEN, seed), task_id=store.task_id, node_id=nid, seed=seed,
                          stage=Stage.SCREEN, status=RunStatus.CRASH, log_tail=tail))


def test_retro_on_deploy_reopens_oom_crashes_and_queues_other_crashes_once(tmp_path):
    ctx = _world(tmp_path, review_every=1000)
    driver, store, adapter = _make(ctx, Scripts())
    driver.tick()                                                       # root
    _plant_crash(store, adapter, "c_oom", LIVE_ERR)
    _plant_crash(store, adapter, "c_bug", "ValueError: bad shape")
    _plant_crash(store, adapter, "c_deep", "ValueError: bad shape", depth=ctx.cfg.researcher.max_debug)
    _plant_crash(store, adapter, "c_pt", LIVE_ERR, strategy=Strategy.DESIGN, operator=None)   # old batch's point
    store.put_node(Node(id="c_bug_child", task_id=store.task_id, commit="x", parent_id=ROOT_NODE_ID,   # an
                        strategy=Strategy.REVISIT, operator=Operator.DEBUG, family="optimization",      # earlier
                        mechanism_tag="x", diff_hash="hc", status=NodeStatus.VETOED,                    # bandit
                        second_parent_id="c_bug", debug_depth=1))                                       # debug
    from loop.contracts import K_INFLIGHT
    store.set_state(K_INFLIGHT, [])                                    # no research in flight at the deploy
    driver2, store2, _ = _make(ctx, Scripts(), adapter=adapter)         # a fresh process = the deploy
    driver2.tick()
    n = store2.get_node("c_oom")
    assert n.status is NodeStatus.SCREENING and "oom=expandable_segments" in n.notes
    r = store2.get_run("c_oom:screen:11")
    assert r.mitigation == "expandable_segments" and r.attempt >= 1   # relaunched through the owed retry
    assert store2.get_run("c_oom:screen:12").mitigation == "expandable_segments"
    assert store2.get_node("c_pt").status is NodeStatus.CRASHED       # not the running batch: left alone
    book = store2.get_state(K_DEBUG) or {}
    assert "c_bug" in book and "c_deep" not in book and "c_oom" not in book and "c_pt" not in book
    driver2.tick()
    assert len([e for e in store2.events("research_started") if e.get("deterministic_debug")]) == 1


def test_a_debug_outage_is_asked_again(tmp_path):
    from loop.contracts import ResearchResult
    ctx = _world(tmp_path, review_every=1000)
    driver, store, adapter = _make(ctx, Scripts())
    driver.tick()
    _plant_crash(store, adapter, "c_bug", "ValueError: bad shape")
    real = driver.researcher.research
    calls = []

    def flaky(req, st, snap):
        calls.append(req.operator)
        if req.operator is Operator.DEBUG and calls.count(Operator.DEBUG) == 1:
            return ResearchResult("backend_unavailable", None, "backend_unavailable: down")
        return real(req, st, snap)
    driver.researcher.research = flaky
    for _ in range(3):
        driver.tick()
    assert calls.count(Operator.DEBUG) == 2
    assert any(n.second_parent_id == "c_bug" and n.operator is Operator.DEBUG for n in store.nodes())


# ------------------------------------------------------------------------------ design pre-smoke
def test_heaviest_row_counts_memory_knobs_at_their_larger_level():
    knobs = [{"key": "train.local_size", "low": 128, "high": 192}, {"key": "train.local_views", "low": 14, "high": 9},
             {"key": "dino.lr", "low": 1e-4, "high": 2e-4}, {"key": "data.local_crop_scale", "low": [0.08, 0.35],
                                                            "high": [0.16, 0.35]}]
    rows = D.matrix(4, 16)
    i = D.heaviest(knobs, rows)
    assert rows[i][0] == 1 and rows[i][1] == -1                          # big crops, the 14 views
    assert D.heaviest(knobs[2:], rows) is None                          # lr / crop scale: no memory knob


def test_a_batch_is_held_until_its_heaviest_point_passes_a_memory_smoke(tmp_path, monkeypatch):
    from tests.loop.test_design import _driver
    ctx, driver, store = _driver(tmp_path)
    adapter = driver.adapter
    adapter.oom_above = 10                                              # sim: epochs > 10 OOMs...
    monkeypatch.setattr(D, "MEMORY_KEY", D.re.compile("epochs"))        # ...and epochs is the memory knob
    driver.tick()
    D.request(store, [{"key": "train.lr", "high": 0.0005}, {"key": "train.weight_decay", "high": 0.025},
                      {"key": "aug.strength", "high": 0.25}, {"key": "train.epochs", "high": 20}])
    acts = driver.tick()
    cur = store.get_state(D.K_DESIGN)["current"]
    assert cur["smoke"]["status"] == "passed" and cur["smoke"]["mitigation"] == "activation_checkpointing"
    assert [s["mitigation"] for s in adapter.smokes_seen] == ["", *LADDER] and all(s["memory"] for s in adapter.smokes_seen)
    assert any("memory smoke passed" in a for a in acts)
    pts = [store.get_node(i) for i in cur["points"].values() if i != ROOT_NODE_ID]
    assert all("oom=activation_checkpointing" in p.notes for p in pts)
    for _ in range(12):
        driver.tick()
    runs = [r for p in pts for r in store.runs(node_id=p.id)]
    assert runs and all(r.mitigation == "activation_checkpointing" for r in runs)
    assert not any(r.status is RunStatus.CRASH for r in runs)


def test_points_launch_nothing_while_the_smoke_is_pending(tmp_path, monkeypatch):
    from tests.loop.test_design import _driver
    ctx, driver, store = _driver(tmp_path)
    monkeypatch.setattr(D, "MEMORY_KEY", D.re.compile("epochs"))
    driver.tick()
    driver._design_smoke = lambda acts: None                            # the smoke never returns
    D.request(store, [{"key": "train.lr", "high": 0.0005}, {"key": "train.weight_decay", "high": 0.025},
                      {"key": "aug.strength", "high": 0.25}, {"key": "train.epochs", "high": 20}])
    for _ in range(3):
        driver.tick()
    held = D.held(store)
    assert held and not any(store.runs(node_id=i) for i in held)
    assert D.smoke_done(store, __import__("loop.contracts", fromlist=["SmokeResult"]).SmokeResult(True), LADDER)
    assert not D.held(store)


def test_patch_size_is_heavier_at_its_lower_level_and_val_batches_is_not_a_memory_knob():
    knobs = [{"key": "train.patch_size", "low": 8, "high": 16}, {"key": "eval.val_batches", "low": 2, "high": 8},
             {"key": "train.local_size", "low": 96, "high": 128}, {"key": "dino.wd", "low": 0.04, "high": 0.05}]
    rows = D.matrix(4, 16)
    i = D.heaviest(knobs, rows)
    assert rows[i][0] == -1 and rows[i][2] == 1                           # small patches = more tokens
    assert D.heaviest([knobs[1], knobs[3]], rows) is None                         # val_batches / 'size'-lookalikes: ignored
    assert not D.MEMORY_KEY.search("patch_size") and not D.MEMORY_KEY.search("val_batches")


def test_a_smoke_node_that_vanished_releases_the_batch_instead_of_raising(tmp_path, monkeypatch):
    from tests.loop.test_design import _driver
    ctx, driver, store = _driver(tmp_path)
    monkeypatch.setattr(D, "MEMORY_KEY", D.re.compile("epochs"))
    driver.tick()
    real = driver._design_smoke
    driver._design_smoke = lambda acts: None
    D.request(store, [{"key": "train.lr", "high": 0.0005}, {"key": "train.weight_decay", "high": 0.025},
                      {"key": "aug.strength", "high": 0.25}, {"key": "train.epochs", "high": 20}])
    driver.tick()
    assert D.smoke_due(store) is not None and D.held(store)
    monkeypatch.setattr(store, "get_node", lambda nid: None)
    acts: list[str] = []
    real(acts)                                                          # must not raise
    assert D.smoke_due(store) is None and not D.held(store)
    assert any("batch released" in a for a in acts)
