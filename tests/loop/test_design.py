"""DESIGN mode (docs/loop/DESIGN.md §4.6): the resolution-IV matrix, OLS recovery, the stacked node,
the trigger and one batch end to end on the simulator. Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
import itertools
import json
import random
from types import SimpleNamespace

import pytest
import yaml

from loop import design as D
from loop.contracts import (DESIGN_FAMILY, ROOT_NODE_ID, JobHandle, Node, NodeStatus, Operator, Run, RunStatus,
                            Stage, Strategy, VoidType)
from loop.spec import PoolCfg, ResearcherCfg, ReviewCfg, SlurmCfg, load_context
from tests.loop.conftest import InlineExecutor
from tests.loop.test_driver import FAKE, Scripts


@pytest.mark.parametrize("runs,k", [(r, k) for r in (16, 32) for k in D.GENERATORS[r]])
def test_matrix_is_resolution_iv(runs, k):
    rows = D.matrix(k, runs)
    cols = list(zip(*rows))
    assert len(rows) == runs == len(set(rows)) and len(cols) == k and set(rows[0]) == {-1}
    assert all(sum(c) == 0 for c in cols)                                  # balanced
    for i, j in itertools.combinations(range(k), 2):                       # mains clear of mains
        assert sum(a * b for a, b in zip(cols[i], cols[j])) == 0
    for i in range(k):                                                     # ... and of every 2FI
        for j, l in itertools.combinations([x for x in range(k) if x != i], 2):
            assert sum(a * b * c for a, b, c in zip(cols[i], cols[j], cols[l])) == 0
    assert D.n_runs(k, runs) == runs or runs == 32 and k <= 8 or runs == 16 and k == 5


def test_ols_recovers_planted_main_effects_and_flags_an_interaction():
    rng, rows = random.Random(3), D.matrix(7, 16)
    truth = [0.012, -0.008, 0.0, 0.004, 0.0, 0.0, -0.001]                  # high - low
    keys = [f"k{j}" for j in range(7)]
    obs = [(i, s, 0.6 + off + sum(t / 2 * x for t, x in zip(truth, r)) + 0.004 * r[0] * r[1]
            + rng.gauss(0, 0.0015)) for i, r in enumerate(rows) for s, off in ((11, 0.01), (12, -0.02))]
    res = D.analyze(keys, rows, obs, z=2.0)
    assert res["n"] == 32 and res["df"] == 32 - 2 - 7
    for e, t in zip(res["effects"], truth):
        assert abs(e["effect"] - t) < 4 * e["se"] and 0 < e["se"] < 0.002
    assert [e["significant"] for e in res["effects"][:2]] == [True, True]
    assert not res["effects"][2]["significant"]
    assert any("k0*k1" in f for f in res["interactions"])                  # its alias chain is flagged
    assert res["effects"][0]["effect"] == pytest.approx(0.012, abs=0.002)  # the seed offsets are removed


def test_stacked_levels_take_only_significant_effects_at_their_better_level():
    knobs = [{"key": k, "low": 1, "high": 2} for k in "abcd"]
    effects = [{"effect": 0.01, "significant": True}, {"effect": -0.01, "significant": True},
               {"effect": 0.02, "significant": False}, {"effect": -0.001, "significant": False}]
    assert D.stacked_levels(knobs, effects) == {"a": 2, "b": 1}


def _driver(tmp_path, *, slots: int = 40):
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    ctx = load_context(FAKE, state_dir=tmp_path / "state")
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, slots=slots, researcher=ResearcherCfg(max_concurrent=0),
        review=ReviewCfg(stall_evals=1000)))
    adapter = SimAdapter(ctx)
    driver, store = build(ctx, llm=Scripts().client(), adapter=adapter, executor=InlineExecutor,
                          rng=random.Random(7))
    driver.sleep = lambda s: None
    return ctx, driver, store


def test_a_requested_batch_runs_on_one_block_and_stacks_only_significant_knobs(tmp_path):
    ctx, driver, store = _driver(tmp_path)
    driver.tick()                                                          # root node
    D.request(store, [
        {"key": "train.lr", "high": 0.0005, "levers": {"effective_lr": -1}},       # sim: lower helps
        {"key": "train.weight_decay", "high": 0.025, "levers": {"regularization": -1}},
        {"key": "aug.strength", "high": 0.25, "levers": {"augmentation_strength": -1}},
        {"key": "train.warmup", "high": 10},
        {"key": "loss.temperature", "high": 0.2},
        {"key": "eval.seed", "high": 3},                                  # protected: dropped
        {"key": "no.such_key", "high": 1}])                               # unknown: dropped
    for _ in range(40):
        driver.tick()
        if store.get_state(D.K_DESIGN, {}).get("batches"):
            break
    (res,) = store.get_state(D.K_DESIGN)["batches"]
    assert [kb["key"] for kb in res["knobs"]] == ["train.lr", "train.weight_decay", "aug.strength",
                                                  "train.warmup", "loss.temperature"]
    points = [n for n in store.nodes() if n.strategy is Strategy.DESIGN and n.operator is None]
    assert len(points) == 15 and res["n"] == 32                          # row 0 is root itself
    assert {n.status for n in points} == {NodeStatus.SCREENED_OUT} and not store.postmortems()
    assert {r.seed for n in points for r in store.runs(node_id=n.id)} == set(ctx.cfg.seeds.screen)
    eff = {e["key"]: e for e in res["effects"]}
    assert eff["train.lr"]["significant"] and eff["train.lr"]["effect"] > 0
    stacked = store.get_node(res["stacked"])
    assert stacked.operator is Operator.DRAFT and stacked.family == DESIGN_FAMILY
    assert store.get_state(f"config_diff:{stacked.id}") == {
        kb["key"]: kb["high"] for kb, e in zip(res["knobs"], res["effects"]) if e["significant"]}
    assert stacked.levers.directions.get("effective_lr") == -1
    assert "train.lr" in D.lines(store) and "±" in D.lines(store)
    for _ in range(12):                                                    # the normal path, fresh seeds
        driver.tick()
    stacked = store.get_node(stacked.id)
    assert stacked.status is not NodeStatus.SCREENING
    assert not {r.seed for r in store.runs(node_id=stacked.id)} & set(ctx.cfg.seeds.screen)
    assert sum(bool(n.levers.directions.get("effective_lr")) for n in store.nodes()) >= 8   # the points carry levers


def test_a_review_design_request_reaches_the_book(tmp_path):
    from loop.contracts import ReviewReport
    ctx, driver, store = _driver(tmp_path)
    knobs = [{"key": "train.lr", "high": 0.0005}]
    driver.persist_review(ReviewReport("rv-1", ctx.task_id, "manual", "s", (), (), (), {}, 0,
                                       design=json.dumps(knobs)), [])
    assert store.get_state(D.K_DESIGN)["request"] == knobs and store.reviews()[0].design


def test_stall_trigger_starts_once_and_waits_for_more_closes(tmp_path):
    ctx, driver, store = _driver(tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, design=dataclasses.replace(
        ctx.cfg.design, stall_closes=20)))
    driver.tick()
    call = lambda evals: D.tick(ctx, store, driver.adapter, driver.snapshot, evals=evals,  # noqa: E731
                                evals_at_confirm=0, new_id=driver._new_id)
    assert call(19) == []
    assert "need 4" in call(20)[0]                     # nothing probed yet: no knobs, no batch
    assert call(25) == [] and store.get_state(D.K_DESIGN)["idle_at"] == 20


def test_auto_knobs_are_the_most_probed_keys_stepped_toward_the_better_side(ctx, store):
    base = {"train.lr": 0.001, "train.warmup": 5, "aug.strength": 0.5, "train.weight_decay": 0.05,
            "loss.temperature": 0.1, "model.depth": 12, "eval.seed": 0}
    from loop.adapters import set_dotted
    tree: dict = {}
    for k, v in base.items():
        set_dotted(tree, k, v)
    text = yaml.safe_dump(tree)
    posts, now = {}, 1.0
    for i, (key, val, u) in enumerate([("train.lr", 0.0005, 0.004), ("train.lr", 0.002, -0.003),
                                       ("train.warmup", 7, -0.002), ("aug.strength", 0.3, 0.001),
                                       ("train.weight_decay", 0.02, 0.002), ("loss.temperature", 0.2, 0.0),
                                       ("model.depth", 16, 0.01), ("eval.seed", 1, 0.0)]):
        nid = f"n{i}"
        store.put_node(Node(id=nid, task_id=ctx.task_id, commit=nid, parent_id=ROOT_NODE_ID,
                            strategy=Strategy.EXPLORE, operator=Operator.DRAFT, family="x", mechanism_tag=nid,
                            diff_hash=nid, status=NodeStatus.SCREENED_OUT, created_ts=now + i))
        store.set_state(f"config_diff:{nid}", {key: val})
        posts[nid] = SimpleNamespace(utility=SimpleNamespace(mean=u))
    knobs = D.pick_knobs(ctx, store, SimpleNamespace(nodes=posts), text, base)
    got = {kb["key"]: kb["high"] for kb in knobs}
    assert list(got)[0] == "train.lr" and got["train.lr"] == pytest.approx(0.001 / 1.5, rel=1e-3)
    assert got["train.warmup"] == 3 and got["train.weight_decay"] == pytest.approx(0.05 / 1.5, rel=1e-3)
    assert "model.depth" not in got and "eval.seed" not in got           # init_check family; protected
    assert D._step(0.999, -1, 1.5) == pytest.approx(0.9985) and D._step(1, -1, 1.5) is None


def test_design_points_prefer_the_preemptible_pool(ctx, store):
    from loop.driver import RunBook
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=(
        PoolCfg("primary", slots=2), PoolCfg("idle", slots=1, preemptible=True)))))
    book = RunBook(ctx, store, SimpleNamespace(submit=lambda run, commit: JobHandle(run.id, "1", "", 0.0)))
    pt = Node(id="n1", task_id=ctx.task_id, commit="c", parent_id=ROOT_NODE_ID, strategy=Strategy.DESIGN,
              operator=None, family=DESIGN_FAMILY, mechanism_tag="d1pt01", diff_hash="h",
              status=NodeStatus.SCREENING)
    store.put_node(pt)
    assert book.launch(pt, Stage.SCREEN, 11).pool == "idle"
    assert book.launch(pt, Stage.SCREEN, 12) is None and book.pool(pt) == ""     # waits: never primary
    assert store.get_run("n1:screen:12") is None
    assert book.launch(store.ensure_root("c0"), Stage.BASELINE, 11).pool == "primary"
    assert store.get_run("n1:screen:11").status is RunStatus.RUNNING


def test_batch_points_hold_at_most_max_share_of_the_slots_tuning_trials_excepted(ctx, store):
    from loop.driver import RunBook
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=(
        PoolCfg("primary", slots=2), PoolCfg("idle", slots=8, preemptible=True))),
        design=dataclasses.replace(ctx.cfg.design, max_share=0.15)))                # int(0.15 * 10) = 1 slot
    book = RunBook(ctx, store, SimpleNamespace(submit=lambda run, commit: JobHandle(run.id, "1", "", 0.0)))
    mk = lambda nid, notes="": Node(id=nid, task_id=ctx.task_id, commit="c", parent_id=ROOT_NODE_ID,  # noqa: E731
                                    strategy=Strategy.DESIGN, operator=None, family=DESIGN_FAMILY,
                                    mechanism_tag=nid, diff_hash=nid, status=NodeStatus.SCREENING, notes=notes)
    for n in (mk("p1"), mk("p2"), mk("t1", "tune=t3 design x=1")):
        store.put_node(n)
    assert book.launch(store.get_node("p1"), Stage.SCREEN, 11).pool == "idle"
    assert book.launch(store.get_node("p2"), Stage.SCREEN, 11) is None           # the batch holds its share
    assert book.launch(store.get_node("t1"), Stage.SCREEN, 11).pool == "idle"    # a tuning trial is not capped


def test_only_major_knobs_lever_backed_and_at_most_max_knobs(ctx, store):
    text = "".join(f"k{i}:\n  v: {i + 1}.0\n" for i in range(8))
    base = {f"k{i}.v": float(i + 1) for i in range(8)}
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, design=dataclasses.replace(
        ctx.cfg.design, max_knobs=4, lever_p=0.8)))
    from loop.select import intake
    fams = {k: intake.family_of(ctx, k) for k in base}
    req = [{"key": k, "high": v * 3} for k, v in base.items() if fams[k]]
    got = D.pick_knobs(ctx, store, SimpleNamespace(nodes={}), text, base, requested=req)
    assert len(got) == min(4, len(req))                                          # review knobs, capped
    lev = ctx.cfg.levers.vocabulary[0]
    kb = {"levers": {lev: 1}}
    snap = lambda p: SimpleNamespace(levers=(SimpleNamespace(lever=lev, target="panel", p_positive=p),))  # noqa: E731
    assert D.backed(ctx, snap(0.9), kb["levers"]) and not D.backed(ctx, snap(0.5), kb["levers"])
    assert D.backed(ctx, snap(0.1), {lev: -1}) and not D.backed(ctx, None, kb["levers"])


def test_review_requested_knob_in_an_init_check_family_is_refused(ctx, store):
    from loop.select import intake
    text = "train:\n  lr: 0.001\nmodel:\n  depth: 12\n"
    base = {"train.lr": 0.001, "model.depth": 12}
    assert ctx.cfg.families[intake.family_of(ctx, "model.depth")].init_check
    got = D.pick_knobs(ctx, store, SimpleNamespace(nodes={}), text, base,
                       requested=[{"key": "model.depth", "high": 16}, {"key": "train.lr", "high": 0.002}])
    assert [kb["key"] for kb in got] == ["train.lr"]


def test_smoke_jobs_count_in_pool_use_and_go_preemptible_first(ctx, store):
    from collections import Counter
    from loop.driver import RunBook
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=(
        PoolCfg("primary", slots=2), PoolCfg("idle", slots=1, preemptible=True)),
        smoke_skip_pending=0, smoke_primary_max=0)))     # the pre-c4j routing (c4j caps: test_review_c4j_runner)
    adapter = SimpleNamespace(smokes=Counter(), submit=lambda run, commit: JobHandle(run.id, "1", "", 0.0))
    book = RunBook(ctx, store, adapter)
    assert book.smoke_pool() == "idle" and book.used() == Counter(idle=1)   # the smoke reserve
    adapter.smokes["idle"] += 1                          # a smoke holds the only preemptible slot
    assert book.smoke_pool() == "primary" and book.used() == Counter(idle=1)
    adapter.smokes["primary"] += 2
    assert book.smoke_pool() == "idle"                   # nowhere free: it queues on idle, held against runs
    book.pool_states = {"primary": (2, 0), "idle": (0, 6)}   # SLURM runs none of our idle jobs: primary
    assert book.smoke_pool() == "primary"
    book.pool_states = {"primary": (2, 0), "idle": (1, 6)}   # our runs PENDING on idle: a smoke would wait behind
    assert book.smoke_pool() == "primary"                    # them (10-13): primary
    book.pool_states = {"primary": (2, 0), "idle": (1, 0)}
    assert book.smoke_pool() == "idle"
    book.pool_states = None
    pt = Node(id="n1", task_id=ctx.task_id, commit="c", parent_id=ROOT_NODE_ID, strategy=Strategy.DESIGN,
              operator=None, family=DESIGN_FAMILY, mechanism_tag="d1pt01", diff_hash="h",
              status=NodeStatus.SCREENING)
    store.put_node(pt)
    assert book.launch(pt, Stage.SCREEN, 11) is None     # the design point waits for the smoke


# ------------------------------------------------------------------ degenerate batches (d2: singular)
def _obs_for(rows, seeds=(11, 12), drop=(), truth=(0.01, -0.006, 0.0, 0.004), noise=0.0005, rng_seed=5):
    rng = random.Random(rng_seed)
    return [(i, s, 0.6 + off + sum(t / 2 * x for t, x in zip(truth, r)) + rng.gauss(0, noise))
            for i, r in enumerate(rows) if i not in drop for s, off in zip(seeds, (0.01, -0.02))]


def test_missing_rows_still_estimate_every_knob():
    rows = D.matrix(4, 16)
    res = D.analyze(list("abcd"), rows, _obs_for(rows, drop={3, 7, 12}), z=2.0)
    assert res["n"] == 26 and res["df"] == 26 - 2 - 4
    assert all(e["estimable"] for e in res["effects"])
    assert res["effects"][0]["significant"] and res["effects"][0]["effect"] == pytest.approx(0.01, abs=0.002)


def test_a_constant_column_is_flagged_and_the_rest_estimated():
    """d2: every point at dino.jepa_pred_width's high level (-1) crashed, so the scored rows hold it
    constant — collinear with the seed intercepts. That knob is reported, the other nine estimated."""
    rows = D.matrix(10, 32)
    keys = [f"k{j}" for j in range(10)]
    obs = [o for o in _obs_for(rows, truth=(0.01,) + (0.0,) * 9) if rows[o[0]][8] < 0]
    res = D.analyze(keys, rows, obs, z=2.0)
    eff = {e["key"]: e for e in res["effects"]}
    assert not eff["k8"]["estimable"] and "constant" in eff["k8"]["reason"] and eff["k8"]["effect"] is None
    assert all(eff[k]["estimable"] for k in keys if k != "k8") and eff["k0"]["significant"]
    assert "aliased_2fi" in eff["k0"]                   # the surviving half is resolution III: said so
    assert "not estimable" in D.fmt_effect(eff["k8"]) and D.stacked_levels(
        [{"key": k, "low": 0, "high": 1} for k in keys], res["effects"]) == {"k0": 1}


def test_a_duplicate_knob_and_an_aliased_column_are_refused():
    rows = D.matrix(4, 16)
    res = D.analyze(["a", "b", "a", "d"], rows, _obs_for(rows), z=2.0)
    assert [e["estimable"] for e in res["effects"]] == [True, True, False, True]
    assert res["effects"][2]["reason"] == "duplicate key"
    twin = [list(r) + [r[1]] for r in rows]                 # a 5th column equal to the 2nd
    assert D.aliased(twin) == [4] and D.aliased(rows) == []
    res = D.analyze(list("abcde"), twin, _obs_for(rows), z=2.0)
    assert not res["effects"][4]["estimable"] and "aliased with b" in res["effects"][4]["reason"]


def test_rank_deficient_or_empty_data_gives_no_estimate_never_raises():
    rows = D.matrix(4, 16)
    res = D.analyze(list("abcd"), rows, _obs_for(rows, seeds=(11,), drop=set(range(5, 16))), z=2.0)
    assert res["n"] == 5 and res["df"] == 1                               # rows 0-4: A never high
    assert [e["estimable"] for e in res["effects"]] == [False, True, True, True]
    res = D.analyze(list("abcd"), rows, _obs_for(rows, seeds=(11,), drop=set(range(16)) - {0, 6, 15}), z=2.0)
    assert res["n"] == 3 and res["df"] == 0 and not any(e["estimable"] for e in res["effects"])
    assert "3 scored observations" in res["effects"][1]["reason"]
    res = D.analyze(list("abcd"), rows, [], z=2.0)
    assert res["n"] == 0 and not any(e["estimable"] for e in res["effects"])


def test_knob_levels_must_be_valid_steps_of_roots_value(ctx, store):
    assert D._step(0, -1, 1.5) is None and D._step(0, 1, 1.5) is None     # d2: 0 -> -1 crashed all
    assert D._step(4, -1, 1.5) == 3 and D._step(-2, 1, 1.5) == -3
    text = "train:\n  lr: 0.001\n  weight_decay: 0.05\naug:\n  strength: [0.08, 0.35]\n"
    base = {"train.lr": 0.001, "train.weight_decay": 0.05, "aug.strength": [0.08, 0.35]}
    got = D.pick_knobs(ctx, store, SimpleNamespace(nodes={}), text, base, requested=[
        {"key": "aug.strength", "low": 0.08, "high": 0.16},            # d3: a scalar for root's list
        {"key": "aug.strength", "high": [0.16, 0.35]},
        {"key": "train.lr", "high": 0.001},                            # low == high
        {"key": "train.lr", "high": 0.002}, {"key": "train.lr", "high": 0.003},   # duplicate key
        {"key": "train.weight_decay", "high": "big"}])                 # a str for a number
    assert [(kb["key"], kb["high"]) for kb in got] == [("aug.strength", [0.16, 0.35]), ("train.lr", 0.002)]


def _batch(ctx, store, *, k=4, void=(), crash=(), score=lambda row: 0.6):
    """A closed k-knob 16-run batch in the store: row 0 = root (baselines at 11/12), rows in ``void``
    VOID (infra), in ``crash`` crashed, the rest screened out with ``score(row)`` at both seeds."""
    rows = D.matrix(k, 16)
    knobs = [{"key": f"train.k{j}", "low": 1.0, "high": 2.0, "levers": {}} for j in range(k)]
    store.ensure_root("c0")
    sc = lambda v: {a: v for a in ctx.cfg.axes}                                   # noqa: E731
    for s in (11, 12):
        store.put_run(Run(f"root:baseline:{s}", ctx.task_id, ROOT_NODE_ID, s, Stage.BASELINE, RunStatus.DONE,
                          scores=sc(score(rows[0]))))
    points = {"0": ROOT_NODE_ID}
    for i in range(1, 16):
        nid = f"n{i:02d}"
        st = NodeStatus.VOID if i in void else NodeStatus.CRASHED if i in crash else NodeStatus.SCREENED_OUT
        store.put_node(Node(id=nid, task_id=ctx.task_id, commit=nid, parent_id=ROOT_NODE_ID,
                            strategy=Strategy.DESIGN, operator=None, family=DESIGN_FAMILY,
                            mechanism_tag=f"d1pt{i:02d}", diff_hash=nid, status=st))
        store.set_state(f"config_diff:{nid}", {kb["key"]: kb["high"] for kb, x in zip(knobs, rows[i]) if x > 0})
        for s in (11, 12):
            rs = RunStatus.VOID if i in void else RunStatus.CRASH if i in crash else RunStatus.DONE
            store.put_run(Run(f"{nid}:screen:{s}", ctx.task_id, nid, s, Stage.SCREEN, rs, attempt=3,
                              void_type=VoidType.INFRA if i in void else None,
                              scores=sc(score(rows[i])) if rs is RunStatus.DONE else {}))
        points[str(i)] = nid
    return {"batch": 1, "knobs": knobs, "rows": [list(r) for r in rows], "points": points}


def _tick(ctx, store):
    return D.tick(ctx, store, None, None, evals=0, evals_at_confirm=0, new_id=lambda: "nX")


def test_nothing_estimable_reruns_voided_points_once_then_closes_with_the_reason(ctx, store):
    cur = _batch(ctx, store, void=set(range(1, 16)))
    store.set_state(D.K_DESIGN, {"current": cur, "batches": []})
    (act,) = _tick(ctx, store)
    assert "re-running 15 voided point(s)" in act
    book = store.get_state(D.K_DESIGN)
    assert book["current"]["retries"] == 1 and not book["batches"]
    assert store.get_node("n01").status is NodeStatus.SCREENING
    r = store.get_run("n01:screen:11")
    assert r.attempt == 0 and r.void_type is VoidType.INFRA            # the driver's owed retry takes it
    assert _tick(ctx, store) == []                                       # waits for the re-runs
    for i in range(1, 16):
        store.update_node(f"n{i:02d}", status=NodeStatus.VOID)
    (act,) = _tick(ctx, store)                                           # retries spent: close, loudly
    (res,) = store.get_state(D.K_DESIGN)["batches"]
    assert res["error"].startswith("no estimable effect") and "no estimable effect" in act
    assert all(not e["estimable"] for e in res["effects"]) and "no estimate" not in D.lines(store)
    assert store.events("design_no_estimate")


def test_a_batch_closed_singular_is_reanalysed_once_from_its_point_nodes(ctx, store):
    """The live d2: closed as {error: 'singular matrix'} with no rows / points recorded. The next tick
    recovers its matrix from the d1ptNN nodes' config diffs and re-analyses the scored rows."""
    cur = _batch(ctx, store, crash={i for i, r in enumerate(D.matrix(4, 16)) if r[3] > 0})
    store.set_state(D.K_DESIGN, {"current": None, "batches": [
        {"batch": 1, "knobs": cur["knobs"], "stacked": "", "error": "singular matrix"}]})
    assert D.rebuild(store, store.get_state(D.K_DESIGN)["batches"][0]) == cur
    (act,) = _tick(ctx, store)
    assert "re-analysed" in act and "train.k3" in act
    (res,) = store.get_state(D.K_DESIGN)["batches"]
    assert res["reanalysed"] and res["prior_error"] == "singular matrix" and res["n"] == 16
    eff = {e["key"]: e for e in res["effects"]}
    assert not eff["train.k3"]["estimable"] and "high scored 0/8 points (crashed 8)" in eff["train.k3"]["reason"]
    assert all(eff[f"train.k{j}"]["estimable"] for j in range(3))
    assert _tick(ctx, store) == [] and store.get_state(D.K_DESIGN)["batches"] == [res]   # once


def test_reanalysis_runs_beside_a_running_batch_and_leaves_it_alone(ctx, store):
    cur = _batch(ctx, store, crash={i for i, r in enumerate(D.matrix(4, 16)) if r[3] > 0})
    running = {**cur, "batch": 2, "points": {"1": "n99"}}                # an unclosed point: still running
    store.put_node(Node(id="n99", task_id=ctx.task_id, commit="c", parent_id=ROOT_NODE_ID, strategy=Strategy.DESIGN,
                        operator=None, family=DESIGN_FAMILY, mechanism_tag="d2pt01", diff_hash="h",
                        status=NodeStatus.SCREENING))
    store.set_state(D.K_DESIGN, {"current": running, "batches": [
        {"batch": 1, "knobs": cur["knobs"], "stacked": "", "error": "singular matrix"}]})
    (act,) = _tick(ctx, store)
    book = store.get_state(D.K_DESIGN)
    assert "d1 re-analysed" in act and book["current"] == running and book["batches"][0]["reanalysed"]
    assert _tick(ctx, store) == [] and store.get_state(D.K_DESIGN)["current"] == running



# ------------------------------------------------------------------ crash guard (rv-19#0, rv-26#0)
D3_TAIL = """Traceback (most recent call last):
  File "dataloader.py", line 197, in __init__
    self.local_aug = v2.Compose([v2.RandomResizedCrop(train["local_size"], scale=tuple(data["strength"]), antialias=True)])
TypeError: 'float' object is not iterable"""


def test_smoke_rows_cover_every_non_root_level_heaviest_first():
    knobs = [{"key": f"k{j}", "low": 1, "high": 2} for j in range(5)] + [{"key": "train.local_size", "low": 128, "high": 192}]
    base = {kb["key"]: kb["low"] for kb in knobs}
    rows = D.matrix(6, 16)
    got = D.smoke_rows(knobs, rows, base, list(range(1, 16)))
    assert rows[got[0]][5] == 1                                          # the heaviest point first
    assert {j for i in got for j in range(6) if rows[i][j] > 0} == set(range(6)) and len(got) <= 2
    assert D.smoke_rows(knobs, rows, {**base, **{kb["key"]: kb["high"] for kb in knobs}}, [0]) == [0]  # low moved
    assert D.smoke_rows(knobs, rows, base, []) == []


def test_every_batch_is_smoked_and_a_training_crash_withdraws_it_and_bans_the_level(tmp_path, monkeypatch):
    """d3: no memory knob, so no smoke, and 32/32 points crashed on one bad level. Now every batch is
    smoked (the real entrypoint), a traceback naming a knob withdraws the batch before any point
    launches, bans that knob's level, and the next batch is designed without it."""
    from loop.contracts import SmokeResult
    ctx, driver, store = _driver(tmp_path)
    adapter = driver.adapter
    real = adapter.smoke
    bad = lambda wt: (yaml.safe_load(open(f"{wt.path}/config.yaml")) or {}).get("aug", {}).get("strength") == 0.25  # noqa: E731
    monkeypatch.setattr(adapter, "smoke", lambda wt, **kw: SmokeResult(False, "smoke exited rc=1", D3_TAIL)
                        if bad(wt) else real(wt, **kw))
    driver.tick()
    D.request(store, [{"key": "train.lr", "high": 0.0005}, {"key": "train.weight_decay", "high": 0.025},
                      {"key": "aug.strength", "high": 0.25}, {"key": "train.warmup", "high": 10},
                      {"key": "loss.temperature", "high": 0.2}])
    acts = driver.tick()
    book = store.get_state(D.K_DESIGN)
    (first,) = book["batches"]
    assert first["aborted"] and "TypeError" in first["error"] and any("withdrawn before launch" in a for a in acts)
    assert book["bad_levels"] == [{"key": "aug.strength", "level": 0.25, "batch": 1, "why": first["error"]}]
    d1 = [n for n in store.nodes() if n.mechanism_tag.startswith("d1pt")]
    assert d1 and all(n.status is NodeStatus.REJECTED and not store.runs(node_id=n.id) for n in d1)
    assert store.events("design_smoke")
    for _ in range(3):
        driver.tick()
    cur = store.get_state(D.K_DESIGN)["current"]
    assert cur["batch"] == 2 and "aug.strength" not in [kb["key"] for kb in cur["knobs"]]
    assert len(cur["knobs"]) == 4 and cur["smoke"]["status"] == "passed"


def test_a_smoke_that_fails_outside_training_still_releases_the_batch(ctx, store):
    from loop.contracts import SmokeResult
    store.set_state(D.K_DESIGN, {"current": {"batch": 1, "knobs": [], "rows": [], "points": {"1": "nA"},
                                             "smoke": {"point": "nA", "queue": [], "mitigation": "", "status": "pending"}},
                                 "batches": []})
    (act,) = D.smoke_done(store, SmokeResult(False, "smoke exited rc=124", "sbatch: queued too long"), [])
    assert "batch released" in act and not D.held(store) and not store.get_state(D.K_DESIGN).get("bad_levels")


def test_a_design_smoke_that_got_no_slot_is_smoked_again_then_released(ctx, store):
    """A queue timeout / preemption is no verdict: the batch stays held and is re-smoked (placed afresh) up to
    MAX_SMOKE_UNAVAILABLE times, then released as before."""
    from loop.contracts import SmokeResult
    store.set_state(D.K_DESIGN, {"current": {"batch": 1, "knobs": [], "rows": [], "points": {"1": "nA"},
                                             "smoke": {"point": "nA", "queue": [], "mitigation": "", "status": "pending"}},
                                 "batches": []})
    res = SmokeResult(False, "smoke_unavailable: queue timeout after 3600s pending on idle", "", unavailable=True)
    for i in range(D.MAX_SMOKE_UNAVAILABLE):
        (act,) = D.smoke_done(store, res, [])
        assert "re-smoking" in act and D.smoke_due(store) == ("nA", "") and D.held(store) == {"nA"}
    (act,) = D.smoke_done(store, res, [])
    assert "batch released" in act and not D.held(store)


def test_a_level_that_crashed_while_its_other_level_scored_is_never_designed_again(ctx, store):
    """d2 (jepa_pred_width 0 -> -1: high 0/16) and d6 (local_size 192 OOM: high 0/8)."""
    cur = _batch(ctx, store, crash={i for i, r in enumerate(D.matrix(4, 16)) if r[3] > 0})
    store.set_state(D.K_DESIGN, {"current": cur, "batches": []})
    _tick(ctx, store)
    book = store.get_state(D.K_DESIGN)
    assert [(e["key"], e["level"]) for e in book["bad_levels"]] == [("train.k3", 2.0)]
    store.set_state(D.K_DESIGN, {**book, "bad_levels": book["bad_levels"] + [{"key": "train.lr", "level": 0.002}]})
    text = "train:\n  lr: 0.001\n  weight_decay: 0.05\n"
    got = D.pick_knobs(ctx, store, SimpleNamespace(nodes={}), text, {"train.lr": 0.001, "train.weight_decay": 0.05},
                       requested=[{"key": "train.lr", "high": 0.002}, {"key": "train.lr", "high": 0.003},
                                  {"key": "train.weight_decay", "high": 0.025}])
    assert [(kb["key"], kb["high"]) for kb in got] == [("train.lr", 0.003), ("train.weight_decay", 0.025)]
    for i in range(1, 16):                                               # everything crashed: blame no level
        for sd in (11, 12):
            store.update_run(f"n{i:02d}:screen:{sd}", status=RunStatus.CRASH, scores={})
    assert D._dead_levels(store, {**cur, "points": {k: v for k, v in cur["points"].items() if k != "0"}}) == []
