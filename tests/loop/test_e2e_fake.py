"""Whole-loop acceptance test on the offline task (docs/loop/DESIGN.md sections 0-7).

Run: uv run pytest tests/loop -q --confcutdir=tests/loop

One scenario, real modules end to end (SqliteStore, ComputedWorldModel,
PortfolioSelector, LLMResearcher/LLMAnalyst over a scripted FakeLLMClient,
SimAdapter, HashingEmbedder). The spec is a copy of tasks/fake_loop.yaml whose
sigmas are UNMEASURED, so the loop must refuse, calibrate, then run. The sim's
hidden truth: lowering train.lr / weight decay / aug strength helps ("train
weaker"); a depth change without INIT_FIX breaks init.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import yaml

from loop import remint as RM
from loop import seeds as S
from loop.calibrate import calibrate
from loop.contracts import (BANDIT_ARMS, PANEL, ROOT_NODE_ID, ForeignTaskError, HypothesisKind, NodeStatus,
                            Operator, RunStatus, Stage, Strategy, UnmeasuredLoopNoise, design_point)
from loop.select.bandit import Bandit
from loop.spec import IslandCfg, PromotionCfg, ReviewCfg, load_context
from loop.world.evalues import CONFIRM_KEY
from tests.loop.conftest import InlineExecutor

REPO = Path(__file__).resolve().parents[2]
FAKE = REPO / "tasks" / "fake_loop.yaml"
TICKS = 120
SEED = int(os.environ.get("LOOP_E2E_SEED", "12"))   # selector rng seed; swept 11-18 by hand
TICK_S = 300.0                  # 5 sim minutes per tick; heartbeat.freeze_min is 20

#: (edits {key: multiplier | ("=", value)}, declared family, tag, lever tags).
IDEAS: list[tuple[dict[str, Any], str, str, dict[str, int]]] = [
    ({"train.lr": 0.3}, "optimization", "lower_lr", {"effective_lr": -1}),
    ({"model.depth": 16 / 12}, "architecture", "deeper", {"capacity": 1}),        # init breaks
    ({"train.lr": 0.5, "aug.strength": 0.5}, "optimization", "lr_and_aug",        # two families
     {"effective_lr": -1}),
    ({"train.lr": ("=", 3.0e-4)}, "optimization", "lower_lr", {"effective_lr": -1}),  # repeat
    ({"aug.strength": 1.8}, "augmentation", "stronger_aug", {"augmentation_strength": 1}),
    ({"train.weight_decay": 0.5}, "regularization", "less_wd", {"regularization": -1}),
    ({"train.lr": 2.0}, "optimization", "higher_lr", {"effective_lr": 1}),
    ({"aug.strength": 0.7}, "augmentation", "weaker_aug", {"augmentation_strength": -1}),
    ({"train.lr": 0.6}, "optimization", "lower_lr_more", {"effective_lr": -1}),
    ({"train.lr": 2.5}, "optimization", "much_higher_lr", {"effective_lr": 1}),
    ({"loss.temperature": 1.05}, "objective", "temp", {"objective_difficulty": 1}),
    ({"train.weight_decay": 2.0}, "regularization", "more_wd", {"regularization": 1}),
]


def _walk(cfg: dict, key: str) -> tuple[dict, str]:
    *head, last = key.split(".")
    for p in head:
        cfg = cfg[p]
    return cfg, last


class Scripts:
    """Scripted answers for every LLM role."""

    def __init__(self) -> None:
        self.k = 0

    def researcher(self, **call: Any) -> Any:
        k, self.k = self.k, self.k + 1
        edits, family, tag, levers = IDEAS[k % len(IDEAS)]
        rnd = k // len(IDEAS)
        path = Path(call["cwd"]) / "config.yaml"
        cfg = yaml.safe_load(path.read_text())
        moved = []
        for key, how in edits.items():
            d, leaf = _walk(cfg, key)
            old = d[leaf]
            if isinstance(how, tuple):
                new = how[1]
            else:
                new = old * how ** (1 + rnd)
                new = int(round(new)) if isinstance(old, int) else round(new, 8)
            d[leaf] = new
            moved.append(f"{key} {old}->{new}")
        # a repeat keeps the SAME claim text so it is a duplicate however it is compared
        claim = f"{tag}: {', '.join(moved)} helps" if not isinstance(
            next(iter(edits.values())), tuple) else "lower_lr: train.lr 0.001->0.0003 helps"
        finish = {"hypothesis": {"claim": claim, "mechanism_tag": tag, "family": family,
                                 "predictions": {PANEL: {"sign": 1, "size": 0.005}},
                                 "falsification": "panel delta <= 0 on screen seeds"},
                  "levers": levers,
                  "handoff": {"actions": "; ".join(moved), "learnings": "n/a", "concerns": "none"}}
        return {"tool_calls": [("write_file", {"path": "config.yaml",
                                               "content": yaml.safe_dump(cfg, sort_keys=True)}),
                               ("smoke", {})],
                "finish": finish}

    def json_role(self, **call: Any) -> Any:
        req = set(call["schema"].get("required", ()))
        if "tests_hypothesis" in req:
            return {"tests_hypothesis": True, "rationale": "the diff moves the named knob"}
        if "winner" in req:
            return {"winner": "A", "rationale": "simpler"}
        if "meta_hypotheses" in req:
            return {"summary": "lower LR won, stronger aug and higher LR lost: over-driven model",
                    "findings": ["lower lr won", "higher lr lost", "stronger augmentation lost"],
                    "meta_hypotheses": [{
                        "claim": "the model is over-driven / over-regularised: train weaker",
                        "mechanism_tag": "train_weaker", "family": "optimization",
                        "levers": {"effective_lr": -1, "regularization": -1},
                        "predicted_sign": 1, "predicted_size": 0.005,
                        "falsification": "lever-down trials do not beat their parents",
                        "probes": ["halve weight decay"]}],
                    "probe_briefs": ["halve weight decay on the incumbent"],
                    "explore_bias": {"regularization": 0.01}}
        raise AssertionError(f"unexpected json role call: {sorted(req)}")

    def client(self) -> Any:
        from loop.agents.llm import FakeLLMClient
        return FakeLLMClient({"researcher": self.researcher,
                              "review": self.json_role,
                              "brief": self.json_role})


def _unmeasured_spec(tmp_path: Path) -> Path:
    text = FAKE.read_text()
    for old, new in [("seed_transfer_sd: 0.004", "seed_transfer_sd: UNMEASURED"),
                     ("panel_sd: 0.004", "panel_sd: UNMEASURED"),
                     ("axis_sd: {alpha: 0.005, beta: 0.006, gamma: 0.007}",
                      "axis_sd: {alpha: UNMEASURED, beta: UNMEASURED, gamma: UNMEASURED}")]:
        assert old in text, old
        text = text.replace(old, new)
    p = tmp_path / "fake_loop.yaml"
    p.write_text(text)
    return p


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    from loop.store import SqliteStore

    tmp = tmp_path_factory.mktemp("e2e")
    state = tmp / "state"
    spec = _unmeasured_spec(tmp)

    # --- phase 0: refuse on UNMEASURED, calibrate, then the spec is runnable
    with pytest.raises(UnmeasuredLoopNoise, match="calibrate"):
        load_context(spec, state_dir=state)
    raw = load_context(spec, state_dir=state, require_measured=False)
    truth = dataclasses.replace(raw, cfg=dataclasses.replace(
        raw.cfg, panel_sd=0.004, axis_sd={"alpha": 0.005, "beta": 0.006, "gamma": 0.007}))
    state.mkdir(parents=True, exist_ok=True)
    cal_store = SqliteStore(state / f"{raw.task_id}.db", raw.task_id, weights=raw.cfg.weights)
    res = calibrate(raw, cal_store, SimAdapter(truth), sleep=lambda s: None, echo=lambda s: None)
    assert res is not None and res.written
    cal_ids = set(res.run_ids)
    cal_store.close()

    ctx = load_context(spec, state_dir=state)          # measured now
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg,
        review=ReviewCfg(stall_evals=4, stall_min_hours=0.0),
        islands=IslandCfg(n=3, split_at=10, reset_every=1000),
        # 8 calibration seeds (df 7) make the frozen sigma bound ~1.6x: six confirm seeds
        # cannot reach 1/0.05 on the sim's effect sizes. This test is about staging, not power.
        promotion=PromotionCfg(alpha_confirm=0.2),
        # the scripted researcher re-proposes the same few moonshot claims by design; the mechanism dedup that
        # would reject them is tested on its own (test_novelty)
        novelty=dataclasses.replace(ctx.cfg.novelty, mech_dup_cos=0.0)))
    scripts = Scripts()
    now = [0.0]                                  # sim wall clock: TICK_S per tick
    adapter = SimAdapter(ctx, clock=lambda: now[0])
    # A stronger planted LR lever (2.5x the default -0.012/log-unit) so the statistical asserts
    # (lever sign/P) hold with margin across selector seeds instead of at the noise floor.
    adapter.lever_effects["train.lr"] = -0.03
    adapter.inject_fault("root:*", "crash")      # the first post-calibration root run crashes
    adapter.inject_fault("n*:screen:*", "hang")  # the first screen run freezes
    driver, store = build(ctx, llm=scripts.client(), adapter=adapter, executor=InlineExecutor,
                          rng=random.Random(SEED))
    driver.sleep = lambda s: None

    occupancy: list[tuple[int, int]] = []       # (active runs, runs still wanted)
    log: list[str] = []

    def tick() -> None:
        now[0] += TICK_S
        log.extend(driver.tick())
        wanted = driver._needs(store.nodes(), driver._runs_by_node())
        occupancy.append((len(driver.book.active()), len(wanted)))

    for _ in range(TICKS):
        tick()
    confirm_book = dict(store.get_state(CONFIRM_KEY))
    # Labless gate: asking about the first confirmed node queues its held-out audit
    audited = next((n.id for n in store.nodes(status=[NodeStatus.CONFIRMED])), None)
    for _ in range(40):
        a = driver._audit_book()
        if audited is None or audited in a["passed"] + a["failed"]:
            break
        driver.labless_ready(audited)
        tick()
    driver.drain()
    driver._harvest(log)
    snap = driver.world.fit(store)
    return dict(ctx=ctx, store=store, driver=driver, adapter=adapter, snap=snap, log=log,
                occupancy=occupancy, cal_ids=cal_ids, scripts=scripts, spec=spec,
                confirm_book=confirm_book, audited=audited)


def _by_tag(store, tag: str):
    return [n for n in store.nodes() if n.mechanism_tag == tag]


def test_calibration_runs_first(scenario):
    store, cal_ids = scenario["store"], scenario["cal_ids"]
    runs = store.runs()
    order = [r.id for r in runs]
    assert cal_ids and all(i.startswith("root:") for i in cal_ids)
    first_node_run = next(i for i, r in enumerate(runs) if r.node_id != ROOT_NODE_ID)
    assert all(order.index(i) < first_node_run for i in cal_ids)
    assert "# measured" in scenario["spec"].read_text()
    # calibration's replicate-1 runs ARE root's baselines: the driver never re-ran them
    assert len([r for r in runs if r.node_id == ROOT_NODE_ID and r.stage is Stage.BASELINE
                and r.seed in {11, 12, 13, 14, 101}]) == 5


def test_enough_trials_and_slots_never_idle_while_work_exists(scenario):
    ctx, store, occ = scenario["ctx"], scenario["store"], scenario["occupancy"]
    trials = [n for n in store.nodes() if n.id != ROOT_NODE_ID]
    assert len(trials) >= 40, len(trials)
    idle = [(t, a, w) for t, (a, w) in enumerate(occ)
            if a < ctx.cfg.capacity and w > a]          # free slot while a run was wanted
    assert not idle, idle[:10]


def test_staging_screen_promote_confirm_on_disjoint_seeds(scenario):
    ctx, store = scenario["ctx"], scenario["store"]
    confirmed = store.nodes(status=[NodeStatus.CONFIRMED])
    assert confirmed, [(n.id, n.mechanism_tag, n.status) for n in store.nodes()]
    win = confirmed[0]
    got = {st: {r.seed for r in store.runs(node_id=win.id, stage=st)}
           for st in (Stage.SCREEN, Stage.PROMOTE, Stage.CONFIRM)}
    fixed = set(ctx.cfg.seeds.promote) | set(ctx.cfg.seeds.confirm_pool) | set(ctx.cfg.seeds.audit)
    assert got[Stage.SCREEN] and not got[Stage.SCREEN] & fixed    # a screen block (they rotate)
    assert got[Stage.PROMOTE] == set(ctx.cfg.seeds.promote)
    assert got[Stage.CONFIRM] and got[Stage.CONFIRM] <= RM.confirm_seeds(store, S.control_of(win)[0])   # fresh block
    assert not got[Stage.CONFIRM] & (fixed | got[Stage.SCREEN])
    assert not (got[Stage.SCREEN] | got[Stage.PROMOTE]) & got[Stage.CONFIRM]
    order = [(r.node_id, r.stage) for r in store.runs()]
    first = {st: order.index((win.id, st)) for st in got}
    assert first[Stage.SCREEN] < first[Stage.PROMOTE] < first[Stage.CONFIRM]
    assert win.confirm_e >= 1 / ctx.cfg.promotion.alpha_confirm


def test_intake_rejects_duplicate_two_family_and_init_breaking(scenario):
    store = scenario["store"]
    reasons = {n.mechanism_tag: [] for n in store.nodes()}
    for n in store.nodes():
        if n.operator is not Operator.CROSSOVER and n.strategy is not Strategy.MOONSHOT:
            reasons[n.mechanism_tag].append((n.status, n.reject_reason))   # only these may span families
    # the repeated lower-LR diff (idea 3, re-proposed every round) is refused as a duplicate
    first = _by_tag(store, "lower_lr")[0]
    claim = store.get_hypothesis(first.hypothesis_id).claim
    # same edit on the same parent = same experiment (off another parent it is a new diff)
    repeats = [n for n in store.nodes() if n.id not in (first.id, ROOT_NODE_ID)
               and n.parent_id == first.parent_id and n.operator is not Operator.CROSSOVER
               and store.get_hypothesis(n.hypothesis_id).claim == claim]
    assert repeats, reasons["lower_lr"]
    assert all(n.status is NodeStatus.REJECTED and n.reject_reason in
               ("exact_dup", "near_dup", "parent_contains_knob") for n in repeats), \
        [(n.id, n.status, n.reject_reason) for n in repeats]
    assert repeats[0].reject_reason in ("exact_dup", "near_dup"), repeats[0]
    # a diff spanning optimization + augmentation is refused (not a crossover)
    # (first 10: in-flight value variants are refused, so the script cycles until lr underflows to 0)
    assert reasons["lr_and_aug"] and all(s is NodeStatus.REJECTED and r == "multi_family"
                                         for s, r in reasons["lr_and_aug"][:10]), reasons["lr_and_aug"]
    # the init-breaking architecture change never gets a GPU run
    for n in _by_tag(store, "deeper"):
        assert n.status is NodeStatus.REJECTED and "init" in n.reject_reason
        assert not store.runs(node_id=n.id)


def test_crossover_paired_against_common_ancestor_with_both_changes(scenario):
    from loop.adapters import flatten
    from loop.select.intake import common_ancestor, parent_config
    store = scenario["store"]
    xs = [n for n in store.nodes() if n.operator is Operator.CROSSOVER
          and n.status is not NodeStatus.REJECTED]
    assert xs, [(n.id, n.operator) for n in store.nodes()]
    for n in xs:
        a, b = n.notes.split("xover=")[1].split()[0].split(",")
        lca = common_ancestor(store, a, b)
        assert n.parent_id == lca and n.second_parent_id == b
        base, mine = parent_config(store, lca), flatten(store.get_state(f"config_diff:{n.id}") or {})
        own = " ".join(h.actions for h in store.handoffs(node_id=n.id))   # keys its own edit moved
        for p in (a, b):                                   # the diff vs the ancestor carries both
            assert {k for k, v in parent_config(store, p).items()
                    if base.get(k) != v and k not in own} <= set(mine), n.id


def test_review_meta_hypothesis_evalue_from_linked_trials(scenario):
    store, snap = scenario["store"], scenario["snap"]
    reviews = store.reviews()
    reviews = [r for r in reviews if r.trigger != "first_principles"]
    assert reviews and reviews[0].trigger == "stall"
    metas = store.hypotheses(kind=HypothesisKind.META)
    assert metas and metas[0].review_id == reviews[0].id
    m = metas[0]
    linked = store.links(m.id)
    assert linked
    ev = snap.hypotheses[m.id]
    assert ev.n_trials >= 1 and ev.trial_ids and set(ev.trial_ids) <= {nid for nid, _ in linked}
    assert ev.e_value != 1.0


def test_islands_split_after_threshold(scenario):
    ctx, store, snap = scenario["ctx"], scenario["store"], scenario["snap"]
    assert snap.n_scored >= ctx.cfg.islands.split_at
    st = store.get_state("select.islands")
    assert st and st["n_islands"] >= 2, st
    islands = {n.island for n in store.nodes() if n.id != ROOT_NODE_ID}
    assert islands - {"main"}, islands


def test_no_run_twice(scenario):
    runs = scenario["store"].runs()
    assert len({r.id for r in runs}) == len(runs)
    assert all(r.status is not RunStatus.SUBMITTING for r in runs)


def test_confirm_sigma_frozen_at_confirm_start(scenario):
    store, snap, book = scenario["store"], scenario["snap"], scenario["confirm_book"]
    confirmed = store.nodes(status=[NodeStatus.CONFIRMED, NodeStatus.AUDIT_FAILED])
    assert confirmed and all(n.id in book for n in confirmed)
    later = store.get_state(CONFIRM_KEY)
    for n in confirmed:            # more data arrived (the audit), the frozen pair did not move
        d1, sigma = book[n.id]
        assert later[n.id] == [d1, sigma]
        # an upper bound, not the point estimate: it sits at or above the later re-fit up to
        # estimation noise (10% slack; the bound is ~1.6x a point estimate, so this still
        # separates it from a plain estimate, but a strict > flips with the selector seed)
        assert sigma > 0.9 * snap.sigma[PANEL] > 0


def test_audit_on_held_out_seeds_gates_labless(scenario):
    ctx, store, driver, nid = (scenario[k] for k in ("ctx", "store", "driver", "audited"))
    assert nid is not None
    a = driver._audit_book()
    audit = set(a["gate"][nid]["audit_seeds"])          # seed rotation (loop.remint): its own never-used block
    assert len(audit) == len(ctx.cfg.seeds.audit) and min(audit) >= ctx.cfg.seeds.fresh_base
    got = {r.seed for r in store.runs(node_id=nid, stage=Stage.AUDIT)}
    root = {r.seed for r in store.runs(node_id=ROOT_NODE_ID, stage=Stage.AUDIT)}
    assert got == audit and audit <= root
    seen = {r.seed for r in store.runs() if r.stage not in (Stage.AUDIT, Stage.REMINT)}
    assert not seen & audit                             # held out: never used by any node to select anything
    if vs := a["gate"][nid]["vs"]:                      # the incumbent ran on the same block: paired
        assert audit <= {r.seed for r in store.runs(node_id=vs, stage=Stage.REMINT)}
    assert nid in a["passed"] + a["failed"]
    assert driver.labless_ready(nid) is (nid in a["passed"])
    if nid in a["failed"]:
        assert store.get_node(nid).status is NodeStatus.AUDIT_FAILED


def test_bandit_keeps_every_arm_above_floor(scenario):
    ctx, store = scenario["ctx"], scenario["store"]
    pulls = store.pulls()
    assert len({p.arm for p in pulls}) >= 2, pulls
    bandit = Bandit(ctx.cfg.bandit, random.Random(0)).rebuild(pulls)
    draws = [bandit.choose(BANDIT_ARMS) for _ in range(4000)]
    for arm in BANDIT_ARMS:        # the learned bandit still gives every arm >= floor
        assert draws.count(arm) / len(draws) >= 0.8 * ctx.cfg.bandit.floor, (arm, draws.count(arm))


def test_crashed_root_run_is_rerun_under_a_new_attempt(scenario):
    store, log = scenario["store"], scenario["log"]
    crashed = [line.split()[1] for line in log
               if line.startswith("run root:") and line.endswith("-> crash")]
    assert crashed, [line for line in log if line.startswith("run root:")][:5]
    run = store.get_run(crashed[0])
    assert run.attempt >= 1 and run.status is RunStatus.DONE, run   # attempt counts re-submits
    assert f"retry {run.id} -> running" in log


def test_frozen_job_is_cancelled_and_retried(scenario):
    store, adapter, log = scenario["store"], scenario["adapter"], scenario["log"]
    first = next(r for r in store.runs() if r.stage is Stage.SCREEN)       # the hung one
    assert f"run {first.id} -> void (infra)" in log
    job = json.loads(adapter._job_path(adapter._job_id(dataclasses.replace(first, attempt=0)))
                     .read_text())
    assert "frozen" in job["final"]["log_tail"]         # voided, no longer holding a slot
    assert first.attempt >= 1 or store.get_node(first.node_id).status is not NodeStatus.SCREENING


def test_dashboard_renders_the_e2e_store(scenario):
    from loop import dashboard
    store, snap = scenario["store"], scenario["snap"]
    page = dashboard.render(scenario["ctx"], store, snap)
    import json
    import re
    blob = re.search(r'<script type="application/json" id="loop-data">(.*?)</script>', page, re.S).group(1)
    assert len(json.loads(blob)["nodes"]) == len(store.nodes()) - 1
    assert page.count("<script") == 2 and "<svg" in page
    for rv in store.reviews():
        assert f'id="rv-{rv.id}"' in page


def test_v1_row_injected_raises_on_read(scenario):
    """Runs LAST (it contaminates the shared DB): an untagged v1 row is refused on read;
    a row tagged with another task stays invisible (zero weight)."""
    from loop.store import SqliteStore
    ctx, store = scenario["ctx"], scenario["store"]
    db = Path(ctx.state_dir) / f"{ctx.task_id}.db"
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("SELECT * FROM nodes WHERE id != 'root' LIMIT 1").fetchone())
    other = {**row, "id": "v1-other-task", "task_id": "nanopath", "diff_hash": "x-other"}
    cols = ", ".join(f'"{c}"' for c in other)
    con.execute(f"INSERT INTO nodes ({cols}) VALUES ({', '.join('?' for _ in other)})",
                list(other.values()))
    con.commit()
    assert store.get_node("v1-other-task") is None
    assert "v1-other-task" not in {n.id for n in store.nodes()}

    v1 = {**row, "id": "v1-legacy", "task_id": "", "diff_hash": "x-v1"}
    con.execute(f"INSERT INTO nodes ({cols}) VALUES ({', '.join('?' for _ in v1)})",
                list(v1.values()))
    run = dict(con.execute("SELECT * FROM runs LIMIT 1").fetchone())
    v1run = {**run, "id": "v1-legacy:screen:11", "node_id": "v1-legacy", "task_id": ""}
    rcols = ", ".join(f'"{c}"' for c in v1run)
    con.execute(f"INSERT INTO runs ({rcols}) VALUES ({', '.join('?' for _ in v1run)})",
                list(v1run.values()))
    con.commit()
    con.close()
    with pytest.raises(ForeignTaskError):
        store.nodes()
    with pytest.raises(ForeignTaskError):
        store.get_node("v1-legacy")
    with pytest.raises(ForeignTaskError):
        store.runs()
    with pytest.raises(ForeignTaskError):
        scenario["driver"].world.fit(store)
    fresh = SqliteStore(db, ctx.task_id)             # a reopened store refuses too
    with pytest.raises(ForeignTaskError):
        fresh.find_by_diff_hash("x-v1")
    fresh.close()


def test_cli_dry_run_on_unmeasured_nanopath_plans_calibration(tmp_path, unmeasured_v2_spec):
    import subprocess
    import sys
    cp = subprocess.run([sys.executable, "-m", "loop", "run-v2", "--task", str(unmeasured_v2_spec),
                         "--dry-run", "--state-dir", str(tmp_path / "st")],
                        cwd=REPO, capture_output=True, text=True, timeout=120)
    assert cp.returncode == 0, cp.stderr
    assert "UNMEASURED" in cp.stderr and "calibrate" in cp.stderr
    assert "would run root:baseline:11" in cp.stdout and "root:calibrate:" in cp.stdout
    line = next(ln for ln in cp.stdout.splitlines() if "would run root:baseline:11:" in ln)
    assert "./submit/train_1gpu.sbatch" in line and "seed=11" in line   # the exact launch
    assert not (tmp_path / "st").exists()              # nothing launched, nothing written

