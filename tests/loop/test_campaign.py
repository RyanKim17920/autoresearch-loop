"""DIFFICULT-TUNE CAMPAIGNS (loop/campaign.py): eligibility, the search space, the computed rounds (1 seed per point
on the node's own screen seeds, paired vs its parent), the OOM fix, the selection-aware graduation onto fresh confirm
seeds only, the retro backfill epoch and the status / heartbeat lines.
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from loop import campaign as CP
from loop import tune as MT
from loop.adapters import flatten
from loop.contracts import NodeStatus, Stage, design_point
from loop.spec import CampaignCfg
from tests.loop.test_tune import AUG, _driver, _moonshot, _post, _run_until, _snap, _with

SWAP = "brief=moonshot init=timm/vit_small_patch16 mtype=architecture_addon"


@pytest.fixture
def sim(ctx, store):
    from loop.adapters.sim import SimAdapter
    return SimAdapter(ctx)


def _cfg(sim, node):
    wt = sim.prepare_worktree(parent_commit=node.commit, name=f"look-{node.id}")
    text = (Path(wt.path) / "config.yaml").read_text()
    return text, flatten(yaml.safe_load(text))


def test_eligibility_is_init_swap_objective_code_or_a_declared_campaign(ctx, store, sim):
    swap = _moonshot(ctx, store, sim, "n0001", code=False, notes=SWAP)
    plain = _moonshot(ctx, store, sim, "n0002", code=False)
    obj = _moonshot(ctx, store, sim, "n0003", notes="brief=moonshot mtype=loss_objective")
    numbers = _moonshot(ctx, store, sim, "n0004", code=False, notes="brief=moonshot mtype=loss_objective")
    decl = _moonshot(ctx, store, sim, "n0005", code=False)
    store.set_state("ideation:n0005", {"campaign": True})
    assert CP.why(ctx, store, swap).startswith("init swap") and CP.why(ctx, store, plain) == ""
    assert CP.why(ctx, store, obj) == "new objective / loss" and CP.why(ctx, store, numbers) == ""
    assert CP.why(ctx, store, decl) == "declared campaign"
    from loop.contracts import Strategy
    cross = _moonshot(ctx, store, sim, "n0006", strategy=Strategy.CROSSOVER, notes="brief=x mtype=loss_objective")
    assert CP.why(ctx, store, cross) == ""                                     # a crossover of known parts: no
    assert CP.rank("declared campaign") < CP.rank("init swap (x)") < CP.rank("new objective / loss")
    store.update_node(swap.id, notes=swap.notes + " tune_of=n0009")              # a graduate never campaigns again
    assert CP.why(ctx, store, store.get_node(swap.id)) == ""


def test_default_space_round1_spread_and_declared_ranges(ctx, store, sim):
    m = _moonshot(ctx, store, sim, "n0001", code=False, notes=SWAP)
    text, base = _cfg(sim, m)
    sp = CP.space(ctx, store, m, text, base)
    assert [kb["key"] for kb in sp] == ["train.lr", "train.warmup", "train.weight_decay"]   # only keys it has
    lr = sp[0]
    assert lr["scale"] == "log" and lr["low"] == pytest.approx(base["train.lr"] / 4)
    specs = CP.round1(ctx, sp, [11, 12])
    assert len(specs) == ctx.cfg.campaign.round1_points == 8
    lrs = [s["levels"]["train.lr"] for s in specs if "train.lr" in s["levels"]]
    assert [round(x / base["train.lr"], 2) for x in lrs] == [0.25, 0.63, 1.59, 4.0]   # log-spaced around the parent
    assert [s["seed"] for s in specs] == [11, 12] * 4 and all(len(s["levels"]) == 1 for s in specs)
    assert CP._between(lr, lrs[0], [base["train.lr"], *lrs]) == pytest.approx(
        [(lrs[0] * lrs[1]) ** 0.5, lrs[0] / (lrs[1] / lrs[0])], rel=1e-3)                # an edge: one step out
    store.set_state("ideation:n0001", {"campaign": True, "search_space": [
        {"key": "loss.temperature", "low": 0.05, "high": 0.4, "scale": "log"}, {"key": "nope", "low": 1, "high": 2}]})
    (kb,) = CP.space(ctx, store, m, text, base)
    assert kb["key"] == "loss.temperature" and kb["declared"] and CP.spread(kb, 4)[0] == 0.05
    assert CP._between(kb, 0.05, [0.1, 0.4]) == [pytest.approx(0.07071, rel=1e-3)]  # no ctx: a declared range holds
    assert CP._between(kb, 0.05, [0.1, 0.4], ctx) == pytest.approx([0.07071, 0.025], rel=1e-3)   # the edge: past it
    assert CP._between(kb, 0.4, [0.2], ctx) == pytest.approx([0.2828, 0.8], rel=1e-3)          # ... within the
    assert CP._between(kb, 0.8, [0.4], ctx) == [pytest.approx(0.5657, rel=1e-3)]                # hard bounds [0, 1]
    assert CP.oom_fix(ctx, text, {**base, "train.batch_size": 64}) == {}           # not a settable key here
    assert CP.oom_fix(ctx, text, base) == {}


INPUT = {"input_families": ("augmentation",), "input_code": "def masked_view",
         "input_space": ({"key": "train.weight_decay", "high": 0.2, "scale": "log"},
                         {"key": "loss.temperature", "high": 0.4, "scale": "linear"},
                         {"key": "nope", "high": 1.0})}


def test_a_declared_knob_at_its_range_end_is_kept_and_a_dropped_one_is_logged(ctx, store, sim):
    """Live c8: n2058 at layer decay 1.0 / patch-embed mult 1.0 declared 0.8..1.0 / 0.2..1.0 and searched only LR --
    the range end equal to the node's value read as "no move" and the knob vanished without a trace."""
    m = _moonshot(ctx, store, sim, "n0001", code=False, notes=SWAP)
    _, base = _cfg(sim, m)
    wd, lr = base["train.weight_decay"], base["train.lr"]
    store.set_state("ideation:n0001", {"campaign": True, "search_space": [
        {"key": "train.weight_decay", "low": wd / 4, "high": wd, "scale": "log"},       # its value = the top end
        {"key": "train.lr", "low": lr, "high": lr * 4, "scale": "log"},                 # ... = the bottom end
        {"key": "nope", "low": 1, "high": 2}]})
    text, base = _cfg(sim, m)
    dropped = []
    sp = CP.space(ctx, store, m, text, base, dropped)
    assert [kb["key"] for kb in sp] == ["train.weight_decay", "train.lr"]
    assert dropped == [{"key": "nope", "why": "not in the node's config"}]
    assert CP.spread(sp[0], 2) == pytest.approx([wd / 4, wd / 2], rel=1e-3)          # 2 levels that move it
    assert CP.spread(sp[1], 2) == pytest.approx([lr * 2, lr * 4], rel=1e-3)
    # a running campaign planned before the fix (only LR) gets the lost knob in a follow-up round
    assert [kb["key"] for kb in CP.missing(ctx, store, {"space": [sp[1]], "base": m.id}, text, base)] == \
        ["train.weight_decay"]
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    from tests.loop.test_tune import _screened
    _screened(ctx, store, m.id)
    assert CP.enqueue(store, m.id, "test")
    acts = CP.tick(ctx, store, sim, _snap({m.id: _post(m.id, 0.3)}), new_id=lambda: next(ids), now=1e12)
    assert "dropped nope (not in the node's config)" in acts[0], acts
    assert [e["key"] for e in store.events("campaign_knob_dropped")] == ["nope"]


def test_an_input_side_change_also_searches_the_input_space_and_round1_scales_with_the_knobs(ctx, store, sim):
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, campaign=CampaignCfg(**INPUT)))
    m = _moonshot(ctx, store, sim, "n0001", code=False, notes=SWAP)      # family augmentation, an init swap
    obj = _moonshot(ctx, store, sim, "n0002", code=False, notes="brief=moonshot mtype=loss_objective")
    store.update_node(obj.id, family="objective")
    code = _moonshot(ctx, store, sim, "n0003", notes="brief=moonshot mtype=loss_objective")    # adds masked_view
    store.update_node(code.id, family="objective")
    assert CP.input_side(ctx, store, m) == "init swap" and CP.input_side(ctx, store, store.get_node(obj.id)) == ""
    assert CP.input_side(ctx, store, store.get_node(code.id)) == "input-side code"
    store.update_node(m.id, notes="brief=moonshot")
    assert CP.input_side(ctx, store, store.get_node(m.id)) == "family augmentation"
    text, base = _cfg(sim, m)
    keys = [kb["key"] for kb in CP.space(ctx, store, m, text, base)]
    assert keys == ["train.lr", "train.warmup", "train.weight_decay", "loss.temperature"]
    wd, temp = CP.space(ctx, store, m, text, base)[2:]
    assert wd["declared"] and wd["high"] == 0.2 and wd["low"] == pytest.approx(base["train.weight_decay"] * (
        base["train.weight_decay"] / 0.2) ** 0.5, rel=1e-3)                 # the input space replaced the default wd
    assert temp["high"] == 0.4 and temp["low"] == pytest.approx(0.2 - 0.1)   # half the upward span below (0.2)
    obj = store.get_node(obj.id)
    assert [kb["key"] for kb in CP.space(ctx, store, obj, text, base)] == ["train.lr", "train.warmup",
                                                                           "train.weight_decay"]
    one = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, campaign=CampaignCfg(**INPUT, max_knobs=1)))
    assert [kb["key"] for kb in CP.space(one, store, m, text, base)] == ["train.lr", "train.weight_decay",
                                                                         "loss.temperature"]   # on top of max_knobs
    sp = CP.space(ctx, store, m, text, base)
    assert CP.r1_size(ctx, sp) == 10 and len(CP.round1(ctx, sp, [11, 12])) == 10   # 2 x 4 knobs + 2 > 8
    assert {k for s in CP.round1(ctx, sp, [11, 12]) for k in s["levels"]} == set(keys)   # every knob explored
    assert CP.budget(ctx, {"space": sp}) == pytest.approx(20.0 * 10 / 8)
    assert CP.r1_size(ctx, range(20)) == 16 and CP.r1_size(ctx, range(2)) == 8


def test_an_incumbent_entry_seeds_one_config_node_with_a_declared_campaign_per_incumbent(ctx, store, sim):
    entry = {"tag": "no_decay", "levels": {"train.weight_decay": 0.0001, "nope": 1},
             "search_space": [{"key": "train.weight_decay", "low": 0.00005, "high": 0.05, "scale": "log"},
                              {"key": "train.lr", "factor": 2.0, "scale": "log"}]}
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, campaign=CampaignCfg(incumbent=(entry,))))
    store.ensure_root(sim.root_commit())
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    snap = _snap({})
    acts = CP.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12)
    assert any("no_decay node n0100 on the incumbent root" in a for a in acts)
    n = store.get_node("n0100")
    assert n.status is NodeStatus.SCREENING and n.parent_id == "root" and not design_point(n)
    st = store.get_state("ideation:n0100")
    assert st["campaign"] and [r["key"] for r in st["search_space"]] == ["train.weight_decay", "train.lr"]
    assert st["search_space"][1]["low"] == pytest.approx(0.0005)              # LR x1/2..x2 around the incumbent's
    (q,) = CP.book(store)["queue"]
    assert q["base"] == "n0100" and CP.rank(q["why"]) == 0                    # ranked with the declared campaigns
    assert not [a for a in CP.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 1) if "no_decay" in a]
    from tests.loop.test_tune import _screened
    _screened(ctx, store, "n0100")
    store.update_node("n0100", status=NodeStatus.SCREENED_OUT)                # closed: its campaign starts
    acts = CP.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 2)
    assert any(a.startswith("campaign c1 for n0100") for a in acts)
    assert [kb["key"] for kb in CP.book(store)["current"]["n0100"]["space"]] == ["train.weight_decay", "train.lr"]


def test_a_campaign_runs_rounds_on_its_own_seeds_and_graduates_onto_fresh_confirm_seeds_only(tmp_path):
    ctx, driver, store, adapter = _driver(tmp_path, p_promote=0.999)
    driver.tick()                                                   # the lane's first run stamps ``since``
    assert CP.book(store)["since"] > 0
    m = _moonshot(ctx, store, adapter, "n0001", status=NodeStatus.SCREENING, cfg=AUG, code=False, notes=SWAP)
    assert _run_until(driver, store, lambda: bool(CP.book(store)["done"]), n=250)
    (c,) = CP.book(store)["done"]
    own = set(c["seeds"])
    assert c["base"] == m.id and own == {r.seed for r in store.runs(node_id=m.id, stage=Stage.SCREEN)}
    pts = [store.get_node(i) for i in c["points"]]
    assert c["round"] == 3 and len(pts) > ctx.cfg.campaign.round1_points
    assert all(design_point(p) and p.parent_id == m.parent_id and f"campaign_of={m.id}" in p.notes for p in pts)
    for p in pts:                                                   # ONE seed each, on the node's own seeds
        assert len({r.seed for r in store.runs(node_id=p.id)}) == 1 and {r.seed for r in store.runs(node_id=p.id)} <= own
    assert not store.runs(node_id=m.id, stage=Stage.PARENT)          # paired vs the parent's existing runs
    assert not MT.book(store)["tuned"] and not MT.book(store)["laddered"]   # the campaign's node alone
    assert c["best"].startswith("train.lr=") and "shrunk" in c["summary"] and c["summary"].startswith("campaign c1 n0001")
    g = store.get_node(c["tuned"])
    assert g is not None and f"campaign_of={m.id}" in g.notes and f"tune_of={m.id}" in g.notes
    assert g.parent_id == m.parent_id and g.confirm_alt is not None
    ev = [e for e in store.events() if e["event"] == "node_transition" and e.get("node_id") == g.id]
    assert [e["to"] for e in ev][:2] == ["screening", "confirming"]  # straight to confirm: no screen / promote gate
    assert _run_until(driver, store, lambda: store.get_node(g.id).status is not NodeStatus.CONFIRMING, n=150)
    runs = store.runs(node_id=g.id, stage=Stage.CONFIRM)
    from loop.seeds import planned_confirm                          # nanopath: 3 (confirm_max_seeds), the fake: 6
    assert runs and {r.stage for r in store.runs(node_id=g.id)} <= {Stage.CONFIRM, Stage.AUDIT}
    assert len(runs) <= planned_confirm(ctx.cfg, []) and not {r.seed for r in runs} & own
    # its confirm seeds COUNT (live 10-08: n2086, no selection seed, was never in the ledger -> "seeds spent E=1 n=0")
    g = store.get_node(g.id)
    (close,) = [e for e in store.events() if e["event"] == "node_transition" and e.get("node_id") == g.id
                and e.get("frm") == "confirming"]
    assert g.confirm_e is not None and " n=0" not in close["reason"], close["reason"]
    post = driver._refit().nodes[g.id]            # and it has a posterior: the beats-the-incumbent gate applies
    assert post.confirm_estimate is not None and post.increment.n == len(runs)
    assert "campaign c1" in CP.lines(store)


def test_points_crash_out_of_memory_is_rebuilt_once_and_status_lines(ctx, store, sim):
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, campaign=CampaignCfg(round1_points=2,
                                                                                         round1_max=2)))
    m = _moonshot(ctx, store, sim, "n0001", code=False, notes=SWAP)
    from tests.loop.test_tune import _screened
    _screened(ctx, store, m.id)
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    snap = _snap({m.id: _post(m.id, 0.3)})
    assert CP.enqueue(store, m.id, "test") and not CP.enqueue(store, m.id, "again")
    acts = CP.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12)
    (cur,) = CP.book(store)["current"].values()
    assert "round 1, 2 point(s)" in acts[0] and CP.note(store) == "campaigns=1(2pts) "
    assert "running" in CP.lines(store) and CP.claims(ctx, store, m)
    # an OOM crash with no batch-size key: the point stays void (sim configs have none)
    from loop.contracts import Run, RunStatus
    p = next(iter(cur["points"]))
    store.update_node(p, status=NodeStatus.CRASHED)
    store.put_run(Run(id=Run.make_id(p, Stage.SCREEN, 11), task_id=store.task_id, node_id=p, seed=11,
                      stage=Stage.SCREEN, status=RunStatus.CRASH, log_tail="CUDA out of memory"))
    for q in list(cur["points"])[1:]:
        store.update_node(q, status=NodeStatus.SCREENED_OUT)
    acts = CP.tick(ctx, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 1)
    assert "out of memory -> void" in acts[0] and CP.book(store)["current"][m.id]["oom"] == [p]


def test_a_rounds_last_open_points_run_high_and_may_take_primary(ctx, store, sim):
    """Live 10-08: 80 campaign points ran LOW / preemptible-only and no round closed for 5 h (c1's round 3 waited on
    one point). Once at most max(2, half the round) are open, those points are stragglers: HIGH, primary allowed."""
    from loop.driver import RunBook
    from tests.loop.test_driver import _PoolAdapter, _pools_ctx
    from tests.loop.test_tune import _screened
    ctx = _pools_ctx(dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, campaign=CampaignCfg(
        round1_points=4, round1_max=4))), primary=2, idle=4)
    m = _moonshot(ctx, store, sim, "n0001", code=False, notes=SWAP)
    _screened(ctx, store, m.id)
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    assert CP.enqueue(store, m.id, "test")
    CP.tick(ctx, store, sim, _snap({m.id: _post(m.id, 0.3)}), new_id=lambda: next(ids), now=1e12)
    pts = list(CP.book(store)["current"][m.id]["points"])
    book = RunBook(ctx, store, _PoolAdapter())
    assert len(pts) == 4 and CP.stragglers(store) == set() and all(CP.campaign_point(store.get_node(p)) for p in pts)
    p0 = store.get_node(pts[0])
    assert not book.high(p0, Stage.SCREEN, 11) and book.pool(p0, Stage.SCREEN, 11) == "idle"     # LOW: idle only
    for p in pts[:2]:
        store.update_node(p, status=NodeStatus.SCREENED_OUT)
    assert CP.stragglers(store) == set(pts[2:]) and not CP.campaign_point(m)
    last = store.get_node(pts[3])
    assert book.high(last, Stage.SCREEN, 11) and book.pool(last, Stage.SCREEN, 11) == "primary"


def test_the_plan_schema_carries_a_declared_campaign_and_its_search_space(ctx):
    from loop.agents import ideas as IDEAS
    from loop.agents.researcher import finish_schema
    props = finish_schema(ctx)["properties"]
    assert props["campaign"]["type"] == "boolean" and props["search_space"]["type"] == "array"
    assert {"campaign", "search_space"} <= set(IDEAS.IDEA_SCHEMA["properties"])
    cfg = {"train.lr": 0.001, "aug.strength": 0.5, "model.name": "vit"}
    got = IDEAS.validate_space(cfg, [{"key": "train.lr", "low": 1e-4, "high": 1e-2},
                                     {"key": "aug.strength", "low": 0.0, "high": 1.0, "scale": "linear"},
                                     {"key": "aug.strength", "low": 0.0, "high": 1.0},          # log needs low > 0
                                     {"key": "model.name", "low": 1, "high": 2}, {"key": "nope", "low": 1, "high": 2}])
    assert got == [{"key": "train.lr", "low": 1e-4, "high": 1e-2, "scale": "log"},
                   {"key": "aug.strength", "low": 0.0, "high": 1.0, "scale": "linear"}]


def test_a_graduate_confirms_only_when_it_also_beats_the_incumbent():
    from loop.contracts import Stage as St
    from tests.loop.test_driver import _node, _post, _roots, _run, _step
    seeds = [101, 102, 103]
    own = [_run("n1", St.CONFIRM, s) for s in seeds]
    node = dataclasses.replace(_node(NodeStatus.CONFIRMING), notes="tune_of=n0 campaign_of=n0")
    assert _step(node, own, _roots(seeds), _post(p_inc=0.3), confirm=(50.0, 3)).status is NodeStatus.NOT_CONFIRMED
    assert _step(node, own, _roots(seeds), _post(p_inc=0.95), confirm=(50.0, 3)).status is NodeStatus.CONFIRMED
    plain = _node(NodeStatus.CONFIRMING)                     # any other node: unchanged
    assert _step(plain, own, _roots(seeds), _post(p_inc=0.3), confirm=(50.0, 3)).status is NodeStatus.CONFIRMED


def test_knobs_the_node_became_eligible_for_get_one_follow_up_round_around_the_best(ctx, store, sim, monkeypatch):
    """A campaign started before campaign.input_space was configured: at its next round boundary the knobs its node
    is now eligible for join its space with ONE follow-up round at their low / high around its best config."""
    small = dict(round1_points=2, round1_max=2)
    ctx0 = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, campaign=CampaignCfg(**small)))
    m = _moonshot(ctx0, store, sim, "n0001", code=False, notes=SWAP)          # an init swap: input-side
    from tests.loop.test_tune import _screened
    _screened(ctx0, store, m.id)
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    snap = _snap({m.id: _post(m.id, 0.3)})
    CP.enqueue(store, m.id, "test")
    CP.tick(ctx0, store, sim, snap, new_id=lambda: next(ids), now=1e12)
    cur = CP.book(store)["current"][m.id]
    before = [kb["key"] for kb in cur["space"]]
    assert "loss.temperature" not in before and len(cur["points"]) == 2
    for p in cur["points"]:
        store.update_node(p, status=NodeStatus.SCREENED_OUT)
    best = next(iter(cur["points"].values()))
    monkeypatch.setattr(CP, "rows", lambda c, st, sn, cu: {best["cfg"]: [0.01, 0.001, 1.0], "1x": [0.0, 0.001, 1.0]})
    ctx1 = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, campaign=CampaignCfg(**small, **INPUT)))
    acts = CP.tick(ctx1, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 1)
    cur = CP.book(store)["current"][m.id]
    assert acts == [f"campaign c1 for {m.id}: follow-up round over ['loss.temperature'] (missing from its space) "
                    f"around its best {best['cfg']}: 2 point(s); budget {CP.budget(ctx1, cur):g} GPU-h"]
    assert cur["followup"] == ["loss.temperature"] and [kb["key"] for kb in cur["space"]] == before + ["loss.temperature"]
    new = [p for p in cur["points"].values() if "loss.temperature" in p["levels"]]
    assert sorted(p["levels"]["loss.temperature"] for p in new) == [0.1, 0.4] and cur["round"] == 1
    assert all({k: v for k, v in p["levels"].items() if k != "loss.temperature"} == best["levels"] for p in new)
    assert store.events("campaign_followup")[0]["knobs"] == ["loss.temperature"]
    for p in cur["points"]:
        store.update_node(p, status=NodeStatus.SCREENED_OUT)
    acts = CP.tick(ctx1, store, sim, snap, new_id=lambda: next(ids), now=1e12 + 2)
    assert not any("follow-up" in a for a in acts)                            # once: the knob is in the space now


def test_a_best_whose_shrunk_delta_clears_the_confirm_mde_graduates_though_its_raw_misses_the_null_max(ctx, store, sim):
    """Live c3 (2026-10-08): n1511's best read shrunk +0.0032 vs n0655 (raw +0.0046 under null q0.9 +0.0076) and was
    discarded. Fresh confirmation guards the winner's curse; the shrunk estimate >= the confirm MDE graduates."""
    from types import SimpleNamespace
    from loop import legs as LG
    from loop.contracts import PANEL
    m = _moonshot(ctx, store, sim, "n0001", status=NodeStatus.SCREENING, cfg=AUG, code=False, notes=SWAP)
    snap = SimpleNamespace(sigma={PANEL: 0.002758}, nodes={}, incumbent_id="")
    bar = LG.shrunk_bar(ctx, store, snap)
    cur = {"id": "c3", "base": m.id, "parent": "root", "round": 3, "points": {}}
    rs = {"a": [bar * 1.2, 0.01, 1.0], "b": [0.0, 0.01, 1.0], "1x": [0.0, 0.01, 1.0]}
    eff = {k: {"shrunk": s, "se": 0.01} for k, s in (("a", bar * 1.05), ("b", 0.0), ("1x", 0.0))}
    why, tuned, best, _ = CP.graduate(ctx, store, snap, cur, rs, eff, None, 0.0, dry=True)
    assert best == "a" and tuned == "(dry)" and "null gate not passed" in why
    eff["a"]["shrunk"] = bar * 0.9
    why, tuned, _, _ = CP.graduate(ctx, store, snap, cur, rs, eff, None, 0.0, dry=True)
    assert not tuned and "no config credibly beats" in why and "< bar" in why
