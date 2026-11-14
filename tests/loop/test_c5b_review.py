"""v3-c5b pre-deploy review fixes (M1, M2, M4, S1-S8)."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from loop.contracts import Stage
from loop.core import data as CD
from loop.core import model as M
from tests.loop.test_c5_evalonly_lane import Adapter, lane, lnode, run

# ------------------------------------------------------------------------------------------------ M1


def flood_ledger(n_flood: int):
    """root + 4 ordinary children (real training noise) + ``n_flood`` eval-only children whose every run sits
    1e-5 above root's same-seed run (a readout re-scored on root's own checkpoint)."""
    rnd = random.Random(5)
    nodes = {"root": lnode("root")}
    rs, os_ = [], []
    seed_fx = {s: rnd.gauss(0, 0.004) for s in range(1, 9)}
    root = {}
    for nid in ("root", "a", "b", "d", "e"):
        if nid != "root":
            nodes[nid] = lnode(nid, "root")
        mu = rnd.gauss(0, 0.002)
        for s in range(1, 9):
            y = 0.6 + mu + seed_fx[s] + rnd.gauss(0, 0.002)
            root.setdefault(s, y) if nid == "root" else None
            rs.append(run(nid, "screen", s))
            os_.append(CD.Obs(f"{nid}:screen:{s}", nid, s, "screen", CD.SELECT, {a: y for a in CD.AXES}, y, 1.0,
                              0.0, 0, None, "B"))
    eo = set()
    for k in range(n_flood):
        nid = f"eo{k}"
        nodes[nid] = lnode(nid, "root")
        for s in range(1, 9):
            y = root[s] + 1e-5
            rs.append(run(nid, "screen", s))
            os_.append(CD.Obs(f"{nid}:screen:{s}", nid, s, "screen", CD.SELECT, {a: y for a in CD.AXES}, y, 1.0,
                              0.0, 0, None, "B"))
            eo.add(f"{nid}:screen:{s}")
    led = CD.Ledger(nodes, tuple(rs), tuple(os_), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    return led, frozenset(eo)


def test_eval_only_flood_does_not_move_the_posterior_noise():
    from loop.core.acquire import Posterior
    clean, _ = flood_ledger(0)
    flood, eo = flood_ledger(30)
    base = Posterior(clean, []).fit("B", True)[0]           # the far class: every child here is a code change
    polluted = Posterior(flood, []).fit("B", True)[0]
    fixed = Posterior(flood, [], eo_runs=eo).fit("B", True)[0]
    assert polluted.rho[M.PANEL] != pytest.approx(base.rho[M.PANEL], abs=1e-3)      # the bug: the flood moves rho
    for t in (M.PANEL, M.COMPOSITE):
        assert fixed.rho[t] == pytest.approx(base.rho[t])
        assert fixed.sigma2[t] == pytest.approx(base.sigma2[t])
    assert Posterior(flood, [], eo_runs=eo).history().rho[M.PANEL] == pytest.approx(Posterior(clean, []).history().rho[M.PANEL])
    # the eval-only nodes still get their estimates (the dedicated eval-only noise / CUPED read them)
    assert "eo0" in Posterior(flood, [], eo_runs=eo).est


def test_eval_only_flood_does_not_move_the_test_sigma_d():
    from loop.tick import ModuleCore
    clean, _ = flood_ledger(0)
    flood, eo = flood_ledger(30)
    core = ModuleCore()
    core.classify(frozenset(), lambda c, i: False)
    blk = SimpleNamespace(seeds=())
    base = core.pair_fit(clean, "a", "root", blk)
    core.eo_runs = eo
    fixed = core.pair_fit(flood, "a", "root", blk)
    assert fixed.noise.rho[M.PANEL] == pytest.approx(base.noise.rho[M.PANEL])
    assert fixed.noise.sigma2[M.PANEL] == pytest.approx(base.noise.sigma2[M.PANEL])
    assert core.noise(flood).rho[M.PANEL] == pytest.approx(ModuleCore().noise(clean).rho[M.PANEL])
    core.eo_runs = frozenset()
    assert core.pair_fit(flood, "a", "root", blk).noise.rho[M.PANEL] != pytest.approx(base.noise.rho[M.PANEL],
                                                                                       abs=1e-3)


def test_lane_eval_runs_are_the_planned_runs(tmp_path):
    ad = Adapter()
    ln, store, _ = lane(tmp_path, ad)
    ln.ctx.cfg.evalonly = __import__("dataclasses").replace(ln.cfg, enabled=False)   # switched off: past runs stay out
    from loop import evalonly as EO
    store.set_state(EO.K_V3, {"elig": {"c": [True, "", False]}, "elig_v": 1, "null": {"B": {"runs": {"p:calibrate:3": "p:screen:3"}}}})
    ad.plan_eval("c", Stage.SCREEN, 1, {"commit": "x"})
    ad.plan_eval("p", Stage.CALIBRATE, 3, {"commit": "y"})
    obs = [SimpleNamespace(run_id=f"{n}:{st}:{s}", node=n, stage=st, seed=s)
           for n, st, s in (("c", "screen", 1), ("c", "screen", 2), ("p", "calibrate", 3), ("p", "screen", 3))]
    # an older rule's verdict is re-judged, but the node's past eval-only runs stay out of the fits
    assert ln._state()["elig"] == {} and ln._state()["planned"] == ["c"]
    assert ln.eval_runs(SimpleNamespace(obs=obs)) == {"c:screen:1", "p:calibrate:3"}


# ------------------------------------------------------------------------------------------------ M2


def test_launch_order_tests_then_primary_then_capped_wave_points(ctx, tmp_path, monkeypatch):
    import dataclasses
    from loop import tick as TK
    from loop.contracts import RunStatus
    from tests.loop.test_port import _two_epochs
    from tests.loop.test_waves import _bank
    sim, store = _two_epochs(ctx, tmp_path)
    small = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=2))
    core = TK.ModuleCore()
    core.knobs = {"inc@new": _bank()}
    tick = TK.Tick(small, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    first = tick._launch([TK.Action("wave", "inc@new", "factorial", 8.0, 1.0, 8)], led, {}, 50.0, [])
    assert len(first) == 2                                                  # 6 of the 8 points never ran
    for r in first:
        store.update_run(r, status=RunStatus.DONE, aggregate=0.67, scores={k: 0.67 for k in CD.PROBE_KEYS.values()},
                         gpu_h=1.0)
    tick.close()
    big = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    tick = TK.Tick(big, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 60.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    blk = SimpleNamespace(pool=CD.TEST, open=True, node_ids=("cand", "inc"), seeds=(), id="b")
    monkeypatch.setattr(tick, "blocks", lambda: [blk])
    tags = {"test": 101, "seed": 102, "fill": 103}

    def fake_act(a, data, statuses, now, acts):
        return [TK.RunSpec(f"{a.kind}-{a.node}", Stage.SCREEN, tags[a.kind])]
    monkeypatch.setattr(tick, "_act", fake_act)
    seen = []

    class Lane:
        def route(self, data, specs, acts):
            seen.extend(specs)
            return []
    monkeypatch.setattr(tick, "_eo", lambda: Lane())
    acts = []
    tick._launch([TK.Action("fill", "f", "", 1.0, 0.0, 1), TK.Action("seed", "s", "", 1.0, 1.0, 1),
                  dataclasses.replace(TK.Action("seed", "r", "", 1.0, 1.0, 1), pre=True)], led, {}, 60.0, acts)
    kinds = [s.node.split("-")[0] if "-" in s.node else "wave" for s in seen]
    assert kinds[:2] == ["test", "seed"]                                    # owed test runs, then the primary pass
    assert kinds[2:6] == ["wave"] * 4 and sorted(kinds[6:]) == ["fill", "seed"]   # 4 of 6 points, then relax / fill
    assert any("4 of 8 never-ran" in a for a in acts)                     # 6 points + 2 block controls (A2)
    tick.close()
    store.close()


# ------------------------------------------------------------------------------------------------ S2


def test_null_control_goes_first_and_waits_are_bounded(tmp_path):
    from loop import evalonly as EO
    from tests.loop.test_c5_evalonly_lane import ledger, obs, spec
    ln, store, clock = lane(tmp_path, source_wait_s=7200.0)
    data = ledger([run("p", "screen", 5, out="/ckpt/p5")], [obs("p", "screen", 5, 0.6)])
    other = spec("root", 9)                                            # an ordinary run (not eligible: root)
    out = ln.route(data, [other, spec("c", 5)], [])
    assert [(s.node, s.stage) for s in out] == [("p", Stage.CALIBRATE), ("root", Stage.SCREEN)]   # the null first
    clock[0] += EO.NULL_WAIT_S + 1                                     # the null control still pending
    out = [s for s in ln.route(data, [spec("c", 5)], []) if s.node != "p"]   # (the null run: not submitted here)
    assert [(s.node, s.seed) for s in out] == [("c", 5)] and ("c", "screen", 5) not in ln.adapter.plans
    # a source in flight: withheld at most WAIT_MAX_S (< source_wait_s 7200), measured from the first withholding
    ln2, store2, clock2 = lane(tmp_path, source_wait_s=7200.0)
    store2.state[EO.K_V3] = {"null": {"B": {"status": "passed"}}}
    data = ledger([run("p", "screen", 5, status="running", ts=0.0)], [])
    clock2[0] = 5000.0
    assert ln2.route(data, [spec("c", 5)], []) == []                   # first withheld at 5000 (source began at 0)
    clock2[0] = 5000.0 + EO.WAIT_MAX_S - 1
    assert ln2.route(data, [spec("c", 5)], []) == []
    clock2[0] = 5000.0 + EO.WAIT_MAX_S + 1
    assert [(s.node, s.seed) for s in ln2.route(data, [spec("c", 5)], [])] == [("c", 5)]


# ------------------------------------------------------------------------------------------------ S4


def test_eval_only_lane_continues_after_an_eval_only_incumbent(tmp_path):
    """p (now the incumbent) ran eval-only on root's checkpoints: its child c resumes the ORIGINAL checkpoint the
    plan of p's run resumed (same weights p's run scored), not 'no source'."""
    from loop import evalonly as EO
    from tests.loop.test_c5_evalonly_lane import ledger, obs, spec
    ln, store, _ = lane(tmp_path)
    store.state[EO.K_V3] = {"null": {"B": {"status": "passed"}}, "elig_v": EO.ELIG_V}
    ln.adapter.plan_eval("p", Stage.SCREEN, 5, {"commit": "c-p", "seed": 5, "resume": "/ckpt/r5/latest.pt",
                                                "source_run": "root:screen:5", "model": "root"})
    data = ledger([run("root", "screen", 5, out="/ckpt/r5"), run("p", "screen", 5, out="/eo/p5")],
                  [obs("root", "screen", 5, 0.6), obs("p", "screen", 5, 0.601)])
    out = ln.route(data, [spec("c", 5)], [])
    assert [(s.node, s.seed) for s in out] == [("c", 5)]
    plan = ln.adapter.plans[("c", "screen", 5)]
    assert plan["resume"] == "/ckpt/r5/latest.pt" and plan["source_run"] == "root:screen:5" and plan["model"] == "root"
    # a broken chain (the source run unknown): no source, the run trains
    ln.adapter.plans.pop(("c", "screen", 5))
    ln.adapter.plan_eval("p", Stage.SCREEN, 5, {"commit": "c-p", "source_run": "gone:screen:5"})
    out = ln.route(data, [spec("c", 5)], [])
    assert [(s.node, s.seed) for s in out] == [("c", 5)] and ("c", "screen", 5) not in ln.adapter.plans


# ------------------------------------------------------------------------------------------------ S5


def test_harm_guard_leaves_open_tests_and_the_incumbent_alone(tmp_path):
    from loop import evalonly as EO
    from loop.contracts import NodeStatus
    from tests.loop.test_c5_evalonly_lane import eo_data
    ln, store, _ = lane(tmp_path, harm_tol={"slide": 0.005}, harm_prob=0.8)
    store.state[EO.K_V3] = {"elig": {"c": [True, "", False]}, "elig_v": EO.ELIG_V}
    data = eo_data(ln, slide_d=-0.02)
    pure, harm = ln.prepare(data, [], protect={"c"})             # c is in an open TEST (or the incumbent)
    assert pure == {"c"} and harm == {} and store.nodes["c"].status is not NodeStatus.VETOED
    pure, harm = ln.prepare(data, [])                            # before TEST: held + vetoed
    assert "c" in harm and store.nodes["c"].status is NodeStatus.VETOED
