"""v3-c6a: code-picked crossovers (pairs + the stacking-leg filter), the blind holdout, store read connections + the
ideate sub-phase timers, TEST runs first on primary, the eval-only null control's own trigger, the meta A/B minimum
information + not-backtestable knobs."""

from __future__ import annotations

import dataclasses

from loop import tick as TK
from loop.contracts import NodeStatus
from loop.core import data as CD
from loop.core import model as M

from .test_c5_waves import _BankPost, _child
from .test_port import _two_epochs


# -- 1b: the stacking-leg filter ----------------------------------------------------------------------------------------
class _ZPost(_BankPost):
    def __init__(self, beliefs, ns, epoch, sd=0.001):
        super().__init__(beliefs, ns, epoch)
        self._v = sd * sd

    def belief(self, nid):
        return self._b[nid], self._v


def _bank_env(ctx, tmp_path, beliefs):
    sim, store = _two_epochs(ctx, tmp_path)
    car = "inc@new"
    commit = store.get_node(car).commit
    for i, nid in enumerate(beliefs):
        _child(store, nid, commit, car, 20.0 + i, f"f{i}")
    core = TK.ModuleCore()
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    store.set_state(core.K_BANK + car, {"knobs": [], "lines": [], "tau2": 2e-6, "effects": 0, "used": "w000",
                                        "v": core.RETRO_V})
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    post = _ZPost(beliefs, {n: 3 for n in beliefs}, led.epoch)
    return store, core, tick, led, post, car


def _legs(bank):
    return sorted(b.knob.key for b in bank if b.source == "stacking")


def test_stacking_legs_skip_failed_and_vetoed_nodes_and_relax_to_z_min(ctx, tmp_path):
    # z = m / .001: a 1.5, b 1.2, c 0.8, d 0.5
    store, core, tick, led, post, car = _bank_env(ctx, tmp_path, {"a": 0.0015, "b": 0.0012, "c": 0.0008, "d": 0.0005})
    assert _legs(core._carrier_bank(tick, led, post, car, [], {})) == ["a", "b"]           # two at z >= 1
    acts: list[str] = []
    assert _legs(core._carrier_bank(tick, led, post, car, acts, {"b": "failed"})) == ["a", "c"]   # relaxed to .75
    assert any("relaxed to 0.75" in a for a in acts)
    store.update_node("c", status=NodeStatus.VETOED)
    assert _legs(core._carrier_bank(tick, led, post, car, [], {"b": "failed"})) == []      # 1 leg: no stacking wave
    tick.close()
    store.close()


# -- 5: owed TEST runs hold primary first ---------------------------------------------------------------------------
def test_owed_test_runs_reserve_primary_against_repools_churn_screens_and_retries(ctx, store):
    from loop import runner as R
    from loop.contracts import RunStatus, Stage

    from .test_runner import _Pending, _node, _pools
    book = R.RunBook(_pools(ctx, primary=2, idle=8, repool_restarts=2, repool_restarts_any=5), store, _Pending())
    node = _node(store, "n1")
    book.launch(node, Stage.BASELINE, 1)                              # primary: 1 of 2 slots held
    book.test_reserve = 1                                             # one owed TEST run
    assert book.pool(node, Stage.BASELINE, 2) == "idle"               # HIGH non-test: the last slot is the test's
    assert book.pool(node, Stage.SCREEN, 3) == "idle"
    assert book._open(False) == [] and book._open(False, test=True) == ["primary"]
    stuck = book.launch(node, Stage.BASELINE, 4)                      # spilled onto idle, preempted again and again
    book.poll()
    book.restarts[stuck.id] = 9
    book.repool_due.add(stuck.id)
    assert book.promote_stuck() == []                                  # neither into the reserved slot nor queued
    assert book._retry_home(dataclasses.replace(stuck, pool="primary")) == "idle"
    run = book.launch(node, Stage.CONFIRM, 100_000)                   # the owed test takes it
    assert run.pool == "primary" and book.test_reserve == 0
    store.update_run(run.id, status=RunStatus.DONE)
    assert book.pool(node, Stage.BASELINE, 5) == "primary"            # nothing owed: the old routing


def test_the_tick_counts_owed_test_runs_before_the_poll(ctx, tmp_path):
    from loop.core.seeds import SeedBlock
    from loop.runner import K_TEST_RESERVE
    sim, store = _two_epochs(ctx, tmp_path)
    core = TK.ModuleCore()
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    store.put_seed_block(SeedBlock("t1", CD.TEST, (100_500, 100_501), ("inc@new", "root"), "test inc@new", 40.0, None))
    tick._blocks = None
    acts: list[str] = []
    tick._reserve_tests(acts)
    assert tick.book.test_reserve == 4 == store.get_state(K_TEST_RESERVE) and any("test reserve" in a for a in acts)
    tick.close()
    store.close()


# -- 6: the eval-only null control's own trigger ----------------------------------------------------------------------
def test_the_null_control_starts_once_per_epoch_without_any_eligible_candidate(tmp_path):
    from loop import evalonly as EO
    from loop.contracts import Stage

    from .test_c5_evalonly_lane import lane, ledger, obs, run
    ln, store, clock = lane(tmp_path)
    data = ledger([run("p", "screen", 5, out="/ckpt/p5"), run("root", "baseline", 7, out="/ckpt/r7")],
                  [obs("p", "screen", 5, 0.6), obs("root", "baseline", 7, 0.6)])
    acts: list[str] = []
    out = ln.route(data, [], acts)                                    # nothing eval-only to route: the null control anyway
    assert sorted((s.node, s.stage, s.seed) for s in out) == [("p", Stage.CALIBRATE, 5), ("root", Stage.CALIBRATE, 7)]
    assert out[0].node == "root"                                      # the base root (no incumbent recorded) first
    assert store.state[EO.K_V3]["null"]["B"]["status"] == "running" and any("null control" in a for a in acts)
    store.state[EO.K_V3]["null"]["B"]["status"] = "passed"
    assert ln.route(data, [], []) == []                               # once per epoch
    ln2, store2, _ = lane(tmp_path, null_at_start=False)
    assert ln2.route(data, [], []) == [] and not (store2.state.get(EO.K_V3) or {}).get("null")


# -- 7: meta A/B minimum information; not-backtestable knobs ----------------------------------------------------------
def _ab_meta(ctx, store, tmp_path, n_sessions=10):
    from types import SimpleNamespace

    from loop.core import meta as MT
    meta = MT.Meta(dataclasses.replace(ctx, state_dir=str(tmp_path / "st")), store, locks=None)
    for v in (1, 2):
        store.put_config_version("prompt", v, parent=None, ts=0.0, diff={}, reason="", backtest=None, state="active")
    sessions = {f"r{i}": ("arm" if i % 2 else "control") for i in range(2 * n_sessions)}
    store.set_state(MT.K_AB, {"arm": 2, "control": 1, "share": 0.5, "started": 0.0, "sessions": sessions})
    for r in sessions:
        store.log("intake", request_id=r)
    nodes = {f"n{i}": SimpleNamespace(notes=f"request=r{i}") for i in range(2 * n_sessions)}
    led = SimpleNamespace(nodes=nodes, runs=[SimpleNamespace(node=n, gpu_h=1.0) for n in nodes])
    return MT, meta, led


def test_a_0_vs_0_prompt_ab_is_not_retired_and_the_screen_posterior_decides(ctx, store, tmp_path):
    from types import SimpleNamespace
    MT, meta, led = _ab_meta(ctx, store, tmp_path)
    assert "not enough information" in meta.judge_ab(led, [], None)           # 0 vs 0, no screens: keeps running
    assert store.get_state(MT.K_AB)["arm"] == 2
    est = {f"n{i}": {} for i in range(20)}
    post = SimpleNamespace(est=est, belief=lambda n: (0.002 if int(n[1:]) % 2 else 0.0005, 1e-6))   # arm screens better
    msg = meta.judge_ab(led, [], post)
    assert "ADOPTED" in msg and "screen posterior" in msg and not store.get_state(MT.K_AB)


def test_an_uninformative_ab_ends_inconclusive_after_the_session_cap(ctx, store, tmp_path):
    MT, meta, led = _ab_meta(ctx, store, tmp_path, n_sessions=100)           # 100 >= 10 x ab_min_sessions
    msg = meta.judge_ab(led, [], None)
    assert "INCONCLUSIVE" in msg and not store.get_state(MT.K_AB)
    assert next(r for r in store.config_versions("prompt") if r["v"] == 2)["state"] == "inconclusive"


def test_a_knob_the_replay_cannot_see_is_marked_not_backtestable_and_not_listed_again(ctx, store, tmp_path, monkeypatch):
    from loop.core import meta as MT
    meta = MT.Meta(dataclasses.replace(ctx, state_dir=str(tmp_path / "st")), store, locks=None)
    meta.points = [object()] * 20
    flat = {"accept": False, "reason": "replay value 0.01161 -> 0.01161 (gain +0.00000, 2 SE 0.00000)",
            "cur": {"value": 0.0116}, "new": {"value": 0.0116}, "se": {"value": 0.0, "rho": 0.0}}
    monkeypatch.setattr(MT, "backtest_config", lambda *a, **k: dict(flat))
    prop = {"kind": "config", "path": "ideation.fnf_every_gpu_h", "value": 700, "rationale": "r", "predicted_effect": "e"}
    out = meta.accept({"diagnosis": "d", "proposals": [prop]}, None)
    assert any("not backtestable" in x for x in out) and "ideation.fnf_every_gpu_h" in store.get_state(MT.K_NOBT)
    monkeypatch.setattr(MT, "backtest_config", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-run")))
    out = meta.accept({"diagnosis": "d", "proposals": [prop]}, None)            # refused without a replay
    assert any("not backtestable" in x for x in out)
    assert "- ideation.fnf_every_gpu_h =" not in meta.review_prompt(MT.MetaReport(ts=0.0, since=0.0))


# -- 3: store reads off the write lock; lock-wait timing ---------------------------------------------------------------
def test_reads_use_per_thread_connections_while_a_writer_holds_the_lock(ctx, tmp_path):
    import threading

    from loop.store import SqliteStore
    s = SqliteStore(tmp_path / "x.db", ctx.task_id)
    s.set_state("k", 1)
    got, done = [], threading.Event()

    def reader():
        got.append(s.get_state("k"))
        done.set()
    with s._lock:                                                    # a writer holds the lock ...
        threading.Thread(target=reader).start()
        assert done.wait(5.0)                                         # ... a read on another thread does not wait
    assert got == [1]
    s.set_state("k", 2)
    t = threading.Thread(target=lambda: got.append(s.get_state("k")))
    t.start()
    t.join()
    assert got[-1] == 2                                               # a committed write is visible to the readers
    from loop.core import factorial as F
    with F.atomic(s):                                                 # inside a transaction: its own writes are seen
        s.set_state("k", 3)
        assert s.get_state("k") == 3
    st = s.lock_stats()
    assert st["reads"] >= 2 and st["locked_reads"] >= 1 and st["lock_waits"] >= 2 and st["lock_wait_s"] >= 0.0
    s.close()


def test_a_tick_times_ideate_sub_phases_and_the_store_lock_wait(ctx, tmp_path):
    from .test_tick_core import _module_tick, _WithTests
    tick, store, _ = _module_tick(ctx, tmp_path, _WithTests())
    tick.step()
    ps = tick.phase_s
    assert {"store.lock_wait", "store.lock_max", "ideate.view", "ideate.record", "ideate.register",
            "ideate.engagement", "ideate.plateau", "ideate.fnf"} <= set(ps)
    assert all(v >= 0.0 for v in ps.values())
    tick.close()
    store.close()


# -- 2: the blind holdout ---------------------------------------------------------------------------------------------
def _summary(cls, seg, surv, rob, slide=0.7, auc=0.6):
    d = {"final_probe_ucla_lung_val_auc": slide, "final_probe_surgen_val_auc": auc}
    for t, v in cls.items():
        d.update({f"final_probe_{t}_{h}_val_f1": v for h in ("linear", "knn", "fewshot")})
    d.update({f"final_probe_{t}_seg_val_f1": v for t, v in seg.items()})
    d.update({f"final_probe_{t}_val_cindex": v for t, v in surv.items()})
    d.update({f"final_probe_{t}_val_cindex_alpha_fraction_0p1": 0.0 for t in surv})
    d.update({f"final_probe_pathorob_{t}_robustness": v for t, v in rob.items()})
    return d


def test_holdout_tasks_split_and_the_deterministic_pick():
    from loop.core import holdout as HO
    d = _summary({"a": 0.8, "b": 0.6}, {"p": 0.5, "q": 0.7, "r": 0.6}, {"x": 0.6, "y": 0.5}, {"c": 0.6, "t": 0.8})
    t = HO.tasks_of(d)
    assert set(t) == {"cls:a", "cls:b", "seg:p", "seg:q", "seg:r", "surv:x", "surv:y", "rob:c", "rob:t"}
    pick = HO.default_tasks(t)
    assert len(pick) == 4 and {x.split(":")[0] for x in pick} == {"cls", "seg", "surv", "rob"}
    assert pick == HO.default_tasks(reversed(list(t)))                    # order-free
    live = ["cls:" + x for x in ("bach", "bracs", "break_his", "crc", "esca", "mhist", "pcam", "spider_breast",
                                 "spider_colorectal", "spider_skin", "spider_thorax", "wilds")] + \
        ["seg:pannuke", "seg:segpath_epithelial", "seg:segpath_lymphocytes", "surv:cptac_pda_os", "surv:leopard_bcr",
         "rob:camelyon", "rob:tolkach_esca", ]
    assert HO.default_tasks(live) == ("cls:spider_colorectal", "rob:tolkach_esca", "seg:pannuke", "surv:leopard_bcr")
    vis, ho = HO.split(t, frozenset({"cls:a", "seg:p"}))
    assert abs(vis["cls"] - 0.6) < 1e-12 and abs(vis["seg"] - 0.65) < 1e-12
    assert abs(ho["cls"] - 0.8) < 1e-12 and ho["seg"] == 0.5 and set(ho) == {"cls", "seg"}


def _ho_ledger(tmp_path, gain_vis, gain_ho, n_seeds=4, pool=CD.SELECT):
    """root and c on TEST seeds 1..n; c gains ``gain_vis`` on the visible cls task, ``gain_ho`` on the held one."""
    from loop.core.seeds import SeedBlock
    nodes = {"root": CD.Node("root", None, None, "baseline", "baseline", "baseline", "root", 0.0, "", "", "B"),
             "c": CD.Node("c", "root", None, "explore", "f", "explore", "root", 1.0, "", "", "B")}
    runs, obs = [], []
    for nid, gv, gh in (("root", 0.0, 0.0), ("c", gain_vis, gain_ho)):
        for s in range(1, n_seeds + 1):
            d = tmp_path / f"{nid}-{s}"
            d.mkdir()
            summ = _summary({"a": 0.6 + gv, "h": 0.5 + gh}, {"p": 0.5, "q": 0.6}, {"x": 0.6, "y": 0.5},
                            {"c": 0.6, "t": 0.7})
            import json
            (d / "summary.json").write_text(json.dumps(summ))
            rid = f"{nid}:confirm:{s}"
            runs.append(CD.Run(rid, nid, s, "confirm", "done", 1.0, 0.0, 0.0, str(d), 0, "B"))
            axes = {"cls": 0.55 + (gv + gh) / 2, "seg": 0.55, "slide": 0.7, "auc": 0.6, "surv": 0.55, "rob": 0.65}
            obs.append(CD.Obs(rid, nid, s, "confirm", pool, axes, 0.6, 1.0, 0.0, 0, None, "B"))
    led = CD.Ledger(nodes, tuple(runs), tuple(obs), frozenset(range(1, n_seeds + 1)), ((0.0, "root"),),
                    bases=((0.0, "root", "B"),))
    blk = SeedBlock("t1", CD.TEST, tuple(range(1, n_seeds + 1)), ("c", "root"), "test c", 0.0, 5.0,
                    ({"node": "c", "vs": "root", "kind": "pass", "mean": 0.003},))
    return led, blk


def test_the_visible_ledger_drops_held_tasks_and_keeps_the_official_panel(tmp_path):
    from loop.core import holdout as HO
    led, _ = _ho_ledger(tmp_path, 0.01, 0.0)
    cache = HO.TaskCache(tmp_path / "c.jsonl", 4)
    assert cache.update(led) == 8 and cache.update(led) == 0
    vis, miss = HO.visible(led, cache, frozenset({"cls:h", "seg:p"}))
    o = next(x for x in vis.obs if x.node == "c")
    assert miss == 0 and abs(o.axes["cls"] - 0.61) < 1e-12 and o.axes["seg"] == 0.6 and o.panel == 0.6
    assert o.axes["slide"] == 0.7                                       # a one-task axis: the official value
    assert len(HO.TaskCache(tmp_path / "c.jsonl").t) == 8                # persisted


def test_the_audit_compares_visible_and_holdout_gains_and_alerts_on_overfit(tmp_path):
    from loop.core import holdout as HO
    led, blk = _ho_ledger(tmp_path, 0.01, 0.001)
    cache = HO.TaskCache(None, 1)
    cache.update(led)
    [row] = HO.audit(led, [blk], cache, frozenset({"cls:h"}))
    assert row["node"] == "c" and row["n"] == 4 and abs(row["visible"] - 0.01) < 1e-9 and abs(row["holdout"] - 0.001) < 1e-9
    s = HO.summarize([row] * 5, 2.0, 5)
    assert s["alert"] and s["ratio"] == 10.0 and len(s["series"]) == 5
    assert not HO.summarize([row] * 4, 2.0, 5)["alert"]                # < min_nodes
    fair = dict(row, holdout=0.009)
    assert not HO.summarize([fair] * 6, 2.0, 5)["alert"]
    text = "\n".join(HO.report_text({"tasks": ["cls:h"], "rows": [row] * 5, "summary": s}))
    assert "ALERT" in text and "gap ratio over time" in text


def test_the_tick_selects_on_the_visible_ledger_and_judges_on_the_full_one(ctx, tmp_path):
    from .test_tick_core import _module_tick, _WithTests
    seen = {}

    class _Spy(_WithTests):
        def posterior(self, data, blocks):
            seen["post"] = data
            return super().posterior(data, blocks)

        def rank(self, data, blocks, statuses, post, free_slots, llm_free):
            seen["rank"] = data
            return super().rank(data, blocks, statuses, post, free_slots, llm_free)

        def derive(self, data, blocks, noise):
            seen["derive"] = data
            return super().derive(data, blocks, noise)
    tick, store, _ = _module_tick(ctx, tmp_path, _Spy())
    marker = {}

    def vis(led):
        marker["v"] = dataclasses.replace(led, obs=tuple(led.obs))
        return marker["v"]
    tick.holdout.visible = vis
    tick.step()
    assert seen["post"] is marker["v"] and seen["rank"] is marker["v"] and tick.vis is marker["v"]
    assert seen["derive"] is not marker["v"] and tick.data is not marker["v"]
    tick.close()
    store.close()


# -- 1: the code-picked crossover planner -------------------------------------------------------------------------------
def _pairs_env(monkeypatch, legs_spec, epoch="E2"):
    """legs_spec: node -> (parent, base, composite m, n, {axis: (raw delta, z)})."""
    from types import SimpleNamespace

    from loop.core import pairs as PR
    nodes = {"root": CD.Node("root", None, None, "baseline", "baseline", "baseline", "root", 0.0, "", "", epoch),
             "inc": CD.Node("inc", "root", None, "explore", "x", "explore", "root", 1.0, "", "", epoch)}
    for nid, (parent, base, *_rest) in legs_spec.items():
        nodes[nid] = CD.Node(nid, parent, None, "explore", f"fam_{nid}", "explore", "root", 2.0, "", "", base)
    led = CD.Ledger(nodes, (), (), frozenset(), ((0.0, "root"),), bases=((0.0, "root", epoch),))
    est = {nid: {M.COMPOSITE: M.Estimate(m, 1e-6, n, n)} for nid, (_, _, m, n, _) in legs_spec.items()}
    post = SimpleNamespace(est=est, epoch=epoch, near=frozenset(), belief=lambda nid: (legs_spec[nid][2], 1e-6))

    def axes(post_, led_, src, shrink_c=1.0):
        ax = legs_spec[src][4]
        g = {a: CD.OFFICIAL_WEIGHTS[a] * d * max(0.0, 1 - 1 / (z * z)) for a, (d, z) in ax.items()}
        return g, {a: d for a, (d, z) in ax.items()}, {a: z for a, (d, z) in ax.items()}
    monkeypatch.setattr(PR, "_axes", axes)
    return PR, led, post


def test_the_planner_pairs_a_cls_gainer_with_a_rob_slide_repairer_across_epochs(ctx, monkeypatch):
    """The 10-13 acceptance case as a fixture: 4 global crops on the old incumbent (cls +.012, seg +.005, rob -.008,
    slide -.005) x fewer local crops on an early base (rob +.019, slide +.017, a small cls loss)."""
    PR, led, post = _pairs_env(monkeypatch, {
        "g4": ("old_inc", "E1", 0.0012, 3, {"cls": (0.012, 4.0), "seg": (0.005, 2.0), "rob": (-0.008, -2.5),
                                            "slide": (-0.005, -1.2)}),
        "l9": ("root", "E0", -0.0008, 4, {"rob": (0.019, 5.0), "slide": (0.017, 3.0), "cls": (-0.004, -1.0)}),
        "c2": ("inc", "E2", 0.0010, 3, {"cls": (0.008, 3.0), "rob": (-0.004, -2.0)}),        # same dominant axis as g4
        "bad": ("inc", "E2", 0.0030, 3, {"seg": (0.02, 5.0)}),
    })
    got = PR.plan(post, led, ctx, {"bad": "failed"}, incumbent="inc", used={})
    keys = [p.key for p in got]
    assert keys[0] == "g4|l9" and all("bad" not in k for k in keys) and "c2|g4" not in keys
    best = got[0]
    assert best.dom == ("cls", "rob") and best.kinds == ("composite", "axis") and best.repair > 0 and best.comp > 0
    assert "c2|l9" in keys                                             # another cls-gainer x the repairer
    # a used pair is never reused; an axis leg that repairs nothing is no partner
    assert "g4|l9" not in [p.key for p in PR.plan(post, led, ctx, {}, incumbent="inc", used={"g4|l9": {}})]


def test_config_only_pairs_merge_and_code_pairs_go_to_the_ideation_side(ctx, store, monkeypatch):
    from types import SimpleNamespace
    PR, led, post = _pairs_env(monkeypatch, {
        "g4": ("inc", "E2", 0.0012, 2, {"cls": (0.012, 4.0), "rob": (-0.008, -2.5)}),
        "l9": ("inc", "E2", 0.0004, 3, {"rob": (0.019, 5.0)}),
        "code": ("inc", "E2", 0.0015, 3, {"seg": (0.01, 4.0), "cls": (-0.002, -1.0)}),
    })
    diffs = {"g4": {"train.global_views": 4}, "l9": {"train.local_views": 9}, "code": None}
    inc_cfg = {"train.global_views": 2, "train.local_views": 14}
    got = PR.plan(post, led, ctx, {}, incumbent="inc", inc_cfg=inc_cfg, config_of=diffs.get, used={})
    merge = next(p for p in got if p.key == "g4|l9")
    assert merge.config == {"train.global_views": 4, "train.local_views": 9}
    assert all(p.config is None for p in got if "code" in p.key)
    PR.bind(ctx, store, statuses={}, config_of=diffs.get, inc_cfg=inc_cfg)
    view = SimpleNamespace(post=post, led=led, incumbent="inc")
    a, b, why = PR.next_pair(view)
    assert "code" in (a, b) and "pred" in why
    assert not store.get_state(PR.K_USED)                               # v3-c6r: planning a brief burns nothing
    assert PR.next_pair(view)[:2] == (a, b)
    assert PR.claim(store, a, b, "brief", "req-1") and not PR.claim(store, a, b, "brief", "req-2")
    assert PR.pair_key(a, b) in store.get_state(PR.K_USED)              # claimed: never handed out twice
    nxt = PR.next_pair(view)
    assert nxt is None or PR.pair_key(*nxt[:2]) != PR.pair_key(a, b)


def test_a_config_only_pair_is_built_as_a_queued_merge_on_the_incumbent(ctx, store):
    from types import SimpleNamespace

    from loop.contracts import Node, Operator, Strategy
    from loop.core import pairs as PR
    store.ensure_root("r0")
    for nid in ("inc", "a", "b"):
        store.put_node(Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id="root", strategy=Strategy.EXPLORE,
                            operator=Operator.DRAFT, family=f"f{nid}", mechanism_tag=nid, diff_hash=nid,
                            status=NodeStatus.SCREENING))
    built = []
    adapter = SimpleNamespace(config_rel="configs/main.yaml", dry_build=lambda c, lv, kn, df, name: (
        built.append((c, dict(lv))) or ("sha-merge", "")))
    pl = PR.Planner(ctx, store, adapter, lambda: 5.0)
    p = PR.Pair("a", "b", 0.004, 0.003, 0.001, 0.0016, ("cls", "rob"), ("composite", "axis"),
                {"train.global_views": 4, "train.local_views": 9}, "why")
    led = CD.Ledger({}, (), (), frozenset(), ((0.0, "root"),))
    acts = pl._merges(led, {}, "inc", store.get_node("inc"), [p])
    [(commit, levels)] = built
    assert commit == "c-inc" and levels == {"train.global_views": 4, "train.local_views": 9} and acts
    nid = store.get_state(PR.K_USED)["a|b"]["node"]
    n = store.get_node(nid)
    assert n.status is NodeStatus.QUEUED and n.parent_id == "inc" and n.second_parent_id == "b" and "pair_of=a,b" in n.notes
    assert store.get_state(f"config_diff:{nid}") == {"train.global_views": 4, "train.local_views": 9}
