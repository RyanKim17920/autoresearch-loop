"""v3 review fixes on loop/v3 8660e9c (the two adversarial reviews of 10-14): the kind bandit (K1), the GPU split
and the discovery reservation (K2), prerequisite ports (K3), probe ill-conditioning repairs (K4), the pairs ranking
(K5), the seed-share gate (K6), bar.ranked's TEST seeds (K8), the wave journal's design version (O1) and the
preemption backoff (O3)."""

from __future__ import annotations

import dataclasses

from loop import tick as TK
from loop.core import data as CD
from loop.core import factorial as F

from .test_port import _two_epochs
from .test_review_c4j_locks import xenv  # noqa: F401  (the fixture)
from .test_waves import _bank


# ---------------------------------------------------------------------------------------------- O1 wave journals
def test_design_falls_back_to_the_single_block_layout_of_an_old_journal():
    kb = F.Knob("dino.adam_beta2_end", 0.999, 0.99999, "lin")
    w = F.design("w017", "n2678", [kb], [100147])            # c1b75a9's journal: k 1, ONE seed, no blocks
    assert len(w.points) == 8 and set(w.blocks) == {0} and w.seeds == (100147,)
    new = F.design("w018", "n2678", [kb], [1, 2, 3, 4])      # the A2 layout: a block per replicate, dups out
    assert len(new.points) == 8 and set(new.blocks) == {0, 1, 2, 3}
    rows, blocks = F.matrix(1)
    same = F.design("w019", "n2678", [kb], [1, 2, 3, 4], blocks=blocks)
    assert same == dataclasses.replace(new, id="w019")


def test_a_w017_shaped_journal_resumes_and_is_never_quarantined(ctx, tmp_path):
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    tick = TK.Tick(ctx, store, sim, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    store.set_state(TK.K_WAVE_BUILD, {"wave": "w017", "n": 17, "carrier": "inc@new", "bank": [_bank()[0].to_json()],
                                      "seeds": [100147], "diffs": {}, "used": [], "ts": 40.0, "tries": 0,
                                      "pre": False})
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    acts: list[str] = []
    launched = tick._launch([], led, {}, 60.0, acts)
    assert store.get_state(TK.K_WAVE_BUILD) is None and store.get_state("wave:w017")["seeds"] == [100147]
    assert len([r for r in launched if r.startswith("w017.")]) == 8 and not tick.quarantined("wave.resume")
    assert not store.events("quarantine")
    tick.close()
    store.close()


# ---------------------------------------------------------------------------------------------- K5 pairs
def test_pairs_predict_with_the_stack_prior_no_curse_and_cap_a_legs_pairs(ctx, monkeypatch):
    from .test_c6a import _pairs_env
    from loop.core.acquire import stack_mean
    axes = ("cls", "seg", "slide", "auc", "rob")
    spec = {"hub": ("inc", "E2", 0.0040, 4, {"cls": (0.012, 4.0)})}
    for i, a in enumerate(axes[1:]):                     # four partners on other axes: hub would be in all 4
        spec[f"p{i}"] = ("inc", "E2", 0.0020 - 1e-4 * i, 4, {a: (0.006, 3.0)})
    PR, led, post = _pairs_env(monkeypatch, spec)
    info: dict = {}
    got = PR.plan(post, led, ctx, {}, incumbent="inc", used={}, info=info)
    for p in got:                                        # the stack prior, never the additive sum; no curse shift
        la, lb = post.belief(p.a)[0], post.belief(p.b)[0]
        assert p.pred == stack_mean([(la, p.a), (lb, p.b)]) < la + lb
        assert p.value >= p.pred and f"pred {p.pred:+.5f} (stack prior" in p.why
    hub = [p for p in got if "hub" in (p.a, p.b)]
    assert len(hub) == 2 and info["leg_capped"] >= 2     # pairs.leg_cap 2


# ---------------------------------------------------------------------------------------------- K6 the seed gate
def test_the_one_seed_outlier_gate_keeps_power_flat_over_n_at_the_live_config():
    import random

    from loop.core import acquire as A
    from loop.core import status as ST
    from loop.spec import load_context
    ctx = load_context("tasks/nanopath_v2.yaml", require_measured=False)
    A.configure(ctx.cfg)
    assert ST.SHARE_MIN_N == 3 and ST.OUTLIER_Z == 1.8 and ST.TRIM_MIN_N == 4
    W = {a: 1 / 6 for a in CD.AXES}

    def obs(node, seed, y):
        return CD.Obs(f"{node}:screen:{seed}", node, seed, "screen", "select", {a: y for a in CD.AXES}, y, 1.0, 0.0, 0)
    rng, sd = random.Random(11), 0.0034
    power = {}
    for n in (3, 4, 5):
        par, ok = [obs("p", s, 0.68) for s in range(n)], 0
        for _ in range(3000):
            ok += ST.seed_artifact([obs("c", s, 0.68 + rng.gauss(0.004, sd)) for s in range(n)], par, W, sd_d=sd)[0]
        power[n] = ok / 3000
    assert all(0.74 <= p <= 0.88 for p in power.values()) and max(power.values()) - min(power.values()) < 0.06
    par = [obs("p", s, 0.68) for s in (100098, 100099, 100100)]           # n2693 at its test entry (10-13)
    n2693 = [obs("c", s, 0.68 + d) for s, d in zip((100098, 100099, 100100), (0.00062, 0.00074, 0.00858))]
    ok, why = ST.seed_artifact(n2693, par, W, sd_d=0.00339)
    assert not ok and "seed 100100 is a one-seed outlier" in why


# ---------------------------------------------------------------------------------------------- K7 robust cells
def test_a_cells_prior_location_is_robust_like_its_spread():
    from loop.core import model as M
    bulk = [M.PriorPoint(("c", "f", "explore"), 0.001 + 1e-5 * i, 4e-6) for i in range(20)]
    tail = [M.PriorPoint(("c", "f", "explore"), -0.03, 4e-6) for _ in range(3)]       # crashed / diverged nodes
    pr = M.fit_prior(bulk + tail)
    assert abs(pr.means[()] - 0.0011) < 3e-4                        # the median of the bulk, not -.003
    assert M.fit_prior(bulk + tail, spread="all").means[()] < -0.002
    w = [M.PriorPoint(("c",), x, 1e-6, w=wt) for x, wt in ((0.0, 1.0), (0.01, 0.5), (0.02, 0.5), (0.03, 0.5))]
    assert M._wmedian(w) == 0.01 and M._wmedian(bulk[:4]) == M._median([p.delta for p in bulk[:4]])


# ---------------------------------------------------------------------------------------------- K8 bar.ranked
def test_fresh_test_seeds_are_test_stage_runs_and_a_failed_test_ranks_last():
    from loop.core import bar as BR

    from .core.test_bar import CFG, NEW, ledger
    led = ledger({"tested": (0.690, 5), "screened": (0.689, 5, 0), "failed": (0.692, 5)})
    # the 'tested' node's TEST-pool seeds as SCREENS (selected on): no fresh test, no verdict
    scr = [dataclasses.replace(o, stage="screen") if o.node == "tested" else o for o in led.obs]
    bar = BR.bar_from(led, CFG)
    b = BR.beat(dataclasses.replace(led, obs=tuple(scr)), bar, "tested")
    assert b.n_test == 0 and b.p_test is None and not b.beats
    assert BR.beat(led, bar, "tested").beats and BR.beat(led, bar, "tested").n_test == 5
    led = dataclasses.replace(led, verdicts=((5.0, "failed", "pass"), (9.0, "failed", "fail")))
    order = [x.node for x in BR.ranked(led, bar, incumbent="new")]
    assert order.index("failed") > order.index("screened") and order.index("failed") > order.index("tested")
    assert BR.last_fail(led) == {"failed"} and NEW


# ---------------------------------------------------------------------------------------------- K4 ill-conditioning
def test_one_ill_conditioned_crash_is_a_flake_rerun_and_no_repair_without_positive_evidence(ctx, tmp_path,
                                                                                            monkeypatch):
    from loop.contracts import RunStatus

    from .test_c5_test_ideate import _open_block
    from .test_c7a import ILL_TAIL
    tick, store, t, b = _open_block(ctx, tmp_path, cap=2)
    cand = b.node_ids[0]
    data = CD.load(tick.db_path, tick.ctx.task_id)
    screens = [r for r in store.runs(node_id=cand) if r.stage.value == "screen"]
    store.update_run(screens[0].id, status=RunStatus.CRASH, log_tail=ILL_TAIL)     # ONE crash of 4 finished runs
    monkeypatch.setattr(tick.book, "free", lambda: 4)
    tick.dry = False
    acts: list[str] = []
    tick._repair_sessions(data, acts)
    assert not store.get_state(tick.K_REPAIR) and not store.events("probe_ill_conditioned")
    [ev] = store.events("probe_ill_flake")
    assert ev["run_id"] == screens[0].id and store.get_run(screens[0].id).attempt == 1     # re-run as attempt 1
    assert screens[0].id in store.get_state(tick.K_ILL_FLAKE)
    # two crashes (>= 1/3) but the scorable evidence is not positive: no repair, logged once
    for r in screens[:2]:
        store.update_run(r.id, status=RunStatus.CRASH, log_tail=ILL_TAIL)
    store.set_state(tick.K_ILL_FLAKE, {})
    monkeypatch.setattr(tick, "evidence", lambda *a, **k: {"vs": "p", "n": 2, "mean": -0.001, "seeds": {}})
    tick.dry = True
    tick._repair_sessions(data, acts)
    tick._repair_sessions(data, acts)
    assert not store.get_state(tick.K_REPAIR) and len(store.events("probe_ill_no_repair")) == 1
    assert tick.ctx.cfg.test.ill_backoff_h == 72.0
    tick.close()
    store.close()


def test_repairs_and_ports_are_exempt_from_the_saturated_cluster_cap():
    from types import SimpleNamespace

    from loop.core import ideation as ID
    assert ID.cluster_exempt(SimpleNamespace(id=f"{ID.REPAIR_PREFIX}n1-1", brief_kind="repair_probe"))
    assert ID.cluster_exempt(SimpleNamespace(id=f"{ID.PORT_PREFIX}n7", brief_kind="port"))
    assert ID.cluster_exempt(SimpleNamespace(id=f"{ID.REQUEST_PREFIX}abc", brief_kind="repair"))
    assert not ID.cluster_exempt(SimpleNamespace(id=f"{ID.REQUEST_PREFIX}abc", brief_kind="explore"))


# ---------------------------------------------------------------------------------------------- K3 prerequisites
def test_a_prerequisite_grafts_only_the_missing_key_and_its_codefined_keys(ctx, tmp_path):
    from loop.core import factorial as F

    from .test_c7a import _prereq_env
    from .test_port import _commit, _node
    TK, sim, store, core, tick = _prereq_env(ctx, tmp_path)
    inc = store.get_node("inc").commit
    # n1484-shaped: the introducer adds the schedule clock (two new keys) AND moves an existing key (lr)
    _node(store, "g", _commit(sim, inc, "g", {"loss.clock": 1, "loss.clock_ref": 5, "train.lr": 9e-4}), "inc", 8.0)
    car = "inc@new"
    core.knobs = {}
    core._valid(tick, car, [F.BankKnob(F.Knob("loss.clock", 1, 2, "lin"), 0.001, 1e-6, "declared")],
                core._config(tick, car), [])
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    acts: list[str] = []
    core.prereqs(tick, led, {}, None, acts)
    rec = store.get_state(core.K_PREREQ)[f"{car}|loss.clock"]
    assert rec["status"] == "open" and rec["source"] == "g" and rec["what"].startswith("key patch")
    port = store.get_node(rec["port"])
    cfg = sim.config_at(port.commit)
    assert cfg["loss.clock"] == 1 and cfg["loss.clock_ref"] == 5                    # the co-defined key too
    assert cfg["train.lr"] == sim.config_at(store.get_node(car).commit)["train.lr"]   # never the moved key
    assert "prereq_of=g" in port.notes and "port_of=" not in port.notes
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert led.port("g") is None and led.nodes[port.id].port_of == ""                 # Ledger.port never resolves
    assert TK.ModuleCore.codefined("train.clock", ["train.clock_ref", "data.views"]) == ["train.clock", "train.clock_ref"]
    tick.close()
    store.close()


def test_an_inseparable_introducer_with_a_negative_increment_is_refused(ctx, tmp_path):
    from loop.core import factorial as F

    from .test_c7a import _prereq_env
    from .test_port import _commit, _node
    TK, sim, store, core, tick = _prereq_env(ctx, tmp_path)
    inc = store.get_node("inc").commit
    _node(store, "h", _commit(sim, inc, "h", {"loss.clock": 1}, files={"train.py": "def train(cfg): return 2\n"}),
          "inc", 8.0)
    car = "inc@new"
    core._valid(tick, car, [F.BankKnob(F.Knob("loss.clock", 1, 2, "lin"), 0.001, 1e-6, "declared")],
                core._config(tick, car), [])

    class _P:
        est = {"h": {}}
        epoch = ""

        @staticmethod
        def belief(nid):
            return (-0.0044, 1e-6)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    acts: list[str] = []
    core.prereqs(tick, led, {}, _P(), acts)
    rec = store.get_state(core.K_PREREQ)[f"{car}|loss.clock"]
    assert rec["status"] == "failed" and "refused" in rec["why"] and "-0.0044" in rec["why"]
    tick.close()
    store.close()


# ---------------------------------------------------------------------------------------------- K10 clusters
def test_a_cluster_saturates_only_on_current_base_evidence_and_never_with_a_pass(store, ctx):
    from loop.core import alloc as AL
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, alloc=dataclasses.replace(
        ctx.cfg.alloc, family_block=False)))                      # the share-cap rule alone (set-1 family block: its own tests)

    from .core.test_c7i import _led, _put
    nodes, bel = [], {}
    for i in range(10):
        nodes.append(_put(store, f"k{i}", f"kde_uniformity_margin_v{i}", "representation", ts=i))
        bel[f"k{i}"] = (-0.004, 2)
    for i, tag in enumerate(["hed_stain_jitter", "patient_cap_volume", "register_tokens", "drop_path_ramp",
                             "teacher_temp_warmup", "global_crop_scale", "ibot_mask_ratio", "koleo_weight"]):
        nodes.append(_put(store, f"o{i}", tag, "misc", ts=30 + i))
        bel[f"o{i}"] = (0.0, 1)
    led = _led(nodes)
    kde = lambda cls: next(c for c in cls if "k0" in c.members)                              # noqa: E731
    assert kde(AL.clusters(ctx, store, led, bel)).flag == "saturated-negative"
    old = _led([dataclasses.replace(n, base="A") if n.id in {f"k{i}" for i in range(7)} else n for n in nodes])
    got = kde(AL.clusters(ctx, store, old, bel))                  # 3 of 10 on this base < cluster_min_n / 2
    assert got.cur_n == 3 and got.flag == ""
    got = kde(AL.clusters(ctx, store, led, bel, protect=AL.protected(led, "k4")))     # an incumbent ancestor
    assert got.flag == "" and got.protected.startswith("k4")
    led2 = _led(nodes)
    led2.verdicts = ((1.0, "k2", "pass"),)                                  # a TEST pass inside it
    assert kde(AL.clusters(ctx, store, led2, bel, protect=AL.protected(led2, ""))).flag == ""
    # the [differs from:] escape needs a mechanism word none of the cited members has
    cls = AL.clusters(ctx, store, led, bel)
    AL.remember(cls, store, led, 25)
    k = kde(cls)
    ok = "[differs from: " + "; ".join(f"{m}: it changes the kernel bandwidth schedule entirely" for m in k.sample) + "]"
    assert AL.cluster_rule(ctx, "representation", "kde_uniformity_margin_v8", "a kde margin", ok)[0] == \
        "saturated_cluster"
    assert AL.cluster_rule(ctx, "representation", "kde_uniformity_margin_anneal", "a kde margin", ok)[0] == ""


# ---------------------------------------------------------------------------------------------- K1 Thompson kinds
def _kled(per, rng, gpu=(1.0, 4.0), mu=-0.001, sd=0.003):
    from loop.core import alloc as AL

    from .core.test_c7i import _led, _n, _r
    nodes, runs, bel = [], [], {}
    for k in AL.llm_kinds():
        for i in range(per):
            nid = f"{k}{i}"
            nodes.append(_n(nid, notes=f"request=r{nid} brief={k}"))
            runs.append(_r(nid, rng.uniform(*gpu)))
            bel[nid] = (rng.gauss(mu, sd), 2)
    return _led(nodes, runs), bel


def test_thompson_kind_weights_rarely_crown_a_winner_under_the_permutation_null_and_are_deterministic():
    import random

    from loop.core import alloc as AL
    rng, hits, R = random.Random(5), 0, 120
    for s in range(R):                                   # every kind draws its units from ONE distribution
        led, bel = _kled(5, rng)
        tab, _ = AL.kind_rates(led, bel, kinds=AL.llm_kinds(), seed=s, draws=200)
        hits += max(r["w"] for r in tab.values()) > 0.9
    assert hits / R <= 0.05
    led, bel = _kled(5, random.Random(1))
    a, pa = AL.kind_rates(led, bel, seed="tok")
    b, pb = AL.kind_rates(led, bel, seed="tok")
    assert {k: r["w"] for k, r in a.items()} == {k: r["w"] for k, r in b.items()}          # seeded per tick
    assert abs(sum(r["w"] for r in a.values()) - 1.0) < 1e-9
    sub = AL.softmax(a, ["explore", "program"], 0.0, pa)
    assert abs(sum(sub.values()) - 1.0) < 1e-9 and set(sub) == {"explore", "program"}


def test_units_are_valued_over_the_incumbent_split_x_star_with_the_carrier_and_credit_ports_to_their_origin():
    from loop.core import alloc as AL

    from .core.test_c7i import _led, _n, _r
    nodes = [_n("e1", notes="request=r1 brief=explore"),
             _n("w0", parent="e1", notes="wave=w001 point=0 block=0 seed=7", lane="factorial"),
             _n("wx", parent="e1", notes="factorial_of=w001", lane="factorial"),
             _n("q1", parent=None, notes="prereq_of=e1 prereq_for=dino.x"),
             _n("f1", notes="request=r7 brief=repair repair_of=e1")]
    nodes[3] = _n("q1", notes="prereq_of=e1 prereq_for=dino.x")
    led = _led(nodes, [_r(n.id, 1.0) for n in nodes])
    bel = {"e1": (0.003, 3), "w0": (0.01, 1), "wx": (0.002, 3), "q1": (0.001, 2), "f1": (0.001, 2)}
    gains = {"e1": -0.001, "wx": 0.004, "f1": 0.0005}       # vs the incumbent: e1 is BELOW it
    us = AL.units(led, bel, gains=gains)
    assert us["wave:w001"]["best"] == 0.002 and us["e1"]["credit"] == 0.002           # x* split 50/50
    assert us["e1"]["best"] == 0.0 and AL.unit_value(us["e1"]) == 0.002             # its own gain floored at 0
    assert AL.unit_of(led, "q1") == ("e1", "explore") and us["e1"]["nodes"] == 2     # the port: its origin's unit
    assert us["f1"]["kind"] == AL.PROBE_REPAIR                                        # a repair_of= child: own kind


# ---------------------------------------------------------------------------------------------- K2 GPU split
def test_the_gpu_split_is_pinned_at_the_cap_until_its_interval_excludes_it():
    from loop.core import alloc as AL
    tab = {"explore": {"gain": 0.01, "gpu_h": 40.0}, "wave": {"gain": 0.01, "gpu_h": 20.0}}
    noisy = {"rows": {"explore": [(0.01, 4.0)] + [(0.0, 4.0)] * 9, "wave": [(0.005, 10.0), (0.005, 10.0)]},
             "seed": 1}
    sp = AL.gpu_split(tab, 0.4, 0.1, noisy)
    assert sp["point"] < 0.4 <= sp["ci_hi"] and sp["share"] == 0.4              # one lucky unit: not yet excluded
    sure = {"rows": {"explore": [(0.0001, 4.0)] * 10, "wave": [(0.005, 10.0), (0.005, 10.0)]}, "seed": 1}
    tab2 = {"explore": {"gain": 0.001, "gpu_h": 40.0}, "wave": {"gain": 0.01, "gpu_h": 20.0}}
    sp = AL.gpu_split(tab2, 0.4, 0.1, sure)
    assert sp["ci_hi"] < 0.4 and sp["share"] == 0.1                               # excluded: the point, floored


def test_rank_hard_reserves_slots_for_discovery_screens_of_new_llm_nodes():
    from loop.core import acquire as A
    opts = [A.Option("wave", f"W{i}", "factorial", 0.004, 1e-5, 1e-6, 8.0, 8) for i in range(2)] + \
           [A.Option("screen", "N1", "explore", 0.0005, 1e-5, 1e-6, 2.0, 2)]
    soft, groups = {"group:llm": 0}, {"N1": "llm"}
    plain = A.rank(opts, 8, 0, soft=soft, groups=groups)
    assert [a.node for a in plain] == ["W0"]                       # the LLM soft cap: the waves take every slot
    out = A.rank(opts, 8, 0, soft=soft, groups=groups, reserve=(2, {"N1"}))
    assert [a.node for a in out] == ["N1"]                         # 2 slots held: no wave fits the other 6
    out = A.rank(opts[:2], 8, 0, soft=soft, groups=groups, reserve=(2, {"N1"}))
    assert [a.node for a in out] == ["W0"]                                        # nothing to discover: no idle hold


def test_an_unplaced_discovery_reservation_is_released_to_other_work():
    """Live 06:30Z: primary 5/8 used, 3 free, idle jobs pending: the discovery reservation held slots its (blocked)
    discovery action could not take. Now the third pass hands them to the other actions."""
    from loop.core import acquire as A
    opts = [A.Option("screen", "D1", "explore", 0.003, 1e-5, 1e-6, 2.0, 2),
            A.Option("screen", "X1", "explore", 0.002, 1e-5, 1e-6, 2.0, 2),
            A.Option("screen", "X2", "explore", 0.0015, 1e-5, 1e-6, 2.0, 2)]
    groups = {"D1": "llm"}
    # D1's tag budget is exhausted: it cannot be placed, so its 3-slot reservation must not idle the free slots
    out = A.rank(opts, 3, 0, groups=groups, reserve=(3, {"D1"}), tags={"D1": "t"}, tag_left={"t": 0.0})
    assert {a.node for a in out} == {"X1"} or {a.node for a in out} >= {"X1"}
    assert sum(a.runs for a in out) >= 2


# ---------------------------------------------------------------------------------------------- O3 preemption backoff
def test_preempt_backoff_spares_reference_stages_and_unsaved_resumable_runs_and_requeues_a_drop_on_primary(ctx,
                                                                                                         store):
    from loop.contracts import JobStatus, RunStatus, Stage
    from loop.runner import RunBook

    from .test_driver import _pnode, _PoolAdapter, _pools_ctx
    ctx = _pools_ctx(ctx, primary=1, idle=6)
    adapter = _PoolAdapter()
    state = {"fresh:screen:1": {"armed": False, "fraction": 0.0, "resumable": True},
             "cold:screen:1": {"armed": False, "fraction": 0.0}}
    adapter.resume_state = lambda h: state.get(h.run_id, {"armed": False, "fraction": 0.0})
    book = RunBook(ctx, store, adapter)
    book.launch(_pnode(store, "busy"), Stage.CONFIRM, 1)                 # holds the only primary slot
    for n, stage in (("par", Stage.PARENT), ("base", Stage.BASELINE), ("aud", Stage.AUDIT),
                     ("pro", Stage.PROMOTE), ("fresh", Stage.SCREEN), ("cold", Stage.SCREEN)):
        book.launch(_pnode(store, n), stage, 1)
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", restarts=9, requeued=1)
    book.poll()
    acts = book.idle_churn()
    assert [a.split()[3] for a in acts] == ["cold:screen:1"]            # only the never-resumable screen
    cold = store.get_run("cold:screen:1")
    assert cold.pool == "primary" and cold.attempt == 1 and cold.status is not RunStatus.VOID
    assert not [e for e in store.events("run_cancelled") if "re-queued on primary" not in e["reason"]]
    for rid in ("par:parent:1", "base:baseline:1", "aud:audit:1", "pro:promote:1", "fresh:screen:1"):
        assert store.get_run(rid).status is RunStatus.RUNNING and store.get_run(rid).pool == "idle", rid


# ---------------------------------------------------------------------------------------------- live 10-14 03:30Z
def test_the_pair_plan_is_relogged_only_on_a_new_top_set_or_a_material_pred_move():
    from loop.core import pairs as PR
    top = [{"a": "n2725", "b": "n0400", "pred": 0.00191}, {"a": "n2725", "b": "n0335", "pred": 0.00116}]
    assert PR.plan_moved(None, top)
    drift = [dict(t, pred=t["pred"] + 2e-5) for t in top]
    assert not PR.plan_moved(top, drift) and not PR.plan_moved(top, drift[::-1])
    assert PR.plan_moved(top, [dict(top[0], pred=0.0026), top[1]])
    assert PR.plan_moved(top, [top[0], {"a": "n2725", "b": "n0430", "pred": 0.0012}])


def test_queued_node_worktrees_are_prewarmed_on_one_background_thread(ctx, tmp_path):
    sim, store = _two_epochs(ctx, tmp_path)
    tick = TK.Tick(ctx, store, sim, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    made: list[str] = []
    sim.warm_worktree = lambda nid, commit: made.append(nid) or True
    nodes = [n for n in store.nodes() if n.commit][:3]
    acts: list[str] = []
    tick._prewarm({n.id: "QUEUED" for n in nodes} | {"zz": "SCREENING"}, acts)
    tick._warm_th.join(5)
    assert sorted(made) == sorted(n.id for n in nodes)
    tick._prewarm({}, acts)
    assert acts == [f"worktree prewarm: {len(nodes)} queued node worktree(s) created between ticks"]
    tick.close()
    store.close()


# ---------------------------------------------------------------------------------------------- O4 wave controls
def test_a_dead_wave_control_row_is_listed_again_and_resubmitted(ctx, tmp_path):
    from loop.contracts import RunStatus, Stage
    sim, store = _two_epochs(ctx, tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slots=16))
    tick = TK.Tick(ctx, store, sim, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    store.set_state(TK.K_WAVE_BUILD, {"wave": "w017", "n": 17, "carrier": "inc@new", "bank": [_bank()[0].to_json()],
                                      "seeds": [100147], "diffs": {}, "used": [], "ts": 40.0, "tries": 0,
                                      "pre": False})
    tick._launch([], CD.load(tmp_path / "loop.db", ctx.task_id), {}, 60.0, [])
    rid = "inc@new:parent:100147"
    assert store.get_run(rid) is not None
    store.update_run(rid, status=RunStatus.CRASH)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    assert F.wave_controls(led, "w017") == [("inc@new", 100147)]          # a crashed control is no control
    acts: list[str] = []
    assert tick._retry_controls(led, acts) == [rid]
    r = store.get_run(rid)
    assert r.attempt == 1 and r.status is not RunStatus.CRASH and r.stage is Stage.PARENT
    assert store.events("wave_control_retry")
    tick.close()
    store.close()


# ---------------------------------------------------------------------------------------------- O5 launch guard
def test_the_launch_guard_refuses_protected_key_changes_and_values_over_a_cap_ceiling(xenv):
    from types import SimpleNamespace

    from .test_review_c4j_locks import _commit, git
    ad, _, repo, _ = xenv
    surf = SimpleNamespace(locked=(), protected_keys=("probe", "train.max_train_samples"),
                           cap_ceilings={"train.max_train_flops": 100})
    ad.ctx = SimpleNamespace(spec=SimpleNamespace(surface=surf, repo=str(repo)), cfg=ad.ctx.cfg)
    ad.config_rel, ad.locks = "fake_config.yaml", ()
    git(repo, "checkout", "-q", "main")
    ref = _commit(repo, {"fake_config.yaml": "train:\n  lr: 0.001\n  max_train_flops: 100\nprobe:\n  n: 3\n"}, "R")
    ad.lock_ref = ref
    ok = _commit(repo, {"fake_config.yaml": "train:\n  lr: 0.002\n  max_train_flops: 50\nprobe:\n  n: 3\n"}, "ok")
    probe = _commit(repo, {"fake_config.yaml": "train:\n  lr: 0.002\n  max_train_flops: 50\nprobe:\n  n: 4\n"}, "p")
    over = _commit(repo, {"fake_config.yaml": "train:\n  lr: 0.002\n  max_train_flops: 200\n"
                                              "  max_train_samples: 9\nprobe:\n  n: 3\n"}, "o")
    assert ad.config_violations(ok) == []                  # a shrink below the ceiling is fine
    assert ad.config_violations(probe) == ["config:probe.n"]
    assert ad.config_violations(over) == ["config:train.max_train_samples", "cap:train.max_train_flops=200>100"]
    assert "config:probe.n" in ad.lock_violations(probe)


# ---------------------------------------------------------------------------------------------- O6 brief throttle
def test_the_throttle_median_age_is_over_screenable_nodes_and_a_long_throttle_alerts():
    from types import SimpleNamespace

    from loop.core import alloc as AL

    from .core.test_c7i import _led, _n
    led = _led([_n("old", ts=1.0), _n("new", ts=3600.0 * 9), _n("new2", ts=3600.0 * 9.5)])
    st = {"old": "QUEUED", "new": "QUEUED", "new2": "QUEUED"}
    now = 3600.0 * 10
    assert AL.backlog(led, st, now)["median_age_h"] == 1.0
    bl = AL.backlog(led, st, now, {"old"})                # only 'old' can screen: its age is the backlog's
    assert round(bl["median_age_h"], 3) == 10.0 and bl["n"] == 3 and bl["n_screenable"] == 1
    assert AL.screenable(SimpleNamespace(last_opts=[SimpleNamespace(kind="screen", node="new"),
                                                    SimpleNamespace(kind="seed", node="old")])) == {"new"}
    assert AL.screenable(SimpleNamespace()) is None

    class S:
        def __init__(self):
            self.state, self.logs = {}, []

        def get_state(self, k):
            return self.state.get(k)

        def set_state(self, k, v):
            self.state[k] = v

        def log(self, kind, **kw):
            self.logs.append(kind)
    s = S()
    assert AL.throttle_alert(s, "7 queued > 6", 0.0) == ""
    assert AL.throttle_alert(s, "7 queued > 6", 3600.0 * 5) == ""
    assert AL.throttle_alert(s, "7 queued > 6", 3600.0 * 6.5).startswith("ALERT: briefs throttled 6.5 h")
    assert AL.throttle_alert(s, "7 queued > 6", 3600.0 * 7) == ""          # at most hourly
    assert AL.throttle_alert(s, "", 3600.0 * 8) == "" and s.state[AL.K_THROTTLE_SINCE] is None
    assert s.logs == ["brief_throttle_alert"]
