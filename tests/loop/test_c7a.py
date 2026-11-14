"""v3-c7a: the pairs planner's leg prior + winner's-curse N (A9), the test entry gate (A3), probe ill-conditioning (A4),
extrapolation on the current incumbent (A5), missing-prerequisite ports (A6), heartbeat / eval-only (A7) and idle-pool
preemption backoff (A8)."""

from __future__ import annotations

import dataclasses
import math

from loop.contracts import NodeStatus
from loop.core import data as CD
from loop.core import model as M

from .test_c6a import _pairs_env


# -- A9: pairs ---------------------------------------------------------------------------------------------------------
def _well_measured_env(monkeypatch):
    """Two well-measured current-epoch legs with complementary axes (6 seeds, se .0008), near-zero siblings, a
    crashed / diverged tail, a noisy old-epoch n=2 screen and an old crossover (a combination); the cell prior is
    the live 10-14 one (mean -.0095 dragged by the tail, tau .0023)."""
    spec = {
        "la": ("inc", "E2", 0.0030, 6, {"seg": (0.012, 4.0), "slide": (0.008, 3.0), "rob": (-0.004, -1.5)}),
        "lb": ("inc", "E2", 0.0028, 6, {"rob": (0.014, 4.5), "cls": (0.004, 2.0), "seg": (-0.002, -1.0)}),
        "weak": ("inc", "E2", 0.0007, 3, {"slide": (0.004, 0.8)}),            # 3 seeds, se .0018: z < 1, no leg
        "old_n2": ("x", "E1", 0.0060, 2, {"cls": (0.02, 4.0)}),                 # an old n=2 screen: never a leg
        "old_xo": ("x", "E1", 0.0270, 5, {"slide": (0.05, 6.0)}),               # an old crossover: never a leg
        **{f"s{i}": ("inc", "E2", -0.001 + 0.0004 * (i % 3), 4, {"cls": (-0.001, -0.5)}) for i in range(8)},
    }
    PR, led, post = _pairs_env(monkeypatch, spec)
    nodes = dict(led.nodes)
    nodes["old_xo"] = dataclasses.replace(nodes["old_xo"], second_parent="y", lane="crossover")
    led = dataclasses.replace(led, nodes=nodes)
    sd = {"la": 0.0008, "lb": 0.0008, "weak": 0.0018}
    post.est = {n: {M.COMPOSITE: M.Estimate(m, sd.get(n, 0.0015) ** 2, k, k)} for n, (_, _, m, k, _) in spec.items()}
    post.belief = lambda nid: M.posterior(-0.0095, 0.0023 ** 2, post.est[nid][M.COMPOSITE])
    return PR, led, post


def test_well_measured_complementary_current_legs_make_a_buildable_pair(ctx, monkeypatch):
    """A9 fixture (shaped like 10-14: 7 legs from 261 candidates, 0 pairs). Well-measured complementary current-epoch
    legs give a buildable config-only merge; a 3-seed leg at z < 1 and old-epoch noise do not enter."""
    PR, led, post = _well_measured_env(monkeypatch)
    cfg = {"la": {"aug.a": {"from": 1, "to": 2}}, "lb": {"train.b": {"from": 0.1, "to": 0.2}}}
    inc_cfg = {"aug.a": 1, "train.b": 0.1}
    info: dict = {}
    got = PR.plan(post, led, ctx, {}, incumbent="inc", inc_cfg=inc_cfg, config_of=cfg.get, used={}, info=info)
    assert [p.key for p in got] == ["la|lb"] and got[0].config == {"aug.a": 2, "train.b": 0.2}   # config-only merge
    assert got[0].value > 0 and got[0].pred < 0.0058                         # shrunk legs, stack prior (K5)
    assert info["legs"] == 2 and info["curse_n"] == 2                        # the curse: the legs the pairs draw on
    assert "stack prior" in got[0].why


def test_the_curse_counts_independent_legs_and_each_pair_its_own_sd(ctx, monkeypatch):
    PR, led, post = _pairs_env(monkeypatch, {
        "p1": ("inc", "E2", 0.0020, 4, {"seg": (0.006, 3.0)}),
        "p2": ("inc", "E2", 0.0020, 4, {"slide": (0.006, 3.0)}),
        "p3": ("inc", "E2", 0.0020, 4, {"seg": (0.005, 3.0)}),                  # same dominant axis as p1
    })
    post.belief = lambda nid: (0.0020, {"p1": 0.0005, "p2": 0.0005, "p3": 0.0015}[nid] ** 2)
    info: dict = {}
    import dataclasses
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, pairs=dataclasses.replace(ctx.cfg.pairs, curse=1.0)))
    got = {p.key: p for p in PR.plan(post, led, ctx, {}, incumbent="inc", used={}, info=info)}
    assert set(got) == {"p1|p2", "p2|p3"} and info["curse_n"] == 3
    cz = PR.curse_z(3)                                   # curse on explicitly (v3 review K5: off by default)
    assert math.isclose(got["p1|p2"].value, 0.00226 - cz * math.hypot(0.0005, 0.0005) + 0.25 * got["p1|p2"].coverage)
    assert got["p1|p2"].value > got["p2|p3"].value                          # its own (smaller) sd


# -- A3: the test entry gate -------------------------------------------------------------------------------------------
def _o(node, seed, y):
    return CD.Obs(f"{node}:screen:{seed}", node, seed, "screen", "select", {a: y for a in CD.AXES}, y, 1.0, 0.0, 0)


def test_a_four_seed_screen_one_seed_carries_is_stopped_at_entry():
    """n2693 (10-13): +.0026 on 4 seeds, 79% from one; leave-one-out (+.0007) passed, the share check was off below
    5 pairs (and the live task had it off). Now: share from 3 pairs at an n-scaled cap, a trimmed mean from 4."""
    from loop.core import status as ST
    W = {a: 1 / 6 for a in CD.AXES}
    par = [_o("p", s, 0.68) for s in (1, 2, 3, 4)]
    d = (0.0082, 0.0010, 0.0008, 0.0004)                 # 79% from seed 1
    c = [_o("c", s, 0.68 + x) for s, x in zip((1, 2, 3, 4), d)]
    assert ST.seed_artifact(c, par, W, share_min_n=1000, trim_min_n=0) == (True, "")     # the old gate let it in
    ok, why = ST.seed_artifact(c, par, W, share_min_n=3)           # v3 review K6: a one-seed outlier, t 1.90
    assert not ok and "seed 1 is a one-seed outlier" in why and "carries 79%" in why
    trim = [_o("c", s, 0.68 + x) for s, x in zip((1, 2, 3, 4), (0.009, 0.0003, 0.0002, 0.0001))]
    ok, why = ST.seed_artifact(trim, par, W, share_min_n=1000)
    assert not ok and "trimmed mean" in why                # LOO min +.0002 > 0; the middle two keep 10% of the mean
    even3 = [_o("c", s, 0.68 + x) for s, x in zip((1, 2, 3), (0.004, 0.002, 0.003))]
    assert ST.seed_artifact(even3, par, W, share_min_n=3)[0]      # no seed an outlier of the others
    n3 = [_o("c", s, 0.68 + x) for s, x in zip((1, 2, 3), (0.00062, 0.00074, 0.00858))]   # n2693 at its test entry
    assert not ST.seed_artifact(n3, par, W, share_min_n=3, sd_d=0.00339)[0]     # the live far-fit sd_d: t 1.90


def test_the_task_turns_the_share_gate_on_from_three_seeds():
    from loop.spec import load_context
    ctx = load_context("tasks/nanopath_v2.yaml", require_measured=False)
    assert ctx.cfg.status.share_min_n == 3 and ctx.cfg.status.trim_min_n == 4 and ctx.cfg.status.outlier_z == 1.8


# -- A4: probe ill-conditioning -----------------------------------------------------------------------------------------
ILL_TAIL = ("  File \"/wt/n1/probe.py\", line 815, in inline_pathobench_survival\n"
            "sklearn.exceptions.ConvergenceWarning: Optimization terminated early, you might want to increase the "
            "number of iterations (max_iter=100000).\nsubprocess.CalledProcessError: Command '['python', "
            "'/wt/n1/probe.py', 'x.json']' returned non-zero exit status 1.")


def test_an_ill_conditioned_probe_crash_is_its_own_class():
    from loop import tick as TK
    from loop.contracts import Run, RunStatus, Stage
    mk = lambda **kw: Run(id="a", task_id="t", node_id="n", seed=1, stage=Stage.SCREEN, **kw)      # noqa: E731
    assert TK.crash_class(mk(status=RunStatus.CRASH, log_tail=ILL_TAIL)) == TK.ILL_CONDITIONED
    assert TK.crash_class(mk(status=RunStatus.CRASH, log_tail="python /wt/n1/probe.py failed")) == "crash:probe"
    assert TK.ILL_CONDITIONED in TK.EVAL_RUN_FAULT


def test_a_test_closing_on_ill_conditioned_crashes_has_no_verdict_and_queues_a_repair(ctx, tmp_path):
    from loop.contracts import RunStatus

    from .test_c5_test_ideate import _block, _open_block
    tick, store, t, b = _open_block(ctx, tmp_path, cap=2)
    cand, ctrl = b.node_ids
    store.update_run(f"{cand}:confirm:{b.seeds[0]}", status=RunStatus.CRASH, log_tail=ILL_TAIL)
    store.update_run(f"{cand}:confirm:{b.seeds[1]}", status=RunStatus.CRASH, log_tail=ILL_TAIL)
    for s in b.seeds[2:]:                                     # the scorable seeds: the evidence the repair keeps
        for n, st in ((cand, "confirm"), (ctrl, "remint")):
            y = 0.68 + (0.002 if n == cand else 0.0)
            store.update_run(f"{n}:{st}:{s}", status=RunStatus.DONE, aggregate=y,
                             scores={k: y for k in CD.PROBE_KEYS.values()}, gpu_h=1.0)
    t[0] += 60
    rep = tick.step()
    after = _block(store, b.id)
    assert not after.open and after.verdicts == ()                         # NO FAIL: an idea defect, not a verdict
    assert not [e for e in store.events("test_verdict") if e.get("node_id") == cand]
    [ev] = store.events("test_ill_conditioned")
    assert ev["node_id"] == cand and ev["evidence"]["n"] == 3 and abs(ev["evidence"]["mean"] - 0.002) < 1e-9
    book = store.get_state(tick.K_REPAIR)
    assert book[cand]["status"] in ("pending", "started") and sorted(book[cand]["seeds"]) == sorted(b.seeds[:2])
    assert cand in tick._test_backoff(t[0] + 60)                          # no retest while the repair is pending
    assert any("probe ill-conditioned" in x for x in rep.lines)
    brief = tick.repair_brief(cand, book[cand])
    assert "reduce the added feature" in brief and "LOCKED" in brief and "n 3" in brief
    assert len(store.events("probe_ill_conditioned")) == 1
    tick.close()
    store.close()


def test_a_screen_crash_books_one_repair_and_the_session_is_capped(ctx, tmp_path, monkeypatch):
    from loop.contracts import RunStatus

    from .test_c5_test_ideate import _open_block
    tick, store, t, b = _open_block(ctx, tmp_path, cap=2)
    cand = b.node_ids[0]
    data = CD.load(tick.db_path, tick.ctx.task_id)
    screens = [r for r in store.runs(node_id=cand) if r.stage.value == "screen"] or \
        [r for r in store.runs(node_id=cand)]
    for r in screens[:2]:                  # v3 review K4: >= 2 ill-conditioned crashes, >= 1/3 of its finished runs
        store.update_run(r.id, status=RunStatus.CRASH, log_tail=ILL_TAIL)
    monkeypatch.setattr(tick, "evidence", lambda *a, **k: {"vs": "p", "n": 2, "mean": 0.002, "seeds": {}})
    acts: list[str] = []
    tick.dry = True
    tick._repair_sessions(data, acts)
    tick._repair_sessions(data, acts)                                     # booked once
    assert list(store.get_state(tick.K_REPAIR)) == [cand] and len(store.events("probe_ill_conditioned")) == 1
    assert any(a.startswith(f"would start a researcher repair session for {cand}") for a in acts)
    tick.close()
    store.close()


def test_the_width_heuristic_flags_a_second_order_readout_block(ctx, tmp_path):
    from loop.contracts import Node, NodeStatus, Operator, Strategy

    from .test_c5_test_ideate import _open_block
    tick, store, t, b = _open_block(ctx, tmp_path, cap=2)
    inc = b.node_ids[-1]
    diff = tmp_path / "w.diff"
    diff.write_text("+    cov = torch.einsum('bi,bj->bij', x, x)\n+    feats = cov[:, iu[0], iu[1]]  # triu_indices\n")
    n = store.get_node(inc)
    store.put_node(Node(id="nw", task_id=store.task_id, commit=n.commit, parent_id=inc, strategy=Strategy.EXPLORE,
                        operator=Operator.DRAFT, family="eval_readout", mechanism_tag="cov", diff_hash="dw",
                        status=NodeStatus.QUEUED, diff_path=str(diff)))
    data = CD.load(tick.db_path, tick.ctx.task_id)
    acts: list[str] = []
    tick._width_warn(data, acts)
    tick._width_warn(data, acts)
    [ev] = store.events("probe_width_warning")
    assert ev["node_id"] == "nw" and sum("nw: WARNING" in a for a in acts) == 1
    tick.close()
    store.close()


# -- A5: follow the gradient on the current incumbent --------------------------------------------------------------------
def test_extrapolation_rearms_on_the_incumbent_and_continues_winning_moves(ctx, tmp_path):
    from loop import extrapolate as EX
    from loop import tick as TK

    from .test_c5_waves import _BankPost, _child
    from .test_port import _commit, _two_epochs
    sim, store = _two_epochs(ctx, tmp_path)
    car = "inc@new"
    _child(store, "w", _commit(sim, store.get_node(car).commit, "w", {"train.lr": 1e-3}), car, 20.0, "optimization")
    store.set_state("core.change", {f"w|{car}": True, "x|inc": True})       # one config key each (no git here)
    store.set_state(EX.K_EXT, {"nodes": {"old1": {"incumbent": "n0655"}}, "stopped": {}})  # stale, old incumbent
    ex = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, extrapolate=dataclasses.replace(
        ctx.cfg.extrapolate, enabled=True, min_z=1.0)))
    core = TK.ModuleCore()
    tick = TK.Tick(ex, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    post = _BankPost({"w": 0.003, "x": 0.004, "z": -0.002}, {"w": 4, "x": 4, "z": 4}, led.epoch)
    cfg = core._config(tick, car)
    acts: list[str] = []
    got = {b.knob.key: b for b in core._extrapolate_knobs(tick, led, post, car, cfg, 1e-6, acts)}
    lr = got["train.lr"]                                                   # the current child's move 7e-4 -> 1e-3
    assert lr.source == "extrapolate" and lr.knob.lo == 7e-4 and lr.knob.hi > 1e-3 and "w on inc@new" in lr.why
    st = got["aug.strength"]                                               # old epoch, via the port (same start)
    assert cfg["aug.strength"] == 0.5 and (st.knob.lo, st.knob.hi) == (0.5, 0.3) and "via the port" in st.why
    assert not any(b.why.startswith("z on") for b in got.values())        # a losing move is no knob
    bk = store.get_state(EX.K_EXT)
    assert bk["v3"]["incumbent"] == car and bk["v3"]["ts"] == 50.0 and len(bk["v3"]["knobs"]) == 2
    assert [e["incumbent"] for e in store.events("extrapolate_rearmed")] == [car]
    core._extrapolate_knobs(tick, led, post, car, cfg, 1e-6, [])
    assert len(store.events("extrapolate_rearmed")) == 1                   # same incumbent: no re-arm
    no = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, extrapolate=dataclasses.replace(
        ctx.cfg.extrapolate, enabled=False)))
    off = TK.Tick(no, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    assert core._extrapolate_knobs(off, led, post, car, cfg, 1e-6, []) == []          # extrapolate.enabled off
    tick.close()
    off.close()
    store.close()


# -- A6: missing-prerequisite ports -------------------------------------------------------------------------------------
def _prereq_env(ctx, tmp_path):
    from loop import tick as TK

    from .test_port import _commit, _node, _two_epochs
    sim, store = _two_epochs(ctx, tmp_path)
    inc = store.get_node("inc").commit
    _node(store, "d", _commit(sim, inc, "d", {"aug.new_knob": 0.5}), "inc", 5.0)       # introduced the key
    _node(store, "d2", _commit(sim, store.get_node("d").commit, "d2", {"aug.new_knob": 0.7}), "d", 6.0)  # moved it
    _node(store, "e", _commit(sim, inc, "e", {"aug.other_knob": 1.0}), "inc", 7.0)
    core = TK.ModuleCore()
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    return TK, sim, store, core, tick


def test_a_knob_missing_its_key_ports_the_introducer_first_then_retries_on_top(ctx, tmp_path):
    """A knob on a config key the carrier lacks is dropped AND booked: the key's introducer (the node whose own diff
    added it, found from a node holding it up its parent chain -- not the later node that only moved it) is ported
    onto the carrier as its own QUEUED node; at most one prerequisite is open; once the port is measured the knob is
    a bank on the port (the dependent wave retries on top of it)."""
    from loop.core import factorial as F
    TK, sim, store, core, tick = _prereq_env(ctx, tmp_path)
    car = "inc@new"
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    core.knobs = {}
    k1 = F.BankKnob(F.Knob("aug.new_knob", 0.5, 0.9, "lin"), 0.001, 1e-6, "extrapolate")
    k2 = F.BankKnob(F.Knob("aug.other_knob", 1.0, 2.0, "lin"), 0.001, 1e-6, "plateau")
    kept = core._valid(tick, car, [k1, k2], core._config(tick, car), [])
    assert kept == []
    book = store.get_state(core.K_PREREQ)
    assert set(book) == {f"{car}|aug.new_knob", f"{car}|aug.other_knob"}
    assert core.introducer(tick, led, "aug.new_knob", ["d2"]) == "d"            # d2 only moved it
    acts: list[str] = []
    core.prereqs(tick, led, {}, None, acts)
    book = store.get_state(core.K_PREREQ)
    opened = [r for r in book.values() if r["status"] == "open"]
    assert len(opened) == 1 and sum(r["status"] == "waiting" for r in book.values()) == 1   # <= 1 open
    rec = opened[0]
    port = store.get_node(rec["port"])
    assert port.parent_id == car and f"prereq_for={rec['key']}" in port.notes and port.status.value == "queued"
    assert rec["source"] == {"aug.new_knob": "d", "aug.other_knob": "e"}[rec["key"]]
    core.prereqs(tick, CD.load(tmp_path / "loop.db", ctx.task_id), {}, None, [])
    assert sum(r["status"] == "open" for r in store.get_state(core.K_PREREQ).values()) == 1   # unmeasured: still 1

    class _P:
        epoch = led.epoch
        est = {rec["port"]: {}}
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    core.knobs = {}
    acts = []
    core.prereqs(tick, led, {}, _P(), acts)
    book = store.get_state(core.K_PREREQ)
    assert book[f"{car}|{rec['key']}"]["status"] == "done"
    assert [b.knob.key for b in core.knobs[rec["port"]]] == [rec["key"]]        # retried on top of the port
    assert sum(r["status"] == "open" for r in book.values()) == 1               # the waiting one opened next
    assert {e["key"] for e in store.events("prereq_port")} == {"aug.new_knob", "aug.other_knob"}
    tick.close()
    store.close()


def test_a_port_conflict_on_a_missing_key_books_the_prerequisite_and_a_dead_one_gives_up(ctx, tmp_path):
    TK, sim, store, core, tick = _prereq_env(ctx, tmp_path)
    assert TK.ModuleCore.conflict_keys("conflict: config.yaml:aug.new_knob, config.yaml:train.lr") == [
        "aug.new_knob", "train.lr"]
    assert TK.ModuleCore.conflict_keys("contained") == []
    tick._missing_prereqs("inc@new", "conflict: config.yaml:aug.new_knob, config.yaml:train.lr", port="d2",
                          witness="d2")
    book = store.get_state(core.K_PREREQ)
    assert list(book) == ["inc@new|aug.new_knob"] and book["inc@new|aug.new_knob"]["ports"] == ["d2"]  # lr: held
    led = CD.load(tmp_path / "loop.db", ctx.task_id)
    core.knobs = {}
    core.prereqs(tick, led, {}, None, [])
    rec = store.get_state(core.K_PREREQ)["inc@new|aug.new_knob"]
    assert rec["status"] == "open" and rec["source"] == "d"
    store.update_node(rec["port"], status=NodeStatus.REJECTED, reject_reason="x")
    core.prereqs(tick, CD.load(tmp_path / "loop.db", ctx.task_id), {}, None, [])
    assert store.get_state(core.K_PREREQ)["inc@new|aug.new_knob"]["status"] == "failed"
    assert store.events("prereq_failed")
    tick.close()
    store.close()


# -- A7: heartbeat pools, eval-only null on an epoch change, n_wait ------------------------------------------------------
def test_heartbeat_prints_each_pool_running_pending_and_slots_used(ctx, tmp_path):
    from loop import tick as TK

    from .test_port import _two_epochs
    sim, store = _two_epochs(ctx, tmp_path)
    tick = TK.Tick(ctx, store, sim, TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    names = [p.name for p in ctx.cfg.pools]
    hb = tick.heartbeat(1, 0.1)
    assert " pools=" + ",".join(f"{n}:?r/?p 0/" for n in names[:1]) in hb          # before the first poll: ?
    tick.book.pool_states = {n: (3 + i, i) for i, n in enumerate(names)}
    hb = tick.heartbeat(2, 0.1)
    lim = tick.book.limits()
    for i, n in enumerate(names):
        assert f"{n}:{3 + i}r/{i}p 0/{lim[n]}" in hb
    tick.close()
    store.close()


def test_the_null_control_restarts_on_an_epoch_change_and_waits_count_nodes(tmp_path):
    from loop import evalonly as EO

    from .test_c5_evalonly_lane import lane, ledger, obs, run, spec
    ln, store, clock = lane(tmp_path)
    data = ledger([run("p", "screen", 5, out="/ckpt/p5"), run("p", "screen", 6, out="/ckpt/p6"),
                   run("p", "screen", 7, status="running")], [obs("p", "screen", 5, 0.6), obs("p", "screen", 6, 0.6)])
    acts: list[str] = []
    ln.route(data, [spec("c", 5), spec("c", 6), spec("c", 7)], acts)     # c: 3 seed runs, one node
    assert any("0 run(s) on parent checkpoints, 1 node(s) (3 run(s)) waiting" in a for a in acts)
    store.state[EO.K_V3]["null"]["B"]["status"] = "passed"
    assert ln.route(data, [], []) == []                                  # once per epoch
    b2 = dataclasses.replace(data, runs=tuple(dataclasses.replace(r, base="B2") for r in data.runs),
                             obs=tuple(dataclasses.replace(o, base="B2") for o in data.obs),
                             bases=(*data.bases, (5.0, "root", "B2")))
    assert b2.epoch == "B2"
    out = ln.route(b2, [], [])                                           # a new epoch: its own null control
    assert out and store.state[EO.K_V3]["null"]["B2"]["status"] == "running"
    assert store.state[EO.K_V3]["null"]["B"]["status"] == "passed"


# -- A8: non-fill runs preempted again and again on the idle pool -------------------------------------------------------
def test_a_non_fill_run_preempted_3x_moves_to_free_primary_else_is_dropped_only_when_it_cannot_progress(ctx, store):
    from loop.contracts import JobStatus, RunStatus, Stage
    from loop.runner import RunBook

    from .test_driver import _pnode, _PoolAdapter, _pools_ctx
    ctx = _pools_ctx(ctx, primary=1, idle=5)
    adapter = _PoolAdapter()
    state = {"warm:screen:1": {"armed": True, "fraction": 0.3}, "cold:screen:1": {"armed": False, "fraction": 0.0},
             "late:screen:1": {"armed": True, "fraction": 0.3}, "far:screen:1": {"armed": True, "fraction": 0.9}}
    adapter.resume_state = lambda h: state.get(h.run_id, {})
    book = RunBook(ctx, store, adapter)
    names = ("busy", "warm", "cold", "late", "far")
    nodes = {n: _pnode(store, n) for n in names}
    book.launch(nodes["busy"], Stage.CONFIRM, 1)                         # holds the only primary slot
    for n in names[1:]:
        book.launch(nodes[n], Stage.SCREEN, 1)
    assert all(store.get_run(f"{n}:screen:1").pool == "idle" for n in names[1:])
    restarts = {"late:screen:1": 6, "far:screen:1": 6}
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", restarts=2, requeued=1)
    book.poll()
    assert book.idle_churn() == []                                       # 2 < preempt_max_restarts 3
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", restarts=restarts.get(h.run_id, 3),
                                       requeued=1)
    book.poll()
    acts = book.idle_churn()
    # primary full: ONE drop re-queued there (11-10: at most _overflow_cap of ours queue on a full primary -- here
    # max(1, 1 // 4) = 1); the second qualifying run stays on idle until a primary slot frees
    assert [a.split()[3] for a in acts] == ["cold:screen:1"]
    for n, st in (("warm", RunStatus.RUNNING), ("far", RunStatus.RUNNING), ("busy", RunStatus.RUNNING),
                  ("late", RunStatus.RUNNING)):
        rid = f"{n}:{'confirm' if n == 'busy' else 'screen'}:1"
        assert store.get_run(rid).status is st, rid
    assert store.get_run("late:screen:1").pool == "idle" and store.get_run("late:screen:1").attempt == 0
    r = store.get_run("cold:screen:1")                                   # v3 review O3: re-queued on primary
    assert r.pool == "primary" and r.attempt == 1 and r.status is not RunStatus.VOID
    reasons = [e["reason"] for e in store.events("run_cancelled")]
    assert all(r.startswith("idle fill preempted") for r in reasons)    # the churn's accounting
    assert any("not resumable" in r for r in reasons)
    book.pending.add("cold:screen:1")                                    # SLURM holds it PENDING on primary
    assert book.idle_churn() == []                                       # the queue is full: late stays put
    for rid in ("busy:confirm:1", "cold:screen:1", "late:screen:1"):   # primary frees up
        store.update_run(rid, status=RunStatus.DONE)
    book.poll()
    acts = book.idle_churn()
    assert adapter.moved and adapter.moved[0][1] == "primary" and "preempt backoff" in acts[0]


def test_preempt_backoff_never_drops_without_progress_info_or_a_test_run(ctx, store):
    from loop.contracts import JobStatus, RunStatus, Stage
    from loop.runner import RunBook

    from .test_driver import _pnode, _PoolAdapter, _pools_ctx
    ctx = _pools_ctx(ctx, primary=1, idle=3)
    adapter = _PoolAdapter()
    book = RunBook(ctx, store, adapter)
    book.launch(_pnode(store, "busy"), Stage.CONFIRM, 1)
    book.launch(_pnode(store, "x"), Stage.SCREEN, 1)
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", restarts=9, requeued=1)
    book.poll()
    assert book.idle_churn() == [] and store.get_run("x:screen:1").status is RunStatus.RUNNING   # no resume_state
