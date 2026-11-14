"""Fix batch B (10-12) of loop.core.acquire: wave points are never seeded / tested / prior points, value floors, the
port rules (seeds, positive belief, open cap, learned PORT_P), the wave floor and soft share, new-idea screens from
the lane's heavy tail, and the test entry rule."""

from __future__ import annotations

import sqlite3

import pytest

from loop.core import acquire as A
from loop.core import data as D
from loop.core import model as M
from loop.core import status as ST
from loop.core.acquire import Option

from .test_epochs import NEW, _add_node, _add_runs, _state

S2 = 0.003 ** 2


def _new_epoch(db):
    led, blocks, noise = _state(db)
    inc = ST.chain(led, blocks, noise)[-1]
    t = max(r.submitted_ts for r in led.runs) + 100
    con = sqlite3.connect(db)
    _add_node(con, "root-065b87f", None, t, commit=NEW)
    _add_node(con, "p-" + inc, "root-065b87f", t + 1, notes=f"port_of={inc}")
    _add_runs(con, "root-065b87f", t + 2)
    _add_runs(con, "p-" + inc, t + 2)
    con.commit()
    return con, inc, t


def test_wave_points_are_never_seeded_tested_or_prior_points(sample_db, monkeypatch):
    con, inc, t = _new_epoch(sample_db)
    port = "p-" + inc
    _add_node(con, "w1.p00", port, t + 3, notes="wave=w1 point=0 block=0 seed=3", commit="w1p0")
    _add_node(con, "w1.k", port, t + 3, notes="port_of=x for_wave=w1", commit="w1k")
    _add_node(con, "kid", port, t + 3, commit="kid")
    for n in ("w1.p00", "w1.k", "kid"):
        _add_runs(con, n, t + 4, seeds=(1, 2, 3))
    con.commit()
    con.close()
    led, blocks, noise = _state(sample_db)
    seen = []
    real = M.fit_prior
    monkeypatch.setattr(M, "fit_prior", lambda pts, **k: seen.append(len(pts)) or real(pts, **k))
    post = A.Posterior(led, blocks)
    assert {"w1.p00", "w1.k", "kid"} <= set(post.est)
    eligible = [n for n in post.est if not led.nodes[n].port_of and not A.wave_node(led.nodes[n])]
    assert seen[0] == len(eligible) and "kid" in eligible
    opts = A.ledger_options(led, blocks, ST.derive(led, blocks, noise), inc, post=post)
    assert not {o.node for o in opts} & {"w1.p00", "w1.k"}
    assert "kid" in {o.node for o in opts if o.kind == "seed"}


def test_ports_need_two_seeds_and_a_positive_selection_shrunk_belief(sample_db):
    con, inc, _ = _new_epoch(sample_db)
    con.close()
    led, blocks, noise = _state(sample_db)
    post = A.Posterior(led, blocks)
    opts = A.ledger_options(led, blocks, ST.derive(led, blocks, noise), inc, post=post)
    ports = [o for o in opts if o.kind == "port"]
    assert ports and all(o.m > 0 and post.est[o.node][M.COMPOSITE].n >= A.PORT_MIN_SEEDS for o in ports)
    old = [n for n in post.est if led.nodes[n].base != NEW]
    assert len(ports) < len(old)                                    # the 1-seed / negative tail is not offered


def test_tests_need_the_entry_rule(sample_db):
    led, blocks, noise = _state(sample_db)
    st = ST.derive(led, blocks, noise)
    post = A.Posterior(led, blocks)
    opts = A.ledger_options(led, blocks, st, ST.chain(led, blocks, noise)[-1], post=post)
    tests = [o for o in opts if o.kind == "test"]
    inc = ST.chain(led, blocks, noise)[-1]
    screening = [n for n in post.est if st.get(n) is ST.Status.SCREENING]
    assert {o.node for o in tests} == {n for n in screening if A.test_entry_ok(n, post, inc, 16)} | (
        set(A.retest_candidates(led, blocks, post)) & set(post.est))
    assert all(o.m > 0 for o in tests) and len(tests) < len(screening)          # no negative candidate


def test_value_floor_cuts_the_low_value_tail_relative_to_the_best_screen_or_idea():
    screen = Option("screen", "s", "explore", -0.004, 0.0015 ** 2, S2 / 3, 3.0, 3, tail=True)
    idea = Option("idea", None, "explore", -0.004, 0.0015 ** 2, S2 / 3, 3.5, 3, p=0.8)
    tail = [Option("port", f"o{i}", "explore", -0.002, 0.0004 ** 2, S2 / 3, 3.1, 3, 0.5) for i in range(30)]
    weak = Option("test", "w", "explore", -0.002, 0.001 ** 2, S2 / 5, 10.0, 10)
    good = Option("port", "g", "explore", 0.004, 0.002 ** 2, S2 / 3, 3.1, 3, 0.5)
    acts = A.rank([screen, idea, weak, good, *tail], free_slots=200, llm_free=1)
    lo, _ = A.floors(A.score([screen, idea, weak, good, *tail]))
    assert {a.node for a in acts if a.kind != "idea"} == {"s", "g"} and lo > 0
    assert all(a.rate >= lo for a in acts if a.kind not in ("idea", "screen"))
    assert len(A.rank([*tail], free_slots=200, llm_free=0, floor=0.0)) == 30      # without the floor: all of them


def test_ports_open_cap_and_the_wave_floor_and_soft_share():
    scr = Option("screen", "s", "explore", 0.0, 0.002 ** 2, S2 / 3, 3.0, 3)
    ports = [Option("port", f"o{i}", "explore", 0.004, 0.002 ** 2, S2 / 3, 3.1, 3, 0.5) for i in range(5)]
    acts = A.rank([scr, *ports], free_slots=64, llm_free=0, caps={"port": 2})
    assert sum(a.kind == "port" for a in acts) == 2
    wave = Option("wave", "c", "factorial", 0.003, 0.002 ** 2, S2 / 16, 8.0, 8)
    acts = A.rank([scr, wave], free_slots=11, llm_free=0, soft={"wave": 0})
    assert [a.kind for a in acts] == ["screen", "wave"]                  # soft: the slots nobody else wants
    acts = A.rank([scr, wave, *ports], free_slots=11, llm_free=0, soft={"wave": 0})
    assert "wave" not in [a.kind for a in acts]                          # ports take the slots first
    weak = Option("wave", "c", "factorial", -0.004, 0.0005 ** 2, S2 / 16, 8.0, 8)
    assert not [a for a in A.rank([scr, weak], free_slots=64, llm_free=0) if a.kind == "wave"]   # v3-c5: cap 0
    relaxed = [a for a in A.rank([scr, weak], free_slots=64, llm_free=0, relax_caps={"wave": 1}) if a.kind == "wave"]
    assert relaxed and all(a.pre for a in relaxed)       # under the wave floor: idle slots only (relax, preemptible)


def test_a_new_idea_nodes_screen_is_valued_from_the_heavy_tail():
    g = Option("screen", "s", "explore", -0.004, 0.0015 ** 2, S2 / 3, 3.0, 3)
    t = Option("screen", "s", "explore", -0.004, 0.0015 ** 2, S2 / 3, 3.0, 3, tail=True)
    assert A.value(t, 0, 0, 0) == pytest.approx(A.ei_t(t.m, t.v, t.V, 0)) and A.value(t, 0, 0, 0) > 3 * A.value(g, 0, 0, 0)


def _node(nid, parent, notes="", base="b"):
    return D.Node(nid, parent, None, "explore", "f", "explore", "root", 0.0, "", notes, base,
                  (notes.split("port_of=")[1].split()[0] if "port_of=" in notes else ""))


class _Post:
    epoch = "b"

    def __init__(self, est, m):
        self.est, self._m = est, m

    def belief(self, n):
        return self._m[n], 1e-6


def test_port_p_starts_at_the_prior_and_learns_from_measured_ports():
    nodes = {n.id: n for n in [_node("r", None), _node("inc", "r", "port_of=I"), _node("a", "inc", "port_of=x1"),
                               _node("b", "inc", "port_of=x2"), _node("c", "inc", "port_of=x3"),
                               _node("d", "inc", "port_of=x4"), _node("e", "inc", "port_of=x5 for_wave=w1")]}
    led = D.Ledger(nodes, (), (), frozenset(), ((0.0, "r"),), bases=((0.0, "r", "b"),))
    est = {n: {M.COMPOSITE: M.Estimate(0.0, 1e-6, k, 0)} for n, k in (("a", 3), ("b", 3), ("c", 1), ("e", 3))}
    st = {"inc": ST.Status.INCUMBENT, "a": ST.Status.SCREENING, "b": ST.Status.SCREENING, "c": ST.Status.SCREENING,
          "d": ST.Status.VOID, "e": ST.Status.SCREENING}
    p, paid, n = A.port_p(led, st, _Post(est, {"a": 0.002, "b": -0.001, "e": 0.01}), A.Costs())
    assert (paid, n) == (1, 3)                                      # a paid; b negative; d did not graft; c pending
    assert p == pytest.approx((A.PORT_P * A.PORT_P_N + 1) / (A.PORT_P_N + 3))
    assert A.port_p(led, {}, _Post({}, {}), A.Costs())[0] == A.PORT_P
    assert A.open_ports(led, st, A.Costs()) == 3                     # a, b, c have no finished runs here


def test_budgets_count_ports_being_screened_and_wave_runs_in_flight():
    nodes = {n.id: n for n in [_node("r", None), _node("pt", "r", "wave=w1 point=0 block=0 seed=5"),
                               _node("q", "r", "port_of=x")]}
    runs = tuple(D.Run(f"pt:{s}", "pt", s, "screen", "running", 0.0, 0.0, 0.0, "", 0) for s in range(3))
    led = D.Ledger(nodes, runs, (), frozenset(), ((0.0, "r"),), bases=((0.0, "r", "b"),))
    b = A.budgets(led, {"q": ST.Status.QUEUED}, free_slots=37, costs=A.Costs())
    assert b["caps"]["port"] == A.PORT_OPEN_MAX - 1 and b["soft"]["wave"] == int(A.WAVE_SHARE * 40) - 3


# ---------------------------------------------------------------------------------------- idle fill (v3-rc)
def _fill_world(db):
    con, inc, t = _new_epoch(db)
    port = "p-" + inc
    _add_node(con, "far1", port, t + 3, commit="far1")                 # one seed: 4 of the port's CRN seeds left
    _add_runs(con, "far1", t + 4, seeds=(1,))
    _add_node(con, "kid2", port, t + 3, commit="kid2")
    _add_runs(con, "kid2", t + 4, seeds=(1, 2))
    _add_node(con, "w9.p00", port, t + 3, notes="wave=w9 point=0 block=0 seed=3", commit="w9p0")
    _add_runs(con, "w9.p00", t + 4, seeds=(3,))
    _add_node(con, "q1", port, t + 5, commit="q1")                      # queued, never run
    con.commit()
    con.close()
    led, blocks, noise = _state(db)
    return led, blocks, ST.derive(led, blocks, noise), inc, A.Posterior(led, blocks)


def test_idle_fill_calibrates_the_far_class_then_seeds_then_screens(sample_db):
    led, blocks, st, inc, post = _fill_world(sample_db)
    acts = A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs())
    assert acts and {a.kind for a in acts} == {"fill"} and all(a.value == 0 for a in acts)
    by = {a.node: a for a in acts}
    assert "w9.p00" not in by and "p-" + inc not in by                  # never a wave point, never the incumbent
    assert by["far1"].runs == 4 and by["far1"].why.startswith("idle (a)")   # CRN pairs with its parent first
    assert by["q1"].why.startswith("idle (c)") and by["q1"].runs == A.Costs().idea_runs
    seeds: dict[str, set[int]] = {}
    for r in led.runs:
        seeds.setdefault(r.node, set()).add(r.seed)
    assert all(len(seeds.get(a.node, ())) + a.runs <= A.Costs().max_seeds for a in acts)
    whys = [a.why[:8] for a in acts]
    assert whys == sorted(whys)                                         # (a) before (b) before (c)
    few = A.idle_fill(led, blocks, st, inc, post, 3, set(), A.Costs())
    assert sum(a.runs for a in few) == 3                                # never more runs than idle slots
    rest = A.idle_fill(led, blocks, st, inc, post, 200, {"far1", "q1"}, A.Costs())
    assert rest and not {a.node for a in rest} & {"far1", "q1"}        # nodes a ranked action took are left out


def test_idle_fill_skips_calibration_once_the_far_fit_has_its_dfs(sample_db, monkeypatch):
    led, blocks, st, inc, post = _fill_world(sample_db)
    monkeypatch.setattr(M, "NEAR_MIN_DF", 0)
    acts = A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs())
    assert acts and not any(a.why.startswith("idle (a)") for a in acts)
    monkeypatch.setattr(A, "IDLE_FILL", False)
    assert A.idle_fill(led, blocks, st, inc, post, 200, set(), A.Costs()) == []


def test_idle_fill_runs_are_preemptible_only(ctx, tmp_path):
    import dataclasses
    from types import SimpleNamespace

    from loop.adapters.sim import SimAdapter
    from loop.contracts import Stage
    from loop.runner import RunBook
    from loop.spec import PoolCfg, SlurmCfg
    from loop.store import SqliteStore
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=(
        PoolCfg("primary", slots=2), PoolCfg("idle", slots=1, preemptible=True)))))
    store = SqliteStore(tmp_path / "l.db", ctx.task_id)
    rb = RunBook(ctx, store, SimAdapter(ctx))
    n = SimpleNamespace(id="x")
    rb.fill_only = {("x", 7)}
    assert rb.pool(n, Stage.SCREEN, 7) == "idle" and rb.pool(n, Stage.SCREEN, 8) == "idle"
    rb.used = lambda: {"primary": 0, "idle": 1}                        # the preemptible pool is full
    assert rb.pool(n, Stage.SCREEN, 7) == "primary"                     # 10-22: no open preemptible slot -> a free primary one
    rb.high_demand = 1
    assert rb.pool(n, Stage.SCREEN, 7) == ""                            # ... but never while a HIGH run waits for primary
    assert rb.pool(n, Stage.SCREEN, 8) == ""                            # (an ordinary LOW run waits then too)
    rb.high_demand = 0
    assert rb.pool(n, Stage.SCREEN, 8) == "primary"                     # an ordinary LOW run may take primary
    store.close()
