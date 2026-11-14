"""Base epochs (docs/loop-core-v3.md §8): a new base root starts an epoch; nothing pairs across it, the re-baseline
is acquisition's first action, an old-epoch candidate is only a port option, and the incumbent's port takes over."""

from __future__ import annotations

import json
import random
import sqlite3

from loop.core import acquire as A
from loop.core import data as D
from loop.core import status as ST
from loop.core.backtest import fit_at
from loop.core.seeds import reconstruct

TASK = "nanopath-v2-rescore"
NEW = "065b87f0000000000000000000000000000000000"


def _add_node(con, nid, parent, ts, notes="", commit=None):
    con.execute("INSERT INTO nodes (id, task_id, parent_id, strategy, family, created_ts, status, notes, \"commit\") "
                "VALUES (?, ?, ?, 'explore', 'f', ?, 'queued', ?, ?)", (nid, TASK, parent, ts, notes, commit or nid))


def _add_runs(con, nid, ts, status="done", seeds=range(1, 6)):
    row = con.execute("SELECT scores, aggregate, gpu_h FROM runs WHERE node_id='root' AND status='done'").fetchone()
    rng = random.Random(nid)
    for s in seeds:
        rid = f"{nid}:baseline:{s}"
        scores = {k: v + 0.003 * rng.gauss(0, 1) + 0.002 * ((s * 7) % 5 - 2) for k, v in json.loads(row[0]).items()}
        con.execute("INSERT INTO runs (id, task_id, node_id, seed, stage, status, scores, aggregate, gpu_h, "
                    "submitted_ts, output_dir, resumed) VALUES (?, ?, ?, ?, 'baseline', ?, ?, ?, ?, ?, '', 0)",
                    (rid, TASK, nid, s, status, json.dumps(scores), row[1] + 0.003 * rng.gauss(0, 1) +
                     0.002 * ((s * 7) % 5 - 2), row[2], ts))
        if status == "done":
            con.execute("INSERT INTO log (task_id, ts, event, fields) VALUES (?, ?, 'run_finished', ?)",
                        (TASK, ts + 1, json.dumps({"run_id": rid, "status": "done"})))


def _state(db):
    led = D.load(db, TASK)
    blocks = reconstruct(led)
    noise, _ = fit_at(led, {k: n.parent for k, n in led.nodes.items()})
    return led, blocks, noise


def test_one_epoch_ledger_is_unchanged(sample_db):
    led, blocks, noise = _state(sample_db)
    assert len(led.bases) == 1 and all(n.base == led.epoch for n in led.nodes.values())
    assert all(o.base == led.epoch for o in led.obs) and len(led.in_epoch().obs) == len(led.obs)
    inc = ST.chain(led, blocks, noise)[-1]
    assert A.rebase_owed(led, inc) == {}
    assert {o.kind for o in A.ledger_options(led, blocks, ST.derive(led, blocks, noise), inc)} <= {"seed", "test", "idea"}


def test_cutover_rebases_first_then_ports_old_candidates(sample_db):
    led, blocks, noise = _state(sample_db)
    inc = ST.chain(led, blocks, noise)[-1]
    t = max(r.submitted_ts for r in led.runs) + 100
    con = sqlite3.connect(sample_db)
    _add_node(con, "root-065b87f", None, t, commit=NEW)
    con.commit()
    led, blocks, noise = _state(sample_db)
    assert led.epoch == NEW and led.base_root == "root-065b87f" and led.in_epoch().obs == ()
    st = ST.derive(led, blocks, noise)
    assert st[inc] is ST.Status.INCUMBENT                                   # not ported yet: still the old one
    [rebase] = A.ledger_options(led, blocks, st, inc)
    assert rebase.kind == "rebase" and rebase.runs == 10 and A.rebase_owed(led, inc) == {"root-065b87f": 5, inc: 5}
    assert [a.kind for a in A.rank([rebase], free_slots=10, llm_free=0)] == ["rebase"]

    _add_node(con, "p-" + inc, "root-065b87f", t + 1, notes=f"port_of={inc}")
    _add_runs(con, "root-065b87f", t + 2)
    _add_runs(con, "p-" + inc, t + 2)
    con.commit()
    con.close()
    led, blocks, noise = _state(sample_db)
    assert led.port(inc) == "p-" + inc and led.nodes["p-" + inc].base == NEW
    assert {o.node for o in led.in_epoch().obs} == {"root-065b87f", "p-" + inc}
    st = ST.derive(led, blocks, noise)
    assert st["p-" + inc] is ST.Status.INCUMBENT and st[inc] is ST.Status.PASSED
    assert A.rebase_owed(led, inc) == {}
    opts = A.ledger_options(led, blocks, st, inc)
    old = [o for o in opts if o.node and led.nodes[o.node].base != NEW]
    assert old and all(o.kind == "port" and o.p == A.PORT_P for o in old)      # never a seed / test on the old base
    assert {o.kind for o in opts} >= {"port", "idea"}
    acts = A.rank(opts, free_slots=64, llm_free=4)
    assert all(a.kind in ("port", "idea") for a in acts)


def test_a_queued_port_is_screened_with_its_originals_belief_and_its_original_is_no_longer_a_port_option(sample_db):
    led, blocks, noise = _state(sample_db)
    inc = ST.chain(led, blocks, noise)[-1]
    t = max(r.submitted_ts for r in led.runs) + 100
    con = sqlite3.connect(sample_db)
    _add_node(con, "root-065b87f", None, t, commit=NEW)
    _add_node(con, "p-" + inc, "root-065b87f", t + 1, notes=f"port_of={inc}")
    _add_runs(con, "root-065b87f", t + 2)
    _add_runs(con, "p-" + inc, t + 2)
    con.commit()
    led, blocks, noise = _state(sample_db)
    opts = A.ledger_options(led, blocks, ST.derive(led, blocks, noise), inc)
    best = max((o for o in opts if o.kind == "port"), key=lambda o: o.m)
    _add_node(con, f"{best.node}@new", "p-" + inc, t + 3, notes=f"port_of={best.node}")
    _add_node(con, "pt", "p-" + inc, t + 3, notes="wave=w1 point=0 block=0 seed=41")        # a wave point: its wave
    con.commit()
    con.close()
    led, blocks, noise = _state(sample_db)
    st = ST.derive(led, blocks, noise)
    assert st[f"{best.node}@new"] is ST.Status.QUEUED
    opts = A.ledger_options(led, blocks, st, inc)
    assert best.node not in {o.node for o in opts if o.kind == "port"}
    [scr] = [o for o in opts if o.kind == "screen"]
    post = A.Posterior(led, blocks)
    m, v = post.belief(best.node)
    assert scr.node == f"{best.node}@new" and scr.runs == A.Costs().idea_runs and scr.m == m and scr.v == v
    assert "screen" in {a.kind for a in A.rank(opts, free_slots=64, llm_free=4)}


def test_one_wave_option_per_carrier_with_knobs_and_none_while_its_wave_is_open(sample_db):
    from loop.core import factorial as F
    led, blocks, noise = _state(sample_db)
    inc = ST.chain(led, blocks, noise)[-1]
    t = max(r.submitted_ts for r in led.runs) + 100
    con = sqlite3.connect(sample_db)
    _add_node(con, "root-065b87f", None, t, commit=NEW)
    _add_node(con, "p-" + inc, "root-065b87f", t + 1, notes=f"port_of={inc}")
    _add_runs(con, "root-065b87f", t + 2)
    _add_runs(con, "p-" + inc, t + 2)
    con.commit()
    led, blocks, noise = _state(sample_db)
    bank = [F.BankKnob(F.Knob(f"k{i}", 0.1, 0.2), 0.002, 1e-6, "declared") for i in range(5)]
    st = ST.derive(led, blocks, noise)
    opts = A.ledger_options(led, blocks, st, inc, knobs={"p-" + inc: bank, inc: bank})   # inc: an old-base carrier
    [w] = [o for o in opts if o.kind == "wave"]
    assert w.node == "p-" + inc and w.runs == 16 + 2 and w.m > 0 and w.lane == "factorial"   # + 2 block controls
    assert "wave" in {a.kind for a in A.rank(opts, free_slots=64, llm_free=4, wave_floor=0.0)}
    retro = [F.BankKnob(b.knob, b.m, b.v, "retro") for b in bank]       # a retro bank: the factorial cell prior
    post = A.Posterior(led, blocks)
    [rw] = [o for o in A.ledger_options(led, blocks, st, inc, post=post, knobs={"p-" + inc: retro}) if o.kind == "wave"]
    m, v, _ = post.prior.cell((A.cell_carrier(post.led, "p-" + inc), F.FACTORIAL_FAMILY, "factorial"))
    assert (rw.m, rw.v) == (m, v) and rw.m < w.m
    _add_node(con, "w1.p00", "p-" + inc, t + 3, notes="wave=w1 point=0 block=0 seed=100100")
    con.commit()
    con.close()
    led, blocks, noise = _state(sample_db)
    opts = A.ledger_options(led, blocks, ST.derive(led, blocks, noise), inc, knobs={"p-" + inc: bank})
    assert not [o for o in opts if o.kind == "wave"]                       # one open wave per carrier
