"""loop.cutover: the new base's calibration runs enter the ledger as the re-baseline of a new epoch."""

from __future__ import annotations

import json
import random

from loop import cutover as CO
from loop.core import acquire as A
from loop.core import data as D
from loop.core.backtest import fit_at
from loop.store import SqliteStore

from .core.conftest import TASK
from .test_tick import _ledger

SEEDS = (700001, 700002, 700003, 700004, 700005)
#: a synthetic new base: its root, a port of n0655 onto it, and a small-change child of the port
ARMS = (CO.Arm("root-065b87f", "065b87f0000000000000000000000000000000aa", None, "baseline", "root",
               note="base root"),
        CO.Arm("n0655@065b87f", "065b87f0000000000000000000000000000000bb", "root-065b87f", "", "n0655",
               port_of="n0655", note="the port"),
        CO.Arm("n0655-kde@065b87f", "065b87f0000000000000000000000000000000cc", "n0655@065b87f", "kde", "kde",
               note="the small-change calibration arm"))


def _calib(root, rng):
    """root / port / small-change child; a shared seed effect, the child almost identical to its parent."""
    b = {s: rng.gauss(0, 0.004) for s in SEEDS}
    small = {s: rng.gauss(0, 0.0008) for s in SEEDS}
    for arm, mu in (("root", 0.66), ("n0655", 0.6686), ("kde", 0.6683)):
        for s in SEEDS:
            y = mu + (b[s] if arm == "root" else 0.6 * b[s] + small[s]) + rng.gauss(0, 0.0005 if arm != "root" else 0.003)
            if arm == "kde":
                y += small[s] * 0.1
            (root / arm / str(s)).mkdir(parents=True)
            (root / arm / str(s) / "summary.json").write_text(json.dumps(
                {**{k: y for k in D.PROBE_KEYS.values()}, "train_loop_wall_seconds": 3600, "final_probe_wall_seconds": 0}))


def test_ingest_starts_the_epoch_and_satisfies_the_rebaseline(tmp_path):
    db = _ledger(tmp_path / "copy.db")
    _calib(tmp_path / "calib", random.Random(3))
    arms = [CO.Arm(a.node, a.commit, a.parent, a.family, str(tmp_path / "calib" / d), a.port_of, a.note)
            for a, d in zip(ARMS, ("root", "n0655", "kde"))]
    store = SqliteStore(db, TASK)
    lines = CO.ingest_rebase(store, arms, SEEDS, 3e9)
    assert len(lines) == 3 + 15 and CO.ingest_rebase(store, arms, SEEDS, 3e9) == []      # idempotent
    store.close()
    led = D.load(db, TASK)
    assert led.epoch == ARMS[0].commit and led.port("n0655") == "n0655@065b87f"
    assert {o.node for o in led.in_epoch().obs} == {a.node for a in arms} and len(led.in_epoch().obs) == 15
    assert all(abs(o.panel - o.axes["cls"]) < 1e-12 for o in led.in_epoch().obs)        # every axis = y: panel = y
    assert A.rebase_owed(led, "n0655") == {}                                               # already re-baselined
    cur = led.in_epoch()
    assert cur.change_parents()["n0655-kde@065b87f"] == "n0655@065b87f" and "n0655@065b87f" not in cur.change_parents()
    near, _ = fit_at(cur, cur.change_parents())
    far, _ = fit_at(cur, cur.far_parents())
    assert near.rho[D.PANEL] > far.rho[D.PANEL]                     # small change: more of the seed effect shared
