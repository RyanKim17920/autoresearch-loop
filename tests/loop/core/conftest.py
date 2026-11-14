"""Fixtures for loop.core: a SQLite ledger built from ``fixtures/ledger_sample.json`` (a SYNTHETIC ledger in the shape
of a real one: root, the audited carriers, confirmed children and screened-out siblings, with their runs and the
run_finished / seeds_drawn / audit / incumbent log rows; scores are drawn from ``y = base + U(node) + b(seed) + e``,
names, commits, paths and job ids are placeholders), and a synthetic generator."""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from loop.core.data import AXES, Obs

SAMPLE = Path(__file__).parent / "fixtures" / "ledger_sample.json"
TASK = "nanopath-v2-rescore"


def build_db(path: Path, sample: Mapping) -> Path:
    con = sqlite3.connect(path)
    for table in ("nodes", "runs"):
        cols = sorted({k for row in sample[table] for k in row})
        con.execute(f"CREATE TABLE {table} ({', '.join(f'{c!r}' for c in cols)})")
        con.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in cols)})",
                        [[row.get(c) for c in cols] for row in sample[table]])
    con.execute("CREATE TABLE log (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, ts REAL, event TEXT, "
                "fields TEXT)")
    con.executemany("INSERT INTO log (task_id, ts, event, fields) VALUES (?, ?, ?, ?)",
                    [(r["task_id"], r["ts"], r["event"], r["fields"]) for r in sample["log"]])
    con.commit()
    con.close()
    return path


@pytest.fixture(scope="session")
def sample() -> dict:
    return json.loads(SAMPLE.read_text())


@pytest.fixture
def sample_db(tmp_path, sample) -> Path:
    return build_db(tmp_path / "ledger.db", sample)


def synth(rng: random.Random, mu: Mapping[str, float], seeds: Mapping[str, Sequence[int]], *,
          sigma: float = 0.003, rho: float = 0.4, stage: str = "screen") -> list[Obs]:
    """``y[c,s] = mu[c] + b[s] + e[c,s]``, ``Var b = rho sigma²``, ``Var e = (1-rho) sigma²``; every axis = y."""
    b = {s: rng.gauss(0, (rho ** 0.5) * sigma) for ss in seeds.values() for s in ss}
    out = []
    for node, ss in seeds.items():
        for s in ss:
            y = mu[node] + b[s] + rng.gauss(0, ((1 - rho) ** 0.5) * sigma)
            out.append(Obs(f"{node}:{stage}:{s}", node, s, stage, "select", {a: y for a in AXES}, y, 1.0, 0.0, 0))
    return out
