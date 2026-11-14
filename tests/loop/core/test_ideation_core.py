"""loop.core.ideation on the integrated core: it reads the tick's core posterior (no beliefs of its own), and the
README FM reference never shows a banned (GenBio) model."""

from __future__ import annotations

import dataclasses

from loop.core import acquire as A
from loop.core import data as D
from loop.core import ideation as ID
from loop.core.seeds import reconstruct
from loop.store import SqliteStore

from ..test_tick import _ledger
from .conftest import TASK

README = """### Baselines

| # | Name | Description | final score | classification | segmentation | progression | mutation | survival | robustness |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | GenBio-PathFM | GenBio-PathFM ViT-G/16 | **0.6926** | 0.8213 | 0.6100 | 0.6630 | 0.6232 | 0.6312 | 0.6525 |
| 2 | UNI-2-h | MahmoodLab UNI-2-h ViT-H/14 | 0.6876 | 0.8161 | 0.6323 | 0.6771 | 0.6408 | 0.6105 | 0.5837 |
| 4 | H-optimus-0 | H-optimus-0 ViT-G/14-reg | 0.6785 | 0.8082 | 0.5916 | 0.6683 | 0.6485 | 0.6059 | 0.5838 |
"""


def test_fm_reference_excludes_genbio(tmp_path):
    (p := tmp_path / "README.md").write_text(README)
    fm = ID.fm_reference(p)
    assert fm["cls"] == (0.8161, "UNI-2-h") and fm["auc"] == (0.6485, "H-optimus-0")
    assert not any("genbio" in name.lower() for _, name in fm.values())


def test_ideation_reads_the_core_posterior(ctx, tmp_path):
    db = _ledger(tmp_path / "ledger.db")
    store = SqliteStore(db, TASK)
    led = D.load(db, TASK)
    blocks = reconstruct(led)
    post = A.Posterior(led, blocks)
    ide = ID.Ideation(dataclasses.replace(ctx, task_id=TASK), store)
    v = ide.view(led, post)
    assert v.noise is post.noise and v.weights is post.weights
    assert {n: (b.m, b.v, b.n) for n, b in v.beliefs.items()} == post.beliefs()
    store.close()
