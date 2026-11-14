"""Store, spec loader and import-hygiene tests (no network, no SLURM)."""

from __future__ import annotations

import copy
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from loop import contracts as C
from loop.spec import SeedPlan, load_context
from loop.store import SqliteStore

REPO = Path(__file__).resolve().parents[2]
TASK = "fake-loop"


def node(id_="n1", *, task=TASK, parent="root", **kw):
    base = dict(id=id_, task_id=task, commit="c-" + id_, parent_id=parent,
                strategy=C.Strategy.EXPLORE,
                operator=C.Operator.DRAFT, family="optimization", mechanism_tag="lr_down",
                diff_hash="h-" + id_, status=C.NodeStatus.QUEUED,
                levers=C.LeverTags({"effective_lr": -1}), created_ts=1.0)
    base.update(kw)
    return C.Node(**base)


def run(node_id="n1", seed=11, stage=C.Stage.SCREEN, task=TASK, **kw):
    kw.setdefault("status", C.RunStatus.DONE)
    return C.Run(id=C.Run.make_id(node_id, stage, seed), task_id=task, node_id=node_id,
                 seed=seed, stage=stage, **kw)


# -- kernel / spec -----------------------------------------------------------

def test_taskspec_loop_field_default():
    from kernel.model import TaskSpec
    assert TaskSpec.load(REPO / "tasks" / "fake.yaml").loop == {}


def test_context_fake_loop(ctx):
    assert ctx.task_id == "fake-loop" and ctx.cfg.axes == ("alpha", "beta", "gamma")
    assert sum(ctx.cfg.weights.values()) == pytest.approx(1.0)
    assert ctx.cfg.sigma("alpha") == 0.005 and ctx.cfg.sigma(C.PANEL) == 0.004
    assert ctx.noise.seed_transfer_sd == 0.004        # from THIS spec, not config
    assert ctx.cfg.families["architecture"].init_check
    assert ctx.cfg.prior.families["architecture"].mean == -0.02


def test_state_dir_resolution(fake_loop_spec, tmp_path):
    assert load_context(fake_loop_spec, state_dir=tmp_path).state_dir == str(tmp_path)
    with pytest.raises(ValueError, match="no state dir"):
        load_context(fake_loop_spec)
    v2 = load_context(REPO / "tasks/nanopath_v2.yaml", require_measured=False)
    assert v2.state_dir.endswith("/loop")


def test_unmeasured_refused_on_nanopath_v2(unmeasured_v2_spec):
    with pytest.raises(C.UnmeasuredLoopNoise, match="python -m loop calibrate"):
        load_context(unmeasured_v2_spec, state_dir="/tmp/x-unused")
    ok = load_context(unmeasured_v2_spec, state_dir="/tmp/x-unused", require_measured=False)
    assert len(ok.cfg.axes) == 6


def _variant(tmp_path, edit):
    data = yaml.safe_load((REPO / "tasks/fake_loop.yaml").read_text())
    data = copy.deepcopy(data)
    edit(data)
    p = tmp_path / "v.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


@pytest.mark.parametrize("edit,match", [
    (lambda d: d["loop"].update(bogus=1), r"loop\.bogus"),
    (lambda d: d["loop"].setdefault("promotion", {}).update(p_promotee=1), r"loop\.promotion\.p_promotee"),
    (lambda d: d["loop"]["families"]["architecture"].update(kes=[]), r"loop\.families\.architecture\.kes"),
    (lambda d: d["loop"]["axis_sd"].pop("gamma"), "axis_sd must map exactly"),
    (lambda d: d["loop"].update(seeds={"screen": [1, 2], "confirm_pool": [2, 3]}), "in both"),
    (lambda d: d["loop"].update(axes=["alpha"]), "derived from slots"),
])
def test_loop_block_refusals(tmp_path, edit, match):
    with pytest.raises(ValueError, match=match):
        load_context(_variant(tmp_path, edit), state_dir=tmp_path)


def test_one_unmeasured_axis_refuses(tmp_path):
    p = _variant(tmp_path, lambda d: d["loop"]["axis_sd"].update(beta="UNMEASURED"))
    with pytest.raises(C.UnmeasuredLoopNoise, match=r"axis_sd\.beta"):
        load_context(p, state_dir=tmp_path)


def test_axes_fall_back_to_weights(tmp_path):
    def edit(d):
        d["slots"]["vector"] = []
    assert load_context(_variant(tmp_path, edit), state_dir=tmp_path).cfg.axes == \
        ("alpha", "beta", "gamma")


# -- store round trips -------------------------------------------------------

def test_node_roundtrip(store):
    n = node(second_parent_id="x", confirm_alt=0.01, embedding=(0.5, 0.25), notes="hi", prior_bonus=0.001)
    store.put_node(n)
    assert store.get_node("n1") == n
    up = store.update_node("n1", status=C.NodeStatus.SCREENING, confirm_e=2.5)
    assert up.status is C.NodeStatus.SCREENING and up.confirm_e == 2.5 and up.updated_ts > 0
    assert store.nodes(status=[C.NodeStatus.SCREENING]) == [up]
    assert store.nodes(status=[]) == [] and store.nodes(island="zzz") == []
    assert store.find_by_diff_hash("h-n1").id == "n1" and store.find_by_diff_hash("nope") is None
    with pytest.raises(ValueError):
        store.update_node("n1", task_id="other")
    with pytest.raises(KeyError):
        store.update_node("missing", notes="x")


def test_run_roundtrip_and_duplicate(store):
    r = run(scores={"alpha": 0.5, "beta": 0.4}, aggregate=0.45, gpu_h=0.8, job_id="j1",
            log_tail="tail", void_type=None)
    store.put_run(r)
    assert store.get_run(r.id) == r
    with pytest.raises(sqlite3.IntegrityError):
        store.put_run(r)
    # same (node, seed, stage) under a different id text is also a violation
    with pytest.raises(sqlite3.IntegrityError):
        store.put_run(C.Run(id="other-id", task_id=TASK, node_id="n1", seed=11,
                            stage=C.Stage.SCREEN, status=C.RunStatus.SUBMITTING))
    u = store.update_run(r.id, status=C.RunStatus.VOID, void_type=C.VoidType.INFRA, attempt=1)
    assert u.void_type is C.VoidType.INFRA and u.scores == r.scores
    assert store.runs(node_id="n1", status=[C.RunStatus.VOID]) == [u]
    assert store.runs(stage=C.Stage.CONFIRM) == []


def test_resume_safe_after_reopen(ctx, tmp_path):
    db = tmp_path / "r.db"
    s1 = SqliteStore(db, ctx.task_id)
    s1.put_run(run(status=C.RunStatus.SUBMITTING))
    s1.close()
    s2 = SqliteStore(db, ctx.task_id)
    assert [r.status for r in s2.runs()] == [C.RunStatus.SUBMITTING]
    with pytest.raises(sqlite3.IntegrityError):
        s2.put_run(run(status=C.RunStatus.SUBMITTING))


def test_other_records_roundtrip(store):
    h = C.HypothesisRecord(
        id="h1", task_id=TASK, kind=C.HypothesisKind.META, claim="train weaker",
        mechanism_tag="weaker", family="optimization",
        predictions={C.PANEL: C.AxisPrediction(1, 0.01), "alpha": C.AxisPrediction(-1, 0.002)},
        falsification="no gain", levers=C.LeverTags({"effective_lr": -1, "regularization": -1}),
        source="review", review_id="rv1", created_ts=5.0)
    store.put_hypothesis(h)
    assert store.get_hypothesis("h1") == h
    up = store.update_hypothesis("h1", claim="train much weaker")
    assert store.hypotheses(kind=C.HypothesisKind.META) == [up] and up.claim == "train much weaker"
    assert store.hypotheses(kind=C.HypothesisKind.TRIAL) == []
    store.link("h1", "n1", sign=1, auto=True)
    store.link("h1", "n2", sign=-1, auto=False)
    assert store.links("h1") == [("n1", 1), ("n2", -1)]
    with pytest.raises(ValueError):
        store.link("h1", "n3", sign=0, auto=True)

    pm = C.Postmortem("p1", TASK, "n1", "loss", "why", "because", C.LeverTags({"capacity": 1}),
                      ("h1",), "try less", 2.0)
    hn = C.HandoffNote("ho1", TASK, "n1", "did", "learned", "worried", 2.0)
    rv = C.ReviewReport("rv1", TASK, "cadence", "sum", ("f1", "f2"), ("h1",),
                        (("h1", "probe"), (None, "global")), {"optimization": 0.001}, 25, 9.0)
    store.put_postmortem(pm); store.put_handoff(hn); store.put_review(rv)
    assert store.postmortems(node_id="n1") == [pm] and store.postmortems(node_id="zz") == []
    assert store.handoffs() == [hn] and store.reviews() == [rv]

    pull = C.ArmPull(TASK, C.Strategy.EXPLOIT, "n1", 0.5, 0.9, 7.0)
    store.record_pull(pull)
    assert store.pulls() == [pull]

    assert store.get_state("k", 5) == 5
    store.set_state("k", {"a": [1, 2.5], "b": None})
    assert store.get_state("k") == {"a": [1, 2.5], "b": None}
    store.log("evt", x=1, y=C.Stage.SCREEN)
    assert store.events("evt")[0]["y"] == "screen"


# -- task isolation ----------------------------------------------------------

def test_foreign_task_writes_refused(store):
    for call in (lambda: store.put_node(node(task="fake-loop-v1")),
                 lambda: store.put_run(run(task="other")),
                 lambda: store.put_postmortem(C.Postmortem("p", "other", "n", "loss", "", "", C.LeverTags())),
                 lambda: store.put_handoff(C.HandoffNote("h", "", "n", "", "", "")),
                 lambda: store.record_pull(C.ArmPull("other", C.Strategy.EXPLORE, "n", 0, 0, 0))):
        with pytest.raises(C.ForeignTaskError):
            call()
    assert store.nodes() == [] and store.runs() == []


def test_reads_filter_by_task(ctx, tmp_path):
    db = tmp_path / "shared.db"
    a, b = SqliteStore(db, "task-a"), SqliteStore(db, "task-b")
    a.put_node(node(task="task-a")); a.put_run(run(task="task-a"))
    a.set_state("k", 1); a.link("h", "n1", sign=1, auto=False); a.log("e")
    assert b.nodes() == [] and b.get_node("n1") is None and b.runs() == []
    assert b.find_by_diff_hash("h-n1") is None and b.get_state("k") is None
    assert b.links("h") == [] and b.events() == []
    b.put_node(node(id_="n2", task="task-b"))
    assert [n.id for n in b.nodes()] == ["n2"] and [n.id for n in a.nodes()] == ["n1"]


def test_seed_overlap_dataclass():
    with pytest.raises(ValueError, match="in both"):
        SeedPlan(screen=(1, 2), promote=(3,), confirm_pool=(3, 4))


# -- root + table dump -------------------------------------------------------

def test_ensure_root_idempotent(store):
    r1 = store.ensure_root("abc123")
    r2 = store.ensure_root("zzz")
    assert r1 == r2 and r1.id == C.ROOT_NODE_ID and r1.status is C.NodeStatus.BASELINE
    assert r1.commit == "abc123" and len(store.nodes()) == 1


def test_table_dump(store):
    store.ensure_root("abc")
    for seed, (a, b) in {11: (0.50, 0.40), 12: (0.52, 0.42)}.items():
        store.put_run(run(C.ROOT_NODE_ID, seed, C.Stage.BASELINE,
                          scores={"alpha": a, "beta": b}, aggregate=(a + b) / 2))
        store.put_run(run("n1", seed, scores={"alpha": a + 0.01, "beta": b - 0.02},
                          aggregate=(a + b) / 2 - 0.005))
    store.put_run(run("n1", 13, status=C.RunStatus.CRASH))      # never a measurement
    store.put_node(node())
    lines = store.table_dump().splitlines()
    assert lines[0].endswith("axes: alpha beta")
    row = next(l for l in lines if l.startswith("n1 |"))
    cells = [c.strip() for c in row.split("|")]
    assert cells[1:8] == ["root", "main", "optimization", "lr_down", "-effective_lr",
                          "queued", "2"]
    assert cells[8] == "+0.0100 -0.0200" and cells[9] == "-0.0050"
    assert "baseline" in next(l for l in lines if l.startswith("root |"))
    assert "more nodes omitted" in store.table_dump(max_rows=1)


# -- import hygiene ----------------------------------------------------------

def test_no_config_import():
    code = textwrap.dedent("""
        import importlib, pkgutil, sys
        import loop
        skip = {"loop.agents.llm"}
        mods = [m.name for m in pkgutil.walk_packages(loop.__path__, "loop.")]
        for name in mods:
            if name == "loop.__main__":
                continue
            importlib.import_module(name)     # loop.agents.llm imports config only in __init__
        assert "config" not in sys.modules, sorted(m for m in sys.modules if m == "config")
        print(len(mods))
    """)
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]


def test_a_field_added_later_gets_its_column_at_its_default(ctx, tmp_path):
    """An older DB (no runs.pool column) opens, reads its rows at the default, and writes the field."""
    import sqlite3
    from loop.contracts import Run, RunStatus, Stage
    db = tmp_path / "old.db"
    s = SqliteStore(db, ctx.task_id)
    s.put_run(Run(id="root:baseline:11", task_id=ctx.task_id, node_id="root", seed=11,
                  stage=Stage.BASELINE, status=RunStatus.DONE))
    s.close()
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE runs DROP COLUMN pool")
    con.commit()
    con.close()
    s = SqliteStore(db, ctx.task_id)
    assert s.get_run("root:baseline:11").pool == ""
    assert s.update_run("root:baseline:11", pool="idle").pool == "idle"
    s.close()
