"""Labless auto-submit (loop/labless.py): audited nodes only, once each, rate-limited, backoff on
failure, success = a non-null submission_id. No test reaches the real submitter or Labless.

Run: uv run pytest tests/loop -q --confcutdir=tests/loop
"""

from __future__ import annotations

import dataclasses
import sys
from types import SimpleNamespace

import pytest

from loop import labless as L
from loop.contracts import ROOT_NODE_ID, Node, NodeStatus, Run, RunStatus, Stage, Strategy
from loop.spec import LablessCfg
from tests.loop.conftest import InlineExecutor

H = 3600.0


def _ctx(ctx, **kw):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, labless=LablessCfg(enabled=True, **kw)))


def _node(store, nid, status=NodeStatus.CONFIRMED):
    store.ensure_root("c0")
    store.put_node(Node(id=nid, task_id=store.task_id, commit="abc123", parent_id=ROOT_NODE_ID,
                        strategy=Strategy.EXPLORE, operator=None, family="optimization",
                        mechanism_tag="lower_lr", diff_hash=nid, status=status))
    for seed, agg in ((201, 0.60), (202, 0.64), (203, 0.62), (204, 0.61)):
        store.put_run(Run(id=Run.make_id(nid, Stage.AUDIT, seed), task_id=store.task_id, node_id=nid,
                          seed=seed, stage=Stage.AUDIT, status=RunStatus.DONE, aggregate=agg,
                          output_dir=f"/runs/{nid}/{seed}"))


class Fake:
    def __init__(self, results):
        self.results, self.calls = list(results), []

    def __call__(self, ctx, out_dir, name, text, changes=""):
        self.calls.append((out_dir, name, text))
        self.changes = changes
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _sub(ctx, store, fake, now):
    return L.LablessSubmitter(ctx, store, InlineExecutor(), lambda: now[0], submit=fake)


def test_median_audit_run_and_short_name(store):
    _node(store, "n0007")
    assert L.pick_run(store, "n0007").seed == 204            # lower median of .60 .61 .62 .64
    assert L.run_name("n0007") == "loop-v2-n0007" and len(L.run_name("n" * 40)) == 20


def test_only_audited_confirmed_nodes_once_each_and_rate_limited(ctx, store):
    ctx = _ctx(ctx, min_hours_between=4)
    for nid in ("n0001", "n0002"):
        _node(store, nid)
    _node(store, "n0003", NodeStatus.AUDIT_FAILED)
    now, fake = [10 * H], Fake(["sid-1", "sid-2"])
    sub = _sub(ctx, store, fake, now)
    assert sub.step(["n0003"], None) == []                     # not CONFIRMED: never submitted
    assert "submitting n0001" in sub.step(["n0001", "n0002"], None)[0]
    assert "submission_id=sid-1" in sub.step(["n0001", "n0002"], None)[0]
    assert fake.calls[0][:2] == ("/runs/n0001/204", "[aud] lower_lr #0001")
    now[0] += 3 * H
    assert sub.step(["n0001", "n0002"], None) == []            # rate limit: < 4 h since the last
    now[0] += 1.5 * H
    sub.step(["n0001", "n0002"], None), sub.step(["n0001", "n0002"], None)
    now[0] += 100 * H
    assert sub.step(["n0001", "n0002"], None) == []            # both done: exactly one each
    book = store.get_state(L.K_LABLESS)
    assert {k: v["submission_id"] for k, v in book.items()} == {"n0001": "sid-1", "n0002": "sid-2"}
    assert len(fake.calls) == 2
    assert "submitted" in "\n".join(L.status_lines(ctx, store))


def test_failure_backs_off_then_retries_and_null_id_is_failure(ctx, store):
    ctx = _ctx(ctx, min_hours_between=4)
    _node(store, "n0001")
    now, fake = [0.0], Fake([RuntimeError("exit 1"), "sid-9"])
    sub = _sub(ctx, store, fake, now)
    sub.step(["n0001"], None)
    assert "failed (try 1)" in sub.step(["n0001"], None)[0]
    assert store.get_state(L.K_LABLESS)["n0001"]["submission_id"] is None
    assert store.events("labless_failed")
    now[0] += 4.5 * H                                          # backoff 4 h, rate limit 4 h
    sub.step(["n0001"], None)
    sub.step(["n0001"], None)
    assert store.get_state(L.K_LABLESS)["n0001"]["submission_id"] == "sid-9"


def test_disabled_is_a_no_op(ctx, store):
    _node(store, "n0001")
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, labless=LablessCfg(enabled=False)))
    assert _sub(ctx, store, Fake([]), [0.0]).step(["n0001"], None) == []


def test_real_submitter_is_refused_under_the_test_guard(ctx):
    with pytest.raises(RuntimeError, match="refused"):
        L.submit_run(ctx, "/nowhere", "loop-v2-n1", "x")


@pytest.mark.parametrize("rc,receipt,ok", [
    (0, '{"submission_id": "abc", "run": {"validation_errors": []}}', True),
    (0, '{"submission_id": null, "run": {}}', False),                 # a receipt is not proof
    (0, '{"submission_id": "abc", "run": {"validation_errors": ["x"]}}', False),
    (1, '{"submission_id": "abc", "run": {}}', False),                # exit != 0: the POST failed
])
def test_submit_run_requires_exit0_and_a_non_null_id(tmp_path, monkeypatch, rc, receipt, ok):
    monkeypatch.delenv("LOOP_FORBID_SLURM")          # a FAKE submitter script: nothing leaves the box
    (tmp_path / "labless").mkdir()
    (tmp_path / "labless" / "submit_to_labless.py").write_text(
        "import sys, pathlib\nargs = dict(a.split('=', 1) for a in sys.argv[1:])\n"
        f"pathlib.Path(args['output_dir'], 'labless_submission.json').write_text({receipt!r})\n"
        f"assert args['github_token_file'].endswith('.labless_autosubmit.json')\nsys.exit({rc})\n")
    out = tmp_path / "run"
    out.mkdir()
    ctx = SimpleNamespace(cfg=SimpleNamespace(labless=LablessCfg(python=sys.executable)),
                          spec=SimpleNamespace(repo=str(tmp_path)))
    if ok:
        assert L.submit_run(ctx, str(out), "loop-v2-n1", "note") == "abc"
    else:
        with pytest.raises(RuntimeError):
            L.submit_run(ctx, str(out), "loop-v2-n1", "note")


def test_driver_submits_an_audit_passed_node(tmp_path):
    from tests.loop.test_driver import Scripts, _make, _world
    ctx = _ctx(_world(tmp_path))
    driver, store, _ = _make(ctx, Scripts())
    driver.tick()
    _node(store, "n0900")
    store.set_state("driver.audit", {"passed": ["n0900"], "failed": [], "queue": [], "current": None})
    fake = Fake(["sid-x"])
    driver.labless.submit = fake
    driver.tick()
    driver.tick()
    assert store.get_state(L.K_LABLESS)["n0900"]["submission_id"] == "sid-x"
    assert "n0900" in fake.calls[0][2] and fake.calls[0][1] == "[aud] lower_lr #0900" and "lower_lr" in fake.calls[0][2]


def test_recorded_manual_submission_is_backfilled_once_and_never_resubmitted(ctx, store):
    ctx = _ctx(ctx, recorded={"root": {"submission_id": "0c01788397", "run_id": "root:baseline:14"}})
    fake = Fake([])
    sub = _sub(ctx, store, fake, [10 * H])
    for _ in range(3):
        sub.step([ROOT_NODE_ID], None)
    book = store.get_state(L.K_LABLESS)
    assert book == {"root": {"submission_id": "0c01788397", "run_id": "root:baseline:14", "tries": 0, "error": ""}}
    assert fake.calls == [] and [e["event"] for e in store.events()].count("labless_recorded") == 1
    assert "root submission_id=0c01788397" in "\n".join(L.status_lines(ctx, store))


def test_gate_state_phrases(ctx, store):
    ctx = _ctx(ctx)
    assert L.gate_state(ctx, store) == "idle, 0 audit(s) failed"
    store.set_state(L.K_AUDIT, {"passed": [], "failed": [], "queue": ["n0001", "n0002"],
                                "current": {"node": "n0001"}})
    assert L.gate_state(ctx, store) == "auditing n0001 (+1 queued)"
    _node(store, "n0001")
    store.set_state(L.K_AUDIT, {"passed": ["n0001"], "failed": ["n0002"], "queue": [], "current": None})
    assert L.gate_state(ctx, store) == "n0001 passed audit, submission pending"
    store.set_state(L.K_LABLESS, {"n0001": {"submission_id": "sid"}})
    assert L.gate_state(ctx, store) == "idle, last n0001=sid, 1 audit(s) failed"
    assert L.gate_state(dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, labless=LablessCfg())), store) == "DISABLED"


def test_restart_mid_submission_records_the_landed_receipt_instead_of_resubmitting(ctx, store, tmp_path):
    import json
    import os
    ctx = _ctx(ctx, min_hours_between=4)
    _node(store, "n0001")
    out = tmp_path / "run"
    out.mkdir()
    store.update_run("n0001:audit:204", output_dir=str(out))
    now = [1000.0]
    store.set_state(L.K_LABLESS, {"n0001": {"tries": 1, "last_ts": 1000.0, "submission_id": None,
                                            "next_ts": 1000.0, "error": "", "run_id": "n0001:audit:204"}})
    receipt = out / "labless_submission.json"            # the killed call's submitter DID finish
    receipt.write_text(json.dumps({"submission_id": "sid-landed", "run": {"validation_errors": []}}))
    now[0] += 5 * H
    fake = Fake(["sid-dup"])
    assert "submission_id=sid-landed" in _sub(ctx, store, fake, now).step(["n0001"], None)[0]
    assert fake.calls == [] and store.get_state(L.K_LABLESS)["n0001"]["submission_id"] == "sid-landed"
    store.set_state(L.K_LABLESS, {"n0001": {"tries": 1, "last_ts": 1000.0, "submission_id": None,
                                            "next_ts": 1000.0, "error": ""}})
    os.utime(receipt, (10.0, 10.0))                      # a receipt from BEFORE the attempt proves nothing
    assert "submitting n0001" in _sub(ctx, store, fake, now).step(["n0001"], None)[0]


# ------------------------------------------------------------ descriptive title / changes / notes
def test_run_name_is_readable_short_and_unique():
    assert L.run_name("n0104", {"data.local_crop_scale": [0.16, 0.35], "dino.kde_concentration": 6.667}) \
        == "v2 crop.16+kde6.7"
    assert L.run_name("n0009", {"train.lr": 1.9e-4}) == "v2 lr1.9e-4 #0009"
    assert L.run_name("n0009", {"train.lr": 2e-4}) != L.run_name("n0010", {"train.lr": 2e-4})
    long = L.run_name("n0145", {"data.local_crop_scale": [0.16, 0.35], "dino.kde_concentration": 6.667,
                                "dino.kde_loss_weight": 0.06667})
    assert long == "v2 crop.16+kde6.7+1" and len(long) <= 20
    assert L.run_name("n0007", None, "lower_lr") == "v2 lower_lr #0007"
    assert L.run_name("n0007") == "loop-v2-n0007"
    for tag in ("x" * 50, "a b/c"):
        assert len(L.run_name("n123456789", {"a.b": 1.0}, tag)) <= 20 and len(L.run_name("n1", None, tag)) <= 20


PY0 = "def build_loss(cfg):\n" + "    pass\n" * 8 + "    return 1\n"
BASE_CFG = "data:\n  local_crop_scale: [0.08, 0.35]\ndino:\n  kde_loss_weight: 0.1\ntrain:\n  lr: 0.0002\n"


def _git_world(ctx, store, tmp_path, versions):
    """A real repo: root commit + one commit per (name, yaml, train.py text). Returns ctx, {name: sha}."""
    import subprocess
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],  # noqa: E731
                                    check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    shas = {}
    for name, yml, py in [("root", BASE_CFG, PY0), *versions]:
        (repo / "configs" / "main.yaml").write_text(yml)
        (repo / "train.py").write_text(py)
        git("add", "-A")
        git("commit", "-q", "-m", name, "--allow-empty")
        shas[name] = git("rev-parse", "HEAD")
    store.ensure_root(shas["root"])
    ctx = dataclasses.replace(ctx, spec=SimpleNamespace(repo=str(repo), surface=SimpleNamespace(writable=["configs/main.yaml"])))
    return ctx, shas


def _described(ctx, store, tmp_path, versions, name="child", **node_kw):
    from loop.contracts import HypothesisKind, HypothesisRecord
    ctx, shas = _git_world(ctx, store, tmp_path, versions)
    _node(store, "n0007")
    store.put_hypothesis(HypothesisRecord(id="h1", task_id=store.task_id, kind=HypothesisKind.TRIAL,
                                          claim="milder crops help", mechanism_tag="lower_lr",
                                          family="optimization", predictions={}, falsification="x"))
    node = dataclasses.replace(store.get_node("n0007"), hypothesis_id="h1", commit=shas[name], **node_kw)
    return ctx, node


def test_changes_list_config_keys_old_to_new_and_code_functions(ctx, store, tmp_path):
    yml = BASE_CFG.replace("[0.08, 0.35]", "[0.16, 0.35]")
    ctx, node = _described(ctx, store, tmp_path, [("child", yml, PY0.replace("return 1", "return 2\n    # z"))])
    text = L.changes_text(ctx, store, node)
    assert text.splitlines()[0].endswith("| explore from root") and text.startswith("vs baseline ")
    assert "data.local_crop_scale: [0.08, 0.35] -> [0.16, 0.35]" in text
    assert "train.lr" not in text                                   # unchanged vs baseline: not listed
    assert "code: train.py :: build_loss (+2/-1 lines)" in text and "main.yaml" not in text
    assert L.run_name("n0007", {k: n for k, (_, n) in L.actual_changes(ctx, store, node).items()}) == "v2 crop.16 #0007"


def test_crossover_omits_a_parent_key_the_child_did_not_take(ctx, store, tmp_path):
    parent_b = BASE_CFG.replace("0.1\n", "0.01\n")                  # n0031: kde_loss_weight 0.01
    child = BASE_CFG.replace("[0.08, 0.35]", "[0.16, 0.35]")         # the crossover kept only the crop
    ctx, node = _described(ctx, store, tmp_path, [("pb", parent_b, PY0),
                                                  ("child", child, PY0)],
                           strategy=Strategy.CROSSOVER, second_parent_id="n0031",
                           notes="request=r brief=x xover=n0015,n0031")
    store.set_state("config_diff:n0031", {"dino.kde_loss_weight": 0.01})     # ancestor state is never consulted
    store.set_state("config_diff:n0007", {"dino.kde_loss_weight": 0.01, "data.local_crop_scale": [0.16, 0.35]})
    text = L.note(store, None, node, ctx)
    assert "kde_loss_weight" not in text
    assert "data.local_crop_scale: [0.08, 0.35] -> [0.16, 0.35]" in text
    assert "crossover of n0015 x n0031" in text
    assert "data.local_crop_scale=[0.16, 0.35]" in text.split("Reproduce:")[1]


def test_crossover_lineage_falls_back_to_the_parent_ids():
    n = SimpleNamespace(mechanism_tag="t", notes="", strategy=Strategy.CROSSOVER, parent_id="n1", second_parent_id="n2")
    assert L.lineage(n) == "crossover of n1 x n2"


def test_notes_carry_the_computed_evidence(ctx, store, tmp_path):
    ctx, node = _described(ctx, store, tmp_path, [("child", BASE_CFG.replace("0.0002", "0.0003"),
                                                   PY0)],
                           confirm_e=12.5, confirm_alt=0.0042)
    axes = list(ctx.cfg.axes)
    for seed in (201, 202, 203, 204):                              # paired audit runs: node +0.01 over root
        for nid, off in ((node.id, 0.01), (ROOT_NODE_ID, 0.0)):
            rid = Run.make_id(nid, Stage.AUDIT, seed)
            if nid == ROOT_NODE_ID:
                store.put_run(Run(id=rid, task_id=store.task_id, node_id=nid, seed=seed, stage=Stage.AUDIT,
                                  status=RunStatus.DONE, aggregate=0.5, scores={a: 0.5 for a in axes}))
            else:
                store.update_run(rid, scores={a: 0.5 + off + 0.001 * (seed - 201) for a in axes})
    text = L.note(store, None, node, ctx)
    assert "Hypothesis: milder crops help" in text
    assert "train.lr: 0.0002 -> 0.0003" in text and "e-value 12.5" in text and "+0.0042" in text
    assert "held-out audit (live, no verdict yet): panel delta vs baseline (total, incl. inherited) +0.0115 +- 0.0006 se over 4 audit seeds" in text
    # once the gate has decided, the note prints the verdict's frozen numbers, never a re-fit
    store.set_state(L.K_AUDIT, {"passed": [node.id], "gate": {node.id: {
        "verdict": "passed", "mean": 0.0054, "se": 0.0013, "n": 4, "e": 28.6, "alpha": 0.05,
        "delta": 0.00275, "sigma": 0.00363, "source": "gate"}}})
    text = L.note(store, None, node, ctx)
    assert ("held-out audit (gate): panel delta vs baseline +0.0054 +- 0.0013 se over 4 audit seeds, E=28.6 "
            "vs bar 20 (alpha 0.05) -> passed; frozen claim +0.00275, sigma 0.00363") in text
    assert "+0.0115" not in text and "after the verdict" not in text
    assert "node n0007" in text and "train.lr=0.0003" in text


def test_notes_never_crash_on_missing_fields(ctx, store):
    store.ensure_root("c0")
    bare = Node(id="n9", task_id=store.task_id, commit="", parent_id=None, strategy=Strategy.EXPLORE,
                operator=None, family="f", mechanism_tag="", diff_hash="", status=NodeStatus.CONFIRMED,
                diff_path="/nonexistent.diff")
    text = L.note(store, None, bare, ctx)
    assert "Hypothesis: ?" in text and "OWN effect: ?" in text and "total vs root: ?" in text
    assert L.note(store, None, bare) and L.changes_text(ctx, store, bare)      # no ctx / no config: still text
    assert L.run_name("n9", {}, "") == "loop-v2-n9"


def test_submitter_passes_changes_alongside_notes(ctx, store, tmp_path):
    ctx = _ctx(ctx)
    ctx, shas = _git_world(ctx, store, tmp_path, [("c", BASE_CFG.replace("0.0002", "0.0003"), PY0)])
    _node(store, "n0001")
    store.update_node("n0001", commit=shas["c"]) if hasattr(store, "update_node") else store.put_node(
        dataclasses.replace(store.get_node("n0001"), commit=shas["c"]))
    fake = Fake(["sid-1"])
    sub = _sub(ctx, store, fake, [10 * H])
    sub.step(["n0001"], None)
    assert fake.calls[0][1] == "[aud] lr3e-4 #0001" and "Reproduce:" in fake.calls[0][2]
    assert "train.lr" in fake.changes and fake.changes.startswith("on the baseline: train.lr")


def _audit_world(ctx, store, t, nid, offs, fin_ts, root=True):
    """Paired audit runs of ``nid`` (panel +off over root per seed), finished at ``fin_ts``."""
    import time as _t
    axes = list(ctx.cfg.axes)
    store.ensure_root("c0")
    if store.get_node(nid) is None:
        _node(store, nid)
    for seed, off in zip((201, 202, 203, 204), offs):
        for owner, d in ((nid, off), *(((ROOT_NODE_ID, 0.0),) if root else ())):
            rid = Run.make_id(owner, Stage.AUDIT, seed)
            run = Run(id=rid, task_id=store.task_id, node_id=owner, seed=seed, stage=Stage.AUDIT,
                      status=RunStatus.DONE, aggregate=0.5, scores={a: 0.5 + d for a in axes})
            store.put_run(run) if store.get_run(rid) is None else store.update_run(rid, scores=run.scores)
            t[0] = fin_ts
            store.log("run_finished", run_id=rid, status="done")


def test_backfill_freezes_pre_existing_verdicts_exactly_when_no_sibling_had_finished(ctx, tmp_path):
    """n0165 / n0255 on 2026-09-30: their verdicts predate the stored gate numbers. The first audit
    had no sibling (offsets 0): exact. The second's offsets used estimates never stored: 'recomputed'."""
    import math
    from loop.seeds import audit_step
    from loop.store import SqliteStore
    t = [100.0]
    store = SqliteStore(tmp_path / "bf.db", ctx.task_id, weights=ctx.cfg.weights, clock=lambda: t[0])
    offs_a, offs_b = (0.006, 0.007, 0.002, 0.006), (0.009, 0.011, 0.015, 0.008)
    _audit_world(ctx, store, t, "n0165", offs_a, 200.0)
    runs = {k: store.runs(node_id=k) for k in ("n0165", ROOT_NODE_ID)}
    verdict, e, _ = audit_step("n0165", runs["n0165"], runs[ROOT_NODE_ID], cfg=ctx.cfg, delta=0.00275,
                               sigma=0.0036, infra_retry=1)
    t[0] = 300.0
    store.log("audit", node_id="n0165", verdict=verdict, e_value=e)
    _audit_world(ctx, store, t, "n0255", offs_b, 400.0, root=False)
    t[0] = 500.0
    store.log("audit", node_id="n0255", verdict="passed", e_value=8.0e4)
    _audit_world(ctx, store, t, "n0300", offs_b, 600.0, root=False)      # finished after both verdicts
    store.set_state(L.K_AUDIT, {"passed": ["n0165", "n0255"], "failed": [], "queue": [], "current": None})
    log = tmp_path / "loop.log"
    log.write_text("x audit started for n0165 (claimed gain +0.00275)\n"
                   "y audit started for n0255 (claimed gain +0.00457)\n")
    out = L.backfill_audit_gate(ctx, store, None, log_path=log)
    g = store.get_state(L.K_AUDIT)["gate"]
    a = g["n0165"]
    assert a["source"] == "reconstructed" and a["n"] == 4 and a["offsets"] == {} and a["e"] == e
    assert a["mean"] == pytest.approx(sum(offs_a) / 4) and a["delta"] == 0.00275
    assert a["sigma"] == pytest.approx(0.0036, rel=1e-9)                # solved back from E exactly
    assert g["n0255"]["source"] == "recomputed" and g["n0255"]["sigma"] is None and g["n0255"]["e"] == 8.0e4
    assert "n0300" not in g and len(out) == 2 and "(reconstructed)" in out[0]
    assert "n0165" in g["n0255"]["why"] and "n0300" not in g["n0255"]["why"]
    text = L.note(store, None, store.get_node("n0165"), ctx)
    assert "held-out audit (gate): panel delta vs baseline +0.005" in text and "[reconstructed after the verdict" in text
    assert L.backfill_audit_gate(ctx, store, None, log_path=log) == []   # once


# ------------------------------------------------------------ tiers (operator 2026-09-30: submit a lot)
def _lifecycle(store, nid, status, n=5, claim="", tag="attmask"):
    """A node with ``n`` scored lifecycle runs (screen 11, 12 / promote 13 / confirm 14, 101) and no audit run;
    its mechanism (tag + claim) is its own unless ``claim`` is shared."""
    from loop.contracts import HypothesisKind, HypothesisRecord
    store.ensure_root("c0")
    store.put_hypothesis(HypothesisRecord(id=f"h-{nid}", task_id=store.task_id, kind=HypothesisKind.TRIAL,
                                          claim=claim or f"mechanism {nid}", mechanism_tag="attmask",
                                          family="objective", predictions={}, falsification="f"))
    store.put_node(Node(id=nid, task_id=store.task_id, commit="abc123", parent_id=ROOT_NODE_ID,
                        strategy=Strategy.MOONSHOT, operator=None, family="objective",
                        mechanism_tag=tag, diff_hash=nid, status=status, hypothesis_id=f"h-{nid}"))
    plan = [(Stage.SCREEN, 11), (Stage.SCREEN, 12), (Stage.PROMOTE, 13), (Stage.CONFIRM, 14), (Stage.CONFIRM, 101)]
    for i, (st, seed) in enumerate(plan[:n]):
        store.put_run(Run(id=Run.make_id(nid, st, seed), task_id=store.task_id, node_id=nid, seed=seed, stage=st,
                          status=RunStatus.DONE, aggregate=0.6 + 0.01 * i, scores={"alpha": 0.6},
                          output_dir=f"/runs/{nid}/{seed}"))


def _post(nid, p_inc, vetoed=False, u=0.006):
    from loop.contracts import NodePosterior, Posterior
    q = Posterior(u, 0.002, 5)
    return NodePosterior(nid, q, q, {}, 0.999, p_inc, vetoed, ("gamma",) if vetoed else ())


def _tier_world(store):
    """Operator 2026-10-05: one representative per CLOSED tested group, its best member even when poor."""
    _lifecycle(store, "n0305", NodeStatus.CONFIRMED)              # confirmed, audit pending: first
    _lifecycle(store, "n0315", NodeStatus.CONFIRMING)             # its attmask cluster is still open
    _lifecycle(store, "n0308", NodeStatus.NOT_CONFIRMED, tag="lad")          # ladder t3: base + 2 dose points
    for nid in ("n0200", "n0201"):
        _lifecycle(store, nid, NodeStatus.SCREENED_OUT, tag=f"t3{nid}")
    for nid in ("n0400", "n0401"):                                             # design d2: both points lost
        _lifecycle(store, nid, NodeStatus.SCREENED_OUT, tag=f"d2{nid}")
    for nid in ("n0500", "n0501"):                                             # a closed mechanism cluster
        _lifecycle(store, nid, NodeStatus.SCREENED_OUT, tag="kde_anneal")
    _lifecycle(store, "n0600", NodeStatus.SCREENED_OUT, tag="open_mech")       # an open one: not yet
    _lifecycle(store, "n0601", NodeStatus.PROMOTING, tag="open_mech")
    for nid in ("n0700", "n0701"):                                             # ladder t1 closed long ago
        _lifecycle(store, nid, NodeStatus.SCREENED_OUT, tag=f"old{nid}")
    store.set_state("tune", {"batches": [
        {"batch": 3, "kind": "ladder", "base": "n0308", "points": {"0": "n0308"}, "trials": ["n0200", "n0201"],
         "closed": 9 * H},
        {"batch": 1, "kind": "ladder", "base": "n0700", "points": {"0": "n0700"}, "trials": ["n0701"], "closed": -100 * H}]})
    store.set_state("design", {"batches": [{"batch": 2, "points": {"0": ROOT_NODE_ID, "1": "n0400", "2": "n0401"}}]})
    u = {"n0305": 0.006, "n0315": 0.006, "n0308": 0.002, "n0200": 0.004, "n0201": 0.001, "n0400": -0.001,
         "n0401": -0.003, "n0500": 0.003, "n0501": 0.001, "n0600": 0.009, "n0601": 0.009, "n0700": 0.0005,
         "n0701": 0.0002}
    return SimpleNamespace(nodes={k: _post(k, 0.5, u=v) for k, v in u.items()})


def test_exploratory_queue_is_confirmed_then_one_best_member_per_closed_group(ctx, store):
    snap = _tier_world(store)
    q = L.exploratory_queue(_ctx(ctx), store, snap, now=10 * H)
    assert [(n, g) for n, _, g in q] == [("n0305", ""), ("n0200", "t3"), ("n0500", "mech:kde_anneal"), ("n0400", "d2")]
    assert "confirmed on fresh seeds" in q[0][1]
    assert "best of closed dose ladder t3 (3 tested), ranked 1/3 by gain vs root: +0.0040" in q[1][1]
    assert "n0200 +0.0040, n0308 +0.0020, n0201 +0.0010" in q[1][1] and "POORLY" not in q[1][1]
    assert "design batch d2 (2 tested)" in q[3][1] and "did POORLY" in q[3][1]
    assert L.exploratory_queue(_ctx(ctx, exploratory=False), store, snap) == []
    assert [n for n, _, _ in L.exploratory_queue(_ctx(ctx), store, None)] == ["n0305"]   # no snapshot: confirmed only
    assert "n0700" in [n for n, _, _ in L.exploratory_queue(_ctx(ctx, group_max_age_h=0), store, snap, now=10 * H)]


def test_a_group_with_any_member_already_submitted_sends_nothing_more_and_stale_failures_clear(ctx, store):
    """Operator 10-05: ONE representative per closed group, enforced against every past submission (the old rule sent
    not-confirmed nodes one by one); a failed entry no tier would send any more leaves the book and `loop status`."""
    snap = _tier_world(store)
    c = _ctx(ctx)
    store.set_state(L.K_LABLESS, {"n0201/exp": {"submission_id": "old", "tier": "exploratory"},     # t3 member, old era
                                  "n0700/exp": {"submission_id": None, "tries": 2, "error": "exit 1"},   # stale
                                  "n0400/exp": {"submission_id": None, "tries": 1, "error": "exit 1", "group": "d2"}})
    q = [(n, g) for n, _, g in L.exploratory_queue(c, store, snap, now=10 * H)]
    assert ("n0200", "t3") not in q and ("n0400", "d2") in q and ("n0500", "mech:kde_anneal") in q
    now = [10 * H]
    sub = _sub(c, store, Fake(["sid-1"]), now)
    out = sub.step([], snap)
    assert any("cleared 1 stale failed entry (n0700/exp)" in a for a in out), out
    book = store.get_state(L.K_LABLESS)
    assert "n0700/exp" not in book and "n0400/exp" in book and book["n0201/exp"]["submission_id"] == "old"
    assert store.events("labless_stale_cleared")[0]["keys"] == ["n0700/exp"]


def test_exploratory_then_audited_tier_each_once_with_tier_titles(ctx, store):
    snap = _tier_world(store)
    for seed, agg in ((201, 0.60), (202, 0.64), (203, 0.62), (204, 0.61)):     # n0305's audit runs
        store.put_run(Run(id=Run.make_id("n0305", Stage.AUDIT, seed), task_id=store.task_id, node_id="n0305",
                          seed=seed, stage=Stage.AUDIT, status=RunStatus.DONE, aggregate=agg,
                          output_dir=f"/runs/n0305/{seed}"))
    now, fake = [10 * H], Fake(["sid-a", "sid-b", "sid-c", "sid-d", "sid-e"])
    sub = _sub(_ctx(ctx, min_hours_between=0), store, fake, now)
    for _ in range(12):
        sub.step([], snap)
    assert [c[1] for c in fake.calls] == ["[exp] attmask #0305", "[exp] t3n0200 #0200", "[exp] kde_anneal",
                                          "[exp] d2n0400 #0400"]
    assert fake.calls[0][0] == "/runs/n0305/13" and fake.calls[0][2].startswith("Tier: exploratory")
    assert fake.calls[1][2].startswith("Tier: exploratory: best of closed dose ladder t3 (3 tested), ranked 1/3")
    book = store.get_state(L.K_LABLESS)
    assert book["n0200/exp"]["group"] == "t3" and "group" not in book["n0305/exp"]
    snap.nodes["n0201"] = _post("n0201", 0.5, u=0.02)                      # the ranking shifts: the group is done
    for _ in range(4):
        sub.step([], snap)
    assert len(fake.calls) == 4
    for _ in range(4):
        sub.step(["n0305"], snap)
    assert fake.calls[4][:2] == ("/runs/n0305/204", "[aud] attmask #0305")     # its own, audited tier
    assert fake.calls[4][2].startswith("Tier: audited")
    for _ in range(4):
        sub.step(["n0305"], snap)
    assert len(fake.calls) == 5                                                     # once per tier


def test_audited_submission_skips_the_exploratory_tier_and_the_daily_cap_holds(ctx, store):
    snap = _tier_world(store)
    store.set_state(L.K_LABLESS, {"n0305": {"submission_id": "sid-old"}})           # audited already
    now, fake = [10 * H], Fake(["sid-1", "sid-2", "sid-3"])
    sub = _sub(_ctx(ctx, min_hours_between=0, max_per_day=1, group_max_age_h=0), store, fake, now)
    for _ in range(6):
        sub.step([], snap)
    assert [c[0] for c in fake.calls] == ["/runs/n0200/13"]                        # cap 1 per 24 h
    now[0] += 24 * H
    for _ in range(3):
        sub.step([], snap)
    assert [c[0] for c in fake.calls] == ["/runs/n0200/13", "/runs/n0500/13"]


def test_exploratory_near_duplicates_go_once_per_cluster_unless_they_gain_the_bar(ctx, store, tmp_path):
    """Operator 2026-09-30: no 'lr1.9e-4 + 4', 'lr1.9e-4 + 6' spam -- one representative per near-dup cluster."""
    lr = lambda v: BASE_CFG.replace("lr: 0.0002", f"lr: {v}")                       # noqa: E731
    ctx, shas = _git_world(ctx, store, tmp_path, [("a", lr(0.00019), PY0), ("b", lr(0.00017), PY0),
                                                  ("c", lr(0.00017), PY0.replace("return 1", "return 2")),
                                                  ("d", lr(0.00008), PY0)])
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, labless=LablessCfg(enabled=True)))
    for nid in "abcd":
        _lifecycle(store, nid, NodeStatus.CONFIRMED)
        store.update_node(nid, commit=shas[nid])
    snap = SimpleNamespace(nodes={n: _post(n, 0.9) for n in "abcd"})
    a, b, c, d = (store.get_node(n) for n in "abcd")
    assert L.near_dup(ctx, a, b) and not L.near_dup(ctx, b, c) and not L.near_dup(ctx, a, d)  # code; 2.4x
    assert L.duplicate(ctx, store, snap, b, "exploratory", ahead=["a"]).startswith("near-duplicate of a")
    assert L.duplicate(ctx, store, snap, b, "audited", ahead=["a"]) == ""        # the audit is its own bar
    assert L.duplicate(ctx, store, snap, c, "exploratory", ahead=["a"]) == ""    # a distinct mechanism
    store.set_state(L.K_LABLESS, {"a/exp": {"submission_id": "sid-a"}})
    assert "near-duplicate of a" in L.duplicate(ctx, store, snap, b, "exploratory")
    from loop.contracts import NodePosterior, Posterior
    big = NodePosterior("b", Posterior(0.02, 0.002, 5), Posterior(0.02, 0.002, 5), {}, 0.999, 0.9, False)
    assert L.duplicate(ctx, store, SimpleNamespace(nodes={**snap.nodes, "b": big}), b, "exploratory") == ""


def test_title_and_description_are_the_nodes_own_change_on_its_carrier(ctx, store, tmp_path):
    """22/30 exploratory titles read 'lr1.9e-4+N' (the incumbent's knobs, diffed vs root): the title is now the
    node's own change vs its parent (a crossover: vs its first parent), and the notes say what it sits on."""
    inc = BASE_CFG.replace("0.0002", "0.00019")
    child = inc.replace("[0.08, 0.35]", "[0.16, 0.35]")
    ctx, shas = _git_world(ctx, store, tmp_path, [("inc", inc, PY0), ("child", child, PY0),
                                                  ("code", inc, PY0.replace("return 1", "return 2"))])
    for nid, sha, parent in (("n0255", shas["inc"], ROOT_NODE_ID), ("n0300", shas["child"], "n0255"),
                             ("n0301", shas["code"], "n0255")):
        store.put_node(Node(id=nid, task_id=store.task_id, commit=sha, parent_id=parent, strategy=Strategy.EXPLORE,
                            operator=None, family="augmentation", mechanism_tag=f"tag{nid}", diff_hash=nid,
                            status=NodeStatus.CONFIRMED))
    node = store.get_node("n0300")
    c, own, code = L.own_changes(ctx, store, node)
    assert c.id == "n0255" and own == {"data.local_crop_scale": ([0.08, 0.35], [0.16, 0.35])} and not code
    assert L.own_line(ctx, store, node) == "on n0255 (tagn0255): data.local_crop_scale: [0.08, 0.35] -> [0.16, 0.35]"
    text = L.note(store, None, node, ctx, tier="exploratory")
    assert "Change: on n0255 (tagn0255)" in text and "train.lr: 0.0002 -> 0.00019" in text   # the stack stays listed
    root_child = store.get_node("n0255")
    assert L.own_line(ctx, store, root_child).startswith("on the baseline: train.lr")
    assert L.own_line(ctx, store, store.get_node("n0301")).startswith("on n0255 (tagn0255): code: train.py")
    xo = dataclasses.replace(node, parent_id=ROOT_NODE_ID, notes="xover=n0255,n0299")
    assert L.carrier(store, xo).id == "n0255"


def test_notes_report_the_nodes_own_effect_vs_its_measure_parent_not_the_total_vs_root(ctx, store):
    """n0405's notes quoted P(U>0)=1.000 / +0.0077 / per-axis vs root, ~6x its own effect: the evidence is now its
    own increment +- sd over its measure parent and own per-axis deltas; one labelled 'total vs root' line stays."""
    from loop.contracts import NodePosterior, Posterior
    _lifecycle(store, "n0405", NodeStatus.CONFIRMED)
    node = dataclasses.replace(store.get_node("n0405"), parent_id="n0255")
    tot, par = Posterior(0.0077, 0.002, 5), Posterior(0.0064, 0.0015, 9)
    snap = SimpleNamespace(measured={"n0405": ("n0398", 0.0, 0.0)}, nodes={
        "n0405": NodePosterior("n0405", Posterior(0.0013, 0.0011, 5), tot, {"alpha": Posterior(0.010, 0.003, 5)},
                               1.0, 0.6, False),
        "n0398": NodePosterior("n0398", par, par, {"alpha": Posterior(0.008, 0.004, 9)}, 1.0, 0.5, False)})
    text = L.note(store, snap, node, ctx)
    assert "OWN effect: panel increment vs n0398 +0.0013 +- 0.0011" in text
    assert "Per-axis OWN deltas vs n0398 (shrunk): alpha +0.0020+-0.0050" in text
    assert "total vs root (incl. the inherited stack): +0.0077" in text and "screen: P(own gain>0)=0.88" in text
    assert "Per-axis shrunk deltas" not in text


def test_exploratory_tier_sends_one_representative_per_mechanism_cluster(ctx, store):
    """n0308 / n0405 / n0408 all went out as attention-saliency masking: the cluster's best P(own gain>0) goes
    first; a later member only if its own increment beats the representative's by the noise rule."""
    from loop.contracts import NodePosterior, Posterior
    sal = "mask the most salient patches by teacher CLS attention saliency"
    for nid in ("n0308", "n0405", "n0408"):
        _lifecycle(store, nid, NodeStatus.NOT_CONFIRMED, claim=sal + f" variant {nid}")
    for i in range(6):
        _lifecycle(store, f"n02{i:02d}", NodeStatus.SCREENED_OUT, claim=f"unrelated change number {i} on tiles")
    post = lambda nid, m, sd=0.001: NodePosterior(nid, Posterior(m, sd, 5), Posterior(0.0077, 0.002, 5), {},  # noqa: E731
                                                  0.999, 0.7, False)
    snap = SimpleNamespace(nodes={"n0308": post("n0308", 0.0010), "n0405": post("n0405", 0.0030),
                                  "n0408": post("n0408", 0.0012)})
    c, q = _ctx(ctx), ["n0308", "n0405", "n0408"]
    node = lambda i: store.get_node(i)                                                     # noqa: E731
    assert L.duplicate(c, store, snap, node("n0308"), "exploratory", queued=q).startswith("same mechanism as n0405")
    assert L.duplicate(c, store, snap, node("n0405"), "exploratory", queued=q) == ""    # best P(gain): the rep
    assert L.duplicate(c, store, snap, node("n0408"), "exploratory", ahead=["n0405"], queued=q) != ""
    snap.nodes["n0408"] = post("n0408", 0.0150)                                          # clears the bar: goes
    assert L.duplicate(c, store, snap, node("n0408"), "exploratory", ahead=["n0405"], queued=q) == ""
    assert L.duplicate(c, store, snap, node("n0308"), "audited", queued=q) == ""         # the audit is its own bar


def test_resume_copy_run_submits_the_repo_config_with_its_effective_seed(tmp_path, monkeypatch):
    """A resumable run launches from ~/state/resume/<node>/<run>.yaml (absolute, outside configs/): the
    submitter is told review_config=configs/main.yaml, and that snapshot carries the effective seed and
    output_dir with the copy's own resume knobs put back to the repo's."""
    import json
    import yaml
    monkeypatch.delenv("LOOP_FORBID_SLURM")
    (tmp_path / "labless").mkdir()
    (tmp_path / "labless" / "submit_to_labless.py").write_text(
        "import sys, pathlib, json\nargs = dict(a.split('=', 1) for a in sys.argv[1:])\n"
        "pathlib.Path(args['output_dir'], 'labless_submission.json').write_text("
        "json.dumps({'submission_id': args['review_config'], 'run': {}}))\n")
    out = tmp_path / "run"
    (out / "labless_source" / "configs").mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps({"config_path": "/home/u/state/resume/n1/screen_s412_a0.yaml"}))
    repo_cfg = {"project": {"output_dir": "/default"}, "train": {"seed": 7, "save_every": 1000, "resume": None}}
    (out / "labless_source" / "configs" / "main.yaml").write_text(yaml.safe_dump(repo_cfg))
    (out / "labless_source" / "configs" / "screen_s412_a0.yaml").write_text(yaml.safe_dump(
        {"project": {"output_dir": str(out)}, "train": {"seed": 412, "save_every": 500, "resume": "/x/latest.pt"}}))
    ctx = SimpleNamespace(cfg=SimpleNamespace(labless=LablessCfg(python=sys.executable)),
                          spec=SimpleNamespace(repo=str(tmp_path), surface=SimpleNamespace(writable=["configs/main.yaml"])))
    assert L.submit_run(ctx, str(out), "loop-v2-n1", "note") == "configs/main.yaml"
    snap = yaml.safe_load((out / "labless_source" / "configs" / "main.yaml").read_text())
    assert snap == {"project": {"output_dir": str(out)}, "train": {"seed": 412, "save_every": 1000, "resume": None}}
    assert L.review_config(ctx, str(out)) == "configs/main.yaml"         # idempotent on a retry
    assert yaml.safe_load((out / "labless_source" / "configs" / "main.yaml").read_text()) == snap


def test_startup_clears_the_backoff_of_config_path_refusals_only(ctx, store):
    err = "RuntimeError('exit 1; ValueError: summary.config_path must be a repo-relative configs/*.yaml path')"
    store.set_state(L.K_LABLESS, {"a/exp": {"tries": 2, "next_ts": 9e9, "submission_id": None, "error": err},
                                  "b/exp": {"tries": 1, "next_ts": 9e9, "submission_id": None, "error": "exit 1"}})
    _sub(ctx, store, Fake([]), [0.0])
    book = store.get_state(L.K_LABLESS)
    assert book["a/exp"]["next_ts"] == 0.0 and book["b/exp"]["next_ts"] == 9e9
