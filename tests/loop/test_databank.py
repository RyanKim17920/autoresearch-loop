"""THE DATA BANK (loop/databank.py): spec validation / naming, the STATUS registry (pilots / building / failed
ignored), requests (dedupe, name clash, disk cap, one build at a time), holding / voiding nodes on a dataset not
ready, the per-dataset computed adoption into a campaign (index knobs of list values), the smoke stand-in, the prune
list and the digest section.
Run: uv run pytest tests/loop -q --confcutdir=tests/loop"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from loop import campaign as CP
from loop import databank as DB
from loop import digest as DJ
from loop.contracts import ROOT_NODE_ID, Node, NodeStatus, Operator, Strategy
from loop.spec import DatabankCfg, FamilyRule

SPEC = {"tile_px": 448, "target_mpp": 0.5, "n_tiles": "all", "seed": 0, "sampling": {"min_tissue": 0.25}}


def _ctx(ctx, root, **kw):
    fams = dict(ctx.cfg.families, data_curation=FamilyRule(keys=("data.*", "train.num_workers")))
    db = DatabankCfg(enabled=True, root=str(root), **kw)
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, databank=db, families=fams,
                                                            data_bank=("/data/cur/arrow: curated",)))


def _ds(root: Path, name: str, status: str, *, px=448, n=4_000_000, ready="2026-10-07T17:37:01", spec=None):
    d = root / name
    d.mkdir(parents=True)
    (d / "STATUS").write_text(status + "\n")
    sp = {"name": name, "tile_px": px, "target_mpp": "native_mix", "n_tiles": "all", "seed": 0, **(spec or {})}
    (d / "spec.json").write_text(json.dumps(sp))
    (d / "status.json").write_text(json.dumps({"state": status, **({"ready": ready} if status == "ready" else {})}))
    if status == "ready":
        (d / "manifest.json").write_text(json.dumps({
            "n_tiles": n, "n_slides": 11368, "bytes": 452e9, "spec": sp,
            "output_mpp_percentiles": {"25": 0.5, "50": 2.0, "75": 4.04}}))
    return d


def test_validate_names_by_content_hash_and_refuses_bad_specs():
    got, why = DB.validate(SPEC)
    assert not why and got["name"] == f"ds448_{DB.spec_hash(SPEC)[:8]}"
    assert DB.spec_hash({**SPEC, "name": "x", "slurm": {"n_tasks": 3}}) == DB.spec_hash(SPEC)   # content only
    assert DB.spec_hash({**SPEC, "sampling": {"min_tissue": 0.25, "source": "original_arrow"}}) == DB.spec_hash(SPEC)
    assert DB.validate({**SPEC, "name": "mpp_pilot"})[1].startswith("name")
    assert "GenBio" in DB.validate({**SPEC, "sampling": {"score_filter": {"path": "/x/genbio.parquet"}}})[1]
    assert "unknown" in DB.validate({**SPEC, "slurm": {}})[1]
    assert "target_mpp" in DB.validate({**SPEC, "target_mpp": [{"mpp": 0.5, "frac": 0.6}, {"mpp": 1, "frac": 0.6}]})[1]
    assert "n_tiles" in DB.validate({**SPEC, "target_mpp": [{"mpp": 0.5, "frac": 1.0}]})[1]       # 'all': one group
    assert "score_filter.path" in DB.validate({**SPEC, "sampling": {"score_filter": {"path": "/nope.parquet"}}})[1]
    assert "tile_px" in DB.validate({**SPEC, "tile_px": 4096})[1]


def test_registry_registers_ready_non_pilots_only_and_feeds_the_data_bank(ctx, store, tmp_path):
    root = tmp_path / "bank"
    c = _ctx(ctx, root, note="use more workers")
    for name, st in (("mixed_448", "ready"), ("mixed_448_pilot", "ready"), ("mpp050_448", "extracting"),
                     ("bad", "failed")):
        _ds(root, name, st)
    (root / "_meta").mkdir()
    acts = DB.tick(c, store, None, None, new_id=None, now=0.0)
    assert len(acts) == 1 and acts[0].startswith("databank: registered mixed_448 ")
    assert DB.tick(c, store, None, None, new_id=None, now=0.0) == []                  # once
    lines = DB.bank_lines(c, store)
    assert lines[0] == "/data/cur/arrow: curated"
    assert lines[1].startswith(f"{root}/mixed_448: 4,000,000 tiles of 448 px") and "11,368 slides" in lines[1]
    assert lines[1].endswith("use more workers") and len(lines) == 2
    assert DB.bank_lines(ctx, store) == list(ctx.cfg.data_bank)                       # disabled: the spec's only
    assert any("mpp050_448 extracting" in x for x in DB.plan(c, store, None))


def test_requests_dedupe_refuse_and_build_one_at_a_time(ctx, store, tmp_path, monkeypatch):
    root = tmp_path / "bank"
    c = _ctx(ctx, root, max_total_tb=1.3)
    _ds(root, "mixed_448", "ready")
    DB.tick(c, store, None, None, new_id=None, now=0.0)
    launched = []
    monkeypatch.setattr(DB, "launch", lambda ctx, spec, log: launched.append(json.loads(spec.read_text())) or 99999999)
    a = DB.request(c, store, "n0001", {**SPEC, "name": "mpp050"})
    assert "queued" in a and DB.book(store)["requests"]["mpp050"]["state"] == "waiting"
    assert "deduped" in DB.request(c, store, "n0002", {**SPEC, "name": "other"})        # same content
    assert DB.book(store)["requests"]["mpp050"]["by"] == ["n0001", "n0002"]
    assert "deduped" in DB.request(c, store, "n0003", {"name": "zz", "tile_px": 448, "target_mpp": "native_mix",
                                                         "n_tiles": "all", "seed": 0})   # = the registered one
    assert "REFUSED" in DB.request(c, store, "n0004", {**SPEC, "name": "mpp050", "seed": 1})   # name clash
    big = DB.request(c, store, "n0005", {**SPEC, "seed": 2, "tile_px": 896})
    assert "REFUSED" in big and "max_total_tb" in big
    assert [e["outcome"] for e in store.events("databank_request")] == ["queued", "deduped", "deduped", "refused",
                                                                       "refused"]
    _ds(root, "x_pilot", "extracting", px=224, n=20000)                                # a build in flight
    acts = DB.tick(c, store, None, None, new_id=None, now=0.0)
    assert not launched and not acts                                                   # max_building 1
    (root / "x_pilot" / "STATUS").write_text("ready\n")
    acts = DB.tick(c, store, None, None, new_id=None, now=0.0)
    assert launched[0]["name"] == "mpp050" and "build mpp050 launched" in acts[0]
    r = DB.book(store)["requests"]["mpp050"]
    assert r["state"] == "submitted" and Path(c.state_dir, "databank", "mpp050.json").is_file()
    acts = DB.tick(c, store, None, None, new_id=None, now=0.0)                         # build.py exited, no STATUS
    assert DB.book(store)["requests"]["mpp050"]["state"] == "failed" and "FAILED" in acts[0]


def _node(store, nid, cfg, status=NodeStatus.SCREENING):
    store.put_node(Node(id=nid, task_id=store.task_id, commit="c", parent_id=ROOT_NODE_ID, strategy=Strategy.EXPLORE,
                        operator=Operator.DRAFT, family="data_curation", mechanism_tag=nid, diff_hash=nid,
                        status=status))
    store.set_state(f"config_diff:{nid}", cfg)
    return store.get_node(nid)


def test_nodes_on_a_building_dataset_are_held_and_voided_when_it_fails_or_was_never_requested(ctx, store, tmp_path):
    root = tmp_path / "bank"
    c = _ctx(ctx, root)
    _ds(root, "ready1", "ready")
    _ds(root, "busy", "extracting")
    _ds(root, "broke", "failed")
    store.ensure_root("r0")
    nodes = [_node(store, "n0001", {"data.dataset_dir": f"{root}/ready1"}),
             _node(store, "n0002", {"data.dataset_dir": f"{root}/busy/"}),
             _node(store, "n0003", {"data.dataset_dir": f"{root}/broke"}),
             _node(store, "n0004", {"data.dataset_dir": f"{root}/ghost"}),
             _node(store, "n0005", {"train.lr": 1.0})]
    voided = {}
    held = DB.gate(c, store, nodes, lambda n, why: voided.update({n.id: why}), cache={})
    assert held == {"n0002", "n0003", "n0004"} and set(voided) == {"n0003", "n0004"}
    assert "never requested" in voided["n0004"] and voided["n0003"].startswith("dataset broke failed")
    store.set_state(DB.K_BANK, {"requests": {"ghost": {"spec": SPEC, "by": ["n0004"], "state": "waiting", "ts": 0}}})
    voided.clear()
    assert "n0004" in DB.gate(c, store, nodes[3:4], lambda n, why: voided.update({n.id: why})) and not voided
    assert DB.gate(ctx, store, nodes, lambda *a: None) == set()                         # disabled
    assert DB.stand_in(c, f"{root}/busy") == c.cfg.databank.smoke_dataset
    assert DB.stand_in(c, f"{root}/ready1") == "" and DB.stand_in(c, "/data/nanopath_arrow") == ""


def test_index_knobs_read_and_fold_list_values():
    base = {"data.crop": [0.4, 1.0], "lr": 0.1}
    assert CP.value(base, "data.crop[0]") == 0.4 and CP.value(base, "data.crop[5]") is None and CP.value(base, "lr") == 0.1
    assert CP.expand(base, {"data.crop[0]": 0.2, "data.crop[1]": 0.9, "lr": 0.2}) == {"data.crop": [0.2, 0.9], "lr": 0.2}
    assert base == {"data.crop": [0.4, 1.0], "lr": 0.1}


@pytest.fixture
def sim(ctx, store):
    from loop.adapters.sim import SimAdapter
    return SimAdapter(ctx)


def _incumbent(ctx, store, adapter, nid="n0001"):
    """A node whose config has a data block (the sim's base has none) and train.num_workers."""
    root = store.get_node(ROOT_NODE_ID) or store.ensure_root(adapter.root_commit())
    wt = adapter.prepare_worktree(parent_commit=root.commit, name=nid)
    p = Path(wt.path) / "config.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg["data"] = {"dataset_dir": "/data/nanopath_arrow", "crop": [0.4, 1.0]}
    cfg["train"]["num_workers"] = 8
    p.write_text(yaml.safe_dump(cfg, sort_keys=True).replace("  crop:\n  - 0.4\n  - 1.0\n", "  crop: [0.4, 1.0]\n"))
    _, _, cfg_diff = adapter.diff(wt)
    commit = adapter.commit(wt, message=nid)
    adapter.reap_worktree(wt)
    store.put_node(Node(id=nid, task_id=ctx.task_id, commit=commit, parent_id=ROOT_NODE_ID, strategy=Strategy.EXPLORE,
                        operator=Operator.DRAFT, family="data_curation", mechanism_tag="inc", diff_hash=nid,
                        status=NodeStatus.CONFIRMED))
    store.set_state(f"config_diff:{nid}", dict(cfg_diff))
    return store.get_node(nid)


def test_adoption_only_for_listed_datasets_config_node_on_the_incumbent_into_a_campaign(ctx, store, sim, tmp_path):
    root = tmp_path / "bank"
    space = ({"key": "data.crop[0]", "low": 0.1, "high": 0.5, "scale": "log"}, {"key": "train.lr", "factor": 2.0},
             {"key": "nope.key", "low": 1, "high": 2})
    c = _ctx(ctx, root, adopt=("mixed_448",), num_workers=12, adopt_space=space)
    inc = _incumbent(c, store, sim)
    _ds(root, "mixed_448", "ready")
    _ds(root, "mpp050_448", "ready")
    ids = iter(f"n{i:04d}" for i in range(100, 200))
    snap = SimpleNamespace(incumbent_id=inc.id, nodes={})
    acts = DB.tick(c, store, sim, snap, new_id=lambda: next(ids), now=1.0)
    assert sum(a.startswith("databank: registered") for a in acts) == 2
    assert "databank: adoption node n0100 campaign (mixed_448 on the incumbent)" in acts
    assert sum("adoption node" in a for a in acts) == 1             # registered, not adopted
    n = store.get_node("n0100")
    assert n.parent_id == inc.id and n.status is NodeStatus.SCREENING and n.family == "data_curation"
    assert n.operator is Operator.DRAFT and "dataset=mixed_448" in n.notes
    assert store.get_state("config_diff:n0100") == {"data.dataset_dir": f"{root}/mixed_448", "train.num_workers": 12}
    plan = store.get_state("ideation:n0100")
    assert plan["campaign"] and [r["key"] for r in plan["search_space"]] == ["data.crop[0]", "train.lr"]
    assert CP.why(c, store, n) == "declared campaign" and CP.book(store)["queue"][0]["base"] == "n0100"
    from loop import design as D                                                        # the campaign's knobs
    with D._Tree(c, store, sim, "look", base_id="n0100") as t:
        sp = CP.space(c, store, n, t.text, t.base)
        assert [kb["key"] for kb in sp] == ["data.crop[0]", "train.lr"] and sp[0]["value"] == 0.4
        pt = t.node(CP.expand(t.base, {"data.crop[0]": 0.2}), knobs=[], tag="p", operator=None,
                    new_id=lambda: "n0199")
    assert store.get_state(f"config_diff:{pt}") == {"data.crop": [0.2, 1.0]}
    assert DB.tick(c, store, sim, snap, new_id=lambda: next(ids), now=2.0) == []        # once
    assert DB.book(store)["adopted"] == {"mixed_448": "n0100"}


def test_prune_candidates_and_digest_section(ctx, store, tmp_path):
    import time
    root = tmp_path / "bank"
    c = _ctx(ctx, root, retain_days=14)
    _ds(root, "old_unused", "ready", ready="2026-01-01T00:00:00")
    _ds(root, "old_used", "ready", ready="2026-01-01T00:00:00")
    _ds(root, "fresh", "ready", ready="2026-10-07T00:00:00")
    store.ensure_root("r0")
    _node(store, "n0001", {"data.dataset_dir": f"{root}/old_used"}, status=NodeStatus.CONFIRMED)
    _node(store, "n0002", {"data.dataset_dir": f"{root}/old_unused"}, status=NodeStatus.SCREENED_OUT)
    now = time.mktime(time.strptime("2026-10-08T00:00:00", "%Y-%m-%dT%H:%M:%S"))
    got = DB.prune_candidates(c, store, now)
    assert len(got) == 1 and got[0].startswith(f"{root}/old_unused (") and "452 GB" in got[0]
    DB.tick(c, store, None, None, new_id=None, now=now)
    text = DJ.render(DJ.numbers(c, store, now), now)
    assert "## Data bank: fresh, old_unused, old_used registered" in text and "old_unused (" in text
    assert "Data bank" not in DJ.render(DJ.numbers(ctx, store, now), now)


def test_driver_requests_a_finish_dataset_spec_launches_the_build_and_holds_its_node(tmp_path, monkeypatch):
    from loop.agents.researcher import diff_hash, set_yaml_value
    from loop.contracts import (Candidate, CandidateRequest, HandoffNote, HypothesisKind, HypothesisRecord,
                                ResearchResult)
    from tests.loop.test_tune import _driver
    ctx, driver, store, adapter = _driver(tmp_path)
    root = tmp_path / "bank"
    driver._spec_cfg = _ctx(dataclasses.replace(ctx, cfg=driver._spec_cfg), root).cfg   # the effective config's spec
    driver.tick()
    launched = []
    monkeypatch.setattr(DB, "launch", lambda ctx, spec, log: launched.append(spec.name) or 99999999)
    req = CandidateRequest(id="req-1", task_id=ctx.task_id, strategy=Strategy.EXPLORE, operator=Operator.DRAFT,
                           island="main", parent_ids=("root",), brief="b", brief_kind="literature")
    driver._intake(req, ResearchResult("failed", None, "empty_diff", plan={"dataset_spec": {**SPEC, "name": "a1"}}))
    assert DB.book(store)["requests"]["a1"]["by"] == ["req-1"]
    r0 = store.get_node("root")
    wt = adapter.prepare_worktree(parent_commit=r0.commit, name="idea")
    cfg = Path(wt.path) / "config.yaml"
    cfg.write_text(set_yaml_value(cfg.read_text(), "aug.strength", 0.8))
    diff_text, paths, cfg_diff = adapter.diff(wt)
    commit = adapter.commit(wt, message="idea")
    adapter.reap_worktree(wt)
    h = HypothesisRecord("h-x", ctx.task_id, HypothesisKind.TRIAL, "on a1", "on_a1", "augmentation", {}, "no gain")
    res = ResearchResult("ok", Candidate("req-2", ctx.task_id, commit, r0.commit, diff_text, diff_hash(diff_text),
                                         tuple(paths), dict(cfg_diff), h, HandoffNote("", ctx.task_id, "req-2", "a", "", "")),
                         plan={"dataset_spec": {**SPEC, "name": "a1"}})
    out = driver._intake(dataclasses.replace(req, id="req-2"), res)
    nid = out.split()[1]
    assert out.startswith("queued") and DB.book(store)["requests"]["a1"]["by"] == ["req-1", nid]
    store.set_state(f"config_diff:{nid}", {**cfg_diff, "data.dataset_dir": f"{root}/a1"})   # it trains on a1
    acts = driver.tick()
    assert launched == ["a1.json"] and any("build a1 launched" in a for a in acts)
    (root / "a1").mkdir(parents=True)
    (root / "a1" / "STATUS").write_text("extracting\n")
    for _ in range(4):
        driver.tick()
    assert store.get_node(nid).status is NodeStatus.QUEUED and not store.runs(node_id=nid)     # held in the queue
    (root / "a1" / "STATUS").write_text("failed\n")
    driver.tick()
    assert store.get_node(nid).status is NodeStatus.VOID
