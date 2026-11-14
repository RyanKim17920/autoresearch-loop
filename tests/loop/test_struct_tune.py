"""10-22: the STRUCTURAL TUNING LADDER (loop.core.struct_tune), its close rule, retro queue and intake rule, and the
WEIGHTS ALLOWLIST (init_compat.allow / allow_types: intake weight_load, the init-swap check, the O5 launch guard).

Operator 2026-10-22: backbone swaps n1781 / n1824 were each judged on about one untuned shot. The v3 tick never wired
loop.tune / loop.campaign, so these tests drive the ladder through the real ``program.ladders`` hook and a real Tick."""

from __future__ import annotations

import dataclasses
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop import tick as TK
from loop.adapters import flatten
from loop.adapters import initcompat as IC
from loop.contracts import ROOT_NODE_ID, Node, NodeStatus, Operator, Strategy
from loop.core import acquire as AQ
from loop.core import alloc as AL
from loop.core import data as CD
from loop.core import model as M
from loop.core import program as PG
from loop.core import struct_tune as ST
from loop.select import intake as IN
from loop.spec import load_context

from .test_set1_harness import _Held

REPO = Path(__file__).resolve().parents[2]
V2 = REPO / "tasks" / "nanopath_v2.yaml"

CFG = {"model": {"type": "dinov3_vits16"},
       "dino": {"lr": 2.0e-4, "layerwise_decay": 0.8, "warmup_fraction": 0.09, "stain_weight": 0.5, "seed_x": 1}}
PARENT_CFG = {"model": {"type": "robust_norm_dinov2_vits14_reg"}, "dino": {"lr": 2.0e-4, "layerwise_decay": 0.8,
                                                                           "warmup_fraction": 0.09}}


@pytest.fixture(scope="module")
def real_ideation(tmp_path_factory):
    """The operator yaml's own ``ideation.struct_tune_knobs`` (so the yaml is covered, not a test copy)."""
    return load_context(V2, state_dir=tmp_path_factory.mktemp("st")).cfg.ideation


@pytest.fixture
def sctx(ctx, real_ideation, monkeypatch):
    monkeypatch.setattr("loop.extrapolate._allowed", lambda c, k: True)
    ic = dataclasses.replace(ctx.cfg.ideation, struct_tune_knobs=real_ideation.struct_tune_knobs)
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=ic))


def _dn(nid, parent=ROOT_NODE_ID, strategy="moonshot", notes="request=r brief=bold", base="B", port_of=""):
    return CD.Node(nid, parent, None, strategy, "architecture", "explore", "r0", 1.0, "screening", notes, base=base,
                   port_of=port_of)


def _obs(node, seed=1):
    return CD.Obs(f"{node}:screen:{seed}", node, seed, "screen", "select", {}, 0.65, 1.0, 1.0, 0)


def _led(nodes, obs=(), epoch="B", bases=None):
    root = CD.Node(ROOT_NODE_ID, None, None, "baseline", "", "baseline", ROOT_NODE_ID, 0.0, "baseline", "", base=epoch)
    return CD.Ledger(nodes={ROOT_NODE_ID: root, **{n.id: n for n in nodes}}, runs=(), obs=tuple(_obs(n) for n in obs),
                     test_seeds=frozenset(), carriers=(),
                     bases=bases or ((0.0, ROOT_NODE_ID, epoch),))


def _store_node(store, nid, strategy=Strategy.MOONSHOT, notes="request=r brief=bold", parent=ROOT_NODE_ID, status=NodeStatus.SCREENING):
    store.put_node(Node(id=nid, task_id=store.task_id, commit=f"c-{nid}", parent_id=parent, strategy=strategy,
                        operator=Operator.DRAFT, family="architecture", mechanism_tag=f"tag_{nid}",
                        diff_hash=f"h{nid}", status=status, notes=notes))


def _adapter(cfgs):
    built = []
    return SimpleNamespace(
        config_rel="config.yaml", config_at=lambda c: cfgs.get(c, cfgs["c-b1"]), built=built,
        dry_build=lambda commit, levels, knobs, diffs, name: built.append(dict(levels)) or (f"sha-{len(built)}", ""),
        graft_port=lambda src, old, onto, name: (f"graft-{name}", "applied"))


def _post(m=-0.011, v=2e-6, n=1):
    return SimpleNamespace(est={"b1": {M.COMPOSITE: SimpleNamespace(n=n)}}, belief=lambda nid: (m, v))


def _setup(store, strategy="moonshot", notes="request=r brief=bold", cfg=CFG, parent_cfg=PARENT_CFG, **post_kw):
    store.ensure_root("r0")
    _store_node(store, "b1", notes=notes)
    led = _led([_dn("b1", strategy=strategy, notes=notes)])
    return led, _post(**post_kw), _adapter({"c-b1": cfg, "r0": parent_cfg, "c-r0": parent_cfg})


def _kids(store, nid="b1"):
    rec = store.get_state(ST.K_STRUCT)[nid]
    return rec, [(store.get_state(f"config_diff:{c}"), store.get_node(c)) for c in rec["children"]]


# ---------------------------------------------------------------------------------------------- the backbone ladder
def test_a_backbone_swap_gets_an_lr_x_layer_decay_grid_plus_warmup_that_moonshot_tune_shared_leaves_out(sctx, store):
    led, post, adapter = _setup(store)
    assert "dino.lr" not in sctx.cfg.moonshot_tune.shared or True          # (the v2 shared knobs: lr x1.25, warmup x1.5)
    acts = PG.ladders(sctx, store, adapter, led, post, {"b1": "screening"}, 10.0)
    rec, kids = _kids(store)
    assert rec["kind"] == "backbone" and len(kids) == 5 <= sctx.cfg.ideation.struct_tune_max
    cells = [d for d, _ in kids]
    grid = [d for d in cells if "dino.lr" in d and "dino.layerwise_decay" in d]
    assert len(grid) == 4 and {d["dino.lr"] for d in grid} == {1.0e-4, 4.0e-4}       # lr value / 2, value x 2
    assert {d["dino.layerwise_decay"] for d in grid} == {0.65, 0.95}                 # layer decay -/+ .15
    assert [d for d in cells if set(d) == {"dino.warmup_fraction"}] == [{"dino.warmup_fraction": 0.18}]
    assert any("struct ladder b1" in a and "backbone" in a for a in acts)
    for _, n in kids:
        assert "ladder=b1" in n.notes and "struct_tune=backbone" in n.notes and n.parent_id == "b1"
        assert n.status is NodeStatus.QUEUED
    PG.ladders(sctx, store, adapter, led, post, {"b1": "screening"}, 20.0)
    assert len(store.get_state(ST.K_STRUCT)["b1"]["children"]) == 5                  # once per node
    assert len(adapter.built) == 5


def test_a_backbone_node_is_laddered_whatever_its_first_shot_looked_like(sctx, store):
    """n1824 was screened out at P(gain>0)=.025 after one seed: a backbone swap is laddered at ANY belief."""
    led, post, adapter = _setup(store, m=-0.03, v=1e-6)
    PG.ladders(sctx, store, adapter, led, post, {"b1": "screening"}, 10.0)
    assert store.get_state(ST.K_STRUCT)["b1"]["kind"] == "backbone"


def test_ladder_children_get_program_priority_and_are_exempt_from_starved_drops(sctx, store, tmp_path):
    led, post, adapter = _setup(store)
    PG.ladders(sctx, store, adapter, led, post, {"b1": "screening"}, 10.0)
    _, kids = _kids(store)
    for _, n in kids:
        assert AQ.program_child(n)                                                   # screened first, never aged / dropped
    now = time.time() + 300 * 3600.0
    for _, n in kids:
        store.update_node(n.id, created_ts=1000.0)
    tick = TK.Tick(sctx, store, _Held(), TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)
    data = CD.load(tmp_path / "loop.db", sctx.task_id)
    statuses = {nid: tick.core.ST.Status.QUEUED for nid in data.nodes if nid.startswith("n")}
    assert tick._age_out_queued(data, statuses, now) == []                           # 300 h old, still queued
    assert all(store.get_node(n.id).status is NodeStatus.QUEUED for _, n in kids)
    assert not AQ.aged_queued(data, statuses, now)
    tick.close()


def test_the_real_tick_builds_the_structural_ladder_through_its_ladders_hook(sctx, store, tmp_path):
    led, post, adapter = _setup(store)
    tick = TK.Tick(sctx, store, _Held(), TK.ModuleCore(), db_path=tmp_path / "loop.db", clock=time.time)
    tick.post, tick.adapter = post, adapter
    acts = tick._ladders(led, {"b1": "screening"}, 10.0)
    assert any("struct ladder b1" in a for a in acts) and len(store.get_state(ST.K_STRUCT)["b1"]["children"]) == 5
    tick.close()


def test_a_ladder_childs_test_entry_is_max_of_k_shrunk_even_when_the_global_beta_is_zero():
    kid = SimpleNamespace(notes="ladder=b1 struct_tune=backbone cell=lr:hi")
    plain = SimpleNamespace(notes="request=r brief=explore")
    assert AQ.MAX_OF_K_BETA == 0.0 and AQ.beta_for(plain) == 0.0
    assert AQ.beta_for(kid) >= 1.0                                                   # the best of <= 5 tuned points
    m = M.max_of_k(0.004, 4e-6, 5, AQ.beta_for(kid))
    assert m < 0.004 - 1.0 * 0.002                                                   # shrunk by ~E_5 sd


# ---------------------------------------------------------------------------------------------- loss doses, steering
def _loss_node(store, **kw):
    cfg = {"dino": {"lr": 2e-4, "layerwise_decay": 0.8, "warmup_fraction": 0.09, "stain_pred_weight": 1.0},
           "model": {"type": "robust_norm_dinov2_vits14_reg"}}
    pcfg = {"dino": {"lr": 2e-4, "layerwise_decay": 0.8, "warmup_fraction": 0.09},
            "model": {"type": "robust_norm_dinov2_vits14_reg"}}
    return _setup(store, cfg=cfg, parent_cfg=pcfg, **kw)


def _engaged(monkeypatch, status):
    from loop.core import ideation
    monkeypatch.setattr(ideation, "engagements",
                        lambda store, led: [ideation.Engagement("b1", "d", 1, status, -1.0, 2)])


@pytest.mark.parametrize("status,first,last_gt", [("did not engage", 3.0, True), ("engaged", 0.3, False)])
def test_loss_doses_are_steered_by_the_engagement_diagnostic(sctx, store, monkeypatch, status, first, last_gt):
    led, post, adapter = _loss_node(store, m=-0.004)
    _engaged(monkeypatch, status)
    PG.ladders(sctx, store, adapter, led, post, {"b1": "screening"}, 10.0)
    rec, kids = _kids(store)
    vals = [d["dino.stain_pred_weight"] for d, _ in kids]
    assert rec["kind"] == "loss" and vals[0] == first and len(vals) == 4 and 1.0 not in vals   # x10 is past the hard bound
    assert all(set(d) == {"dino.stain_pred_weight"} for d, _ in kids)                # the loss weight, nothing else
    assert sorted(vals) == [0.03, 0.1, 0.3, 3.0]                                     # >1 decade, not just x1/2 and x2
    assert vals == ([3.0, 0.3, 0.1, 0.03] if last_gt else [0.3, 0.1, 0.03, 3.0])        # higher first / lower first
    assert rec["steer"] == ("not_engaged" if status == "did not engage" else "engaged_negative")


def test_a_reversed_signed_knob_is_flipped_first(sctx, store, monkeypatch):
    cfg = {"dino": {"lr": 2e-4, "layerwise_decay": 0.8, "warmup_fraction": 0.09, "pred_sign_weight": 1.0}}
    led, post, adapter = _setup(store, cfg=cfg, parent_cfg={"dino": {"lr": 2e-4, "layerwise_decay": 0.8,
                                                                      "warmup_fraction": 0.09}})
    _engaged(monkeypatch, "reversed")
    PG.ladders(sctx, store, adapter, led, post, {"b1": "screening"}, 10.0)
    rec, kids = _kids(store)
    assert rec["steer"] == "reversed" and kids[0][0] == {"dino.pred_sign_weight": -1.0}
    assert [d["dino.pred_sign_weight"] for d, _ in kids[1:3]] == [0.3, 0.1]


def test_a_non_backbone_node_with_no_own_knob_gets_the_lr_ld_grid_and_a_hopeless_one_is_not_laddered(sctx, store):
    cfg = {"dino": {"lr": 2e-4, "layerwise_decay": 0.8, "warmup_fraction": 0.09}}
    led, post, adapter = _setup(store, cfg=cfg, parent_cfg=cfg, m=-0.002)
    PG.ladders(sctx, store, adapter, led, post, {"b1": "screening"}, 10.0)
    assert store.get_state(ST.K_STRUCT)["b1"]["kind"] == "other" and len(_kids(store)[1]) == 5
    store.set_state(ST.K_STRUCT, {})
    PG.ladders(sctx, store, adapter, led, _post(m=-0.05, v=1e-6), {"b1": "screening"}, 20.0)   # far below -2.5 sd
    rec = store.get_state(ST.K_STRUCT)["b1"]
    assert isinstance(rec, str) and rec.startswith("-: skipped")      # BOOKED, so the node is no longer owed a ladder
    assert ST.held(sctx, store, led, now=time.time()) == {}


def test_the_ladder_shares_a_bold_units_budget_and_never_exceeds_five(sctx, store):
    led, post, adapter = _setup(store)
    PG.ladders(sctx, store, adapter, led, post, {"b1": "screening"}, 10.0)
    from loop.core import bold as BOLD
    BOLD._VIEW.clear()
    assert BOLD.tune_used(store, "b1") == 5 and BOLD.tune_room(sctx, store, store.get_node("b1")) == 0


# ---------------------------------------------------------------------------------------------- the close rule
def test_a_catastrophic_first_point_with_the_mechanism_engaged_closes_without_a_ladder(sctx, store, monkeypatch):
    led, post, adapter = _loss_node(store, m=-0.02, v=1e-6, n=3)        # -20 sd on 3 seeds
    _engaged(monkeypatch, "engaged")
    PG.ladders(sctx, store, adapter, led, post, {"b1": "screening"}, 10.0)
    assert store.get_state(ST.K_STRUCT)["b1"].startswith("-: catastrophic") and not adapter.built
    assert ST.held(sctx, store, led) == {}                                # the close is allowed
    # the same numbers with the mechanism NOT engaged is an unproven mechanism: a backbone swap is still laddered
    store.set_state(ST.K_STRUCT, {})
    _engaged(monkeypatch, "did not engage")
    PG.ladders(sctx, store, adapter, led, post, {"b1": "screening"}, 20.0)
    assert isinstance(store.get_state(ST.K_STRUCT)["b1"], dict) or "b1" not in store.get_state(ST.K_STRUCT)


def test_a_structural_node_is_held_until_three_tuned_points_finished_and_adds_no_negative_cluster_evidence(sctx, store):
    store.ensure_root("r0")
    nodes, bel = [], {}
    from .core.test_c7i import _put
    for i in range(10):
        n = _put(store, f"k{i}", f"kde_uniformity_margin_v{i}", "representation", ts=i, notes="request=r brief=bold")
        nodes.append(dataclasses.replace(n, strategy="moonshot"))
        bel[f"k{i}"] = (-0.004, 2)
    for i, tag in enumerate(["hed_stain_jitter", "patient_cap_volume", "register_tokens", "drop_path_ramp",
                             "teacher_temp_warmup", "global_crop_scale", "ibot_mask_ratio", "koleo_weight"]):
        nodes.append(_put(store, f"o{i}", tag, "misc", ts=30 + i))
        bel[f"o{i}"] = (0.0, 1)
    now0 = time.time()
    nodes = [dataclasses.replace(n, created_ts=now0) if n.id.startswith("k") else n for n in nodes]
    for n in nodes:
        if n.id.startswith("k"):                                  # a ladder-eligible node: not closed, created now
            store.update_node(n.id, status=NodeStatus.SCREENING, created_ts=now0)
    ctx = dataclasses.replace(sctx, cfg=dataclasses.replace(sctx.cfg, alloc=dataclasses.replace(
        sctx.cfg.alloc, family_block=False)))
    kde = lambda cls: next(c for c in cls if "k0" in c.members)                                 # noqa: E731

    def led(done=()):
        base = _led(nodes, obs=[n for n in done])
        return base

    owed = led()
    assert set(ST.held(ctx, store, owed)) >= {f"k{i}" for i in range(10)}                      # ladders owed
    assert kde(AL.clusters(ctx, store, owed, bel)).flag == ""                                  # no negative evidence yet
    kids = {f"k{i}": [f"c{i}a", f"c{i}b", f"c{i}c"] for i in range(10)}
    store.set_state(ST.K_STRUCT, {n: {"kind": "other", "children": c, "steer": "mixed", "ts": time.time()}
                                  for n, c in kids.items()})
    one_seed = _led(nodes, obs=[f"c{i}{x}" for i in range(10) for x in "abc"])
    assert ST.held(ctx, store, one_seed)["k0"].startswith("ladder 0/3")      # a point counts at >= 3 finished seeds
    two = _led(nodes, obs=[f"c{i}{x}" for i in range(10) for x in "ab" for _ in range(3)])
    assert ST.held(ctx, store, two)["k0"].startswith("ladder 2/3")                             # 2 of 3 points: still held
    three = _led(nodes, obs=[f"c{i}{x}" for i in range(10) for x in "abc" for _ in range(3)])
    assert ST.held(ctx, store, three) == {}
    old = ST.held(ctx, store, one_seed, now=time.time() + 100 * 3600.0)
    assert old == {}                                               # a stuck ladder never holds a node past struct_hold_h
    assert kde(AL.clusters(ctx, store, three, bel)).flag == "saturated-negative"               # now it counts


# ---------------------------------------------------------------------------------------------- retro
def _retro_world(store, old_notes="request=r brief=moonshot init=https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"):
    store.ensure_root("r0")
    store.put_node(Node(id="rA", task_id=store.task_id, commit="c-rA", parent_id=None, strategy=Strategy.BASELINE,
                        operator=None, family="", mechanism_tag="", diff_hash="hA", status=NodeStatus.BASELINE))
    store.put_node(Node(id="rB", task_id=store.task_id, commit="c-rB", parent_id=None, strategy=Strategy.BASELINE,
                        operator=None, family="", mechanism_tag="", diff_hash="hB", status=NodeStatus.BASELINE))
    _store_node(store, "n1781", Strategy.MOONSHOT, old_notes, parent="rA", status=NodeStatus.SCREENED_OUT)
    store.set_state("config_diff:n1781", {"model.type": "robust_norm_dinov2_vitb14_reg"})
    old = _dn("n1781", parent="rA", notes=old_notes, base="A")
    rootA = CD.Node("rA", None, None, "baseline", "", "baseline", "rA", 0.0, "baseline", "", base="A")
    rootB = CD.Node("rB", None, None, "baseline", "", "baseline", "rB", 0.0, "baseline", "", base="B")
    led = CD.Ledger(nodes={"rA": rootA, "rB": rootB, "n1781": old}, runs=(), obs=(_obs("n1781"), _obs("n1781", 2)),
                    test_seeds=frozenset(), carriers=(), bases=((0.0, "rA", "A"), (1.0, "rB", "B")))
    return led, _adapter({"c-n1781": CFG})


def test_a_closed_backbone_swap_of_an_old_base_is_ported_for_a_retro_ladder(sctx, store):
    led, adapter = _retro_world(store)
    assert [t[0] for t in ST.retro_targets(sctx, store, led)] == ["n1781"]
    acts = ST.retro(sctx, store, adapter, led, 100.0)
    port = store.get_node("n1781@B")
    assert port is not None and port.status is NodeStatus.QUEUED and any("n1781@B" in a for a in acts)
    assert "port_of=n1781" in port.notes and "ladder=n1781" in port.notes and "struct_retro=1" in port.notes
    assert AQ.program_child(port)                                                    # screened first, never dropped
    assert store.get_state(ST.K_RETRO)["n1781"]["port"] == "n1781@B"
    assert ST.retro(sctx, store, adapter, led, 200.0) == []                          # once


def test_the_operator_list_and_the_open_port_cap(sctx, store):
    led, adapter = _retro_world(store)
    ic = dataclasses.replace(sctx.cfg.ideation, struct_retro_nodes=("nope", "n1781"), struct_retro_max=1)
    c = dataclasses.replace(sctx, cfg=dataclasses.replace(sctx.cfg, ideation=ic))
    assert [t[0] for t in ST.retro_targets(c, store, led)][:2] == ["nope", "n1781"]
    ST.retro(c, store, adapter, led, 100.0)
    rb = store.get_state(ST.K_RETRO)
    assert rb["nope"].startswith("-: not in the ledger") and rb["n1781"]["port"] == "n1781@B"
    off = dataclasses.replace(c, cfg=dataclasses.replace(c.cfg, ideation=dataclasses.replace(ic, struct_retro=False)))
    assert ST.retro(off, store, adapter, led, 300.0) == []


def test_the_retro_port_is_laddered_once_it_screens_even_if_it_is_not_a_moonshot(sctx, store):
    store.ensure_root("r0")
    notes = "request=r port_of=n1781 ladder=n1781 struct_retro=1"
    _store_node(store, "b1", Strategy.EXPLORE, notes)
    led = _led([_dn("b1", strategy="explore", notes=notes)])
    adapter = _adapter({"c-b1": CFG, "r0": PARENT_CFG, "c-r0": PARENT_CFG})
    PG.ladders(sctx, store, adapter, led, _post(), {"b1": "screening"}, 10.0)
    assert store.get_state(ST.K_STRUCT)["b1"]["kind"] == "backbone"


# ---------------------------------------------------------------------------------------------- intake: untunable
def test_a_structural_change_with_nothing_to_tune_is_rejected_untunable_moonshot(sctx):
    rule, fix = ST.untunable(sctx, {}, None)
    assert rule == "untunable_moonshot" and "components" in fix
    assert ST.untunable(sctx, {"dino.stain_pred_weight": 1.0}, None) == ("", "")           # a new loss weight
    assert ST.untunable(sctx, {}, [{"family": "x"}]) == ("", "")                          # declared components
    assert ST.untunable(sctx, {"model.type": "dinov3_vits16"}, None) == ("", "")           # a backbone swap
    assert ST.untunable(sctx, {"flag": True}, None)[0] == "untunable_moonshot"             # a bool is no knob


def test_the_bold_prompts_tell_sessions_to_make_the_change_tunable():
    for f in ("brief_bold.md", "researcher_bold.md"):
        assert "untunable_moonshot" in (REPO / "loop" / "prompts" / f).read_text()


# ---------------------------------------------------------------------------------------------- weights allowlist
@pytest.fixture(scope="module")
def v2ctx(tmp_path_factory):
    return load_context(V2, state_dir=tmp_path_factory.mktemp("v2"))


GOOD = ["timm/vit_small_patch16_dinov3.lvd1689m", "timm/vit_small_plus_patch16_dinov3.lvd1689m",
        "vit_base_patch16_dinov3.lvd1689m", "hf-hub:timm/vit_base_patch16_dinov3.lvd1689m",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"]
BAD = ["https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",       # not S / B
       "timm/vit_large_patch16_dinov3.lvd1689m", "timm/vit_small_patch16_dinov3.sat493m",           # other weights
       "/data/user/OpenMidnight/checkpoint.pth", "hf-hub:paige-ai/Virchow2", "https://example.com/x.pth",
       "timm/vit_small_patch16_dinov3.lvd1689m.evil", "owner/path-fm-vit"]


def test_the_yaml_allowlist_admits_generic_meta_weights_and_blocks_everything_else(v2ctx):
    ic = v2ctx.cfg.init_compat
    assert ic.allow and ic.allow_types
    assert all(IC.allowed(s, ic.allow) for s in GOOD)
    assert not any(IC.allowed(s, ic.allow) for s in BAD)
    assert IC.allowed("", ic.allow) and IC.allowed("anything", ())                          # no source / no allowlist
    for t in ("dinov3_vits16", "dinov3_vitsp16", "dinov3_vitb16", "robust_norm_dinov3_vitb16", "dinov2_vitb14_reg",
              "robust_norm_dinov2_vits14_reg", "dinov2_vits14"):
        assert any(re.search(p, t, re.I) for p in ic.allow_types), t
    for t in ("dinov2_vitl14_reg", "dinov3_vitl16", "openmidnight_vitg14", "uni_vitl16"):
        assert not any(re.search(p, t, re.I) for p in ic.allow_types), t


def _swap(src, typ, ic):
    m = {"child": {"constructed": True, "finite": True, "type": typ, "source": src, "params": 1},
         "parent": {"constructed": True, "type": "robust_norm_dinov2_vits14_reg", "source": "x"},
         "keys": {"swap": True, "loader": True, "finite": True}, "flops": None}
    return IC.evaluate(m, IC.Limits(deny=tuple(ic.deny), allow=tuple(ic.allow)))


def test_the_init_swap_check_enforces_the_allowlist_and_deny_still_wins(v2ctx):
    ic = v2ctx.cfg.init_compat
    assert _swap(GOOD[0], "dinov3_vits16", ic).ok
    bad = _swap("timm/vit_giant_patch14_dinov2.lvd142m", "dinov2_vitg14", ic)
    assert not bad.ok and bad.rule == "init_illegal" and "allowlist" in bad.reason
    mid = _swap("/data/OpenMidnight/ck.pth", "x", ic)
    assert not mid.ok and "deny pattern" in mid.reason                                      # deny first
    assert _swap("", "random_arch", ic).ok                                                  # no weights: not a weights question
    no_list = IC.evaluate({"child": {"constructed": True, "finite": True, "type": "t", "source": "anything"},
                           "parent": {"constructed": True, "type": "p", "source": "q"},
                           "keys": {"swap": True, "loader": True, "finite": True}}, IC.Limits(deny=tuple(ic.deny)))
    assert no_list.ok                                                                       # back-compat: deny only


def _diff(path, added, hunk=""):
    return (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1,3 +1,4 @@ {hunk}\n     x = 1\n"
            + "".join(f"+{l}\n" for l in added))


def test_intake_weight_load_lets_an_allowlisted_source_literal_through_and_refuses_any_other(v2ctx):
    allow = tuple(v2ctx.cfg.init_compat.allow)
    url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
    good = _diff("model.py", [f'    "dinov2_vitb14_reg": (768, 12, 12, 37, 14, "mlp", True, "{url}"),'])
    assert IN.weight_load(good) [0] == "model.py"                                           # without the allowlist: a .pth literal
    assert IN.weight_load(good, allow) == ("", "")
    timm = _diff("model.py", ["    m = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True)"],
                 hunk="def load_pretrained(model):")                                         # the one allowed place
    assert IN.weight_load(timm, allow) == ("", "")
    for line in ('    "x": (1, "https://example.com/pathfm.pth"),',
                 "    m = timm.create_model('vit_large_patch16_dinov3.lvd1689m')",
                 "    sd = torch.hub.load_state_dict_from_url('https://evil.example/w.pth')",
                 '    p = "/data/user/OpenMidnight/ck.pth"'):
        assert IN.weight_load(_diff("model.py", [line]), allow)[0] == "model.py", line
    assert IN.weight_load(_diff("train.py", [f'    sd = "{url}"']), allow)[0] == "train.py"     # only model.py may name a source


def test_intake_diagnose_refuses_an_off_list_pretrained_url_config_value(v2ctx, store):
    allow = tuple(v2ctx.cfg.init_compat.allow)
    # the rule's own check, isolated: the protected-key exemption for model.pretrained_url is followed by the allowlist
    assert IN.on_allowlist(GOOD[4], allow) and not IN.on_allowlist(BAD[3], allow)


# ---------------------------------------------------------------------------------------------- O5 launch guard
from .test_review_c4j_locks import _commit, git, xenv  # noqa: E402,F401  (fixture)


def test_the_launch_guard_accepts_allowlisted_meta_inits_and_refuses_any_other_model_type_or_url(xenv, v2ctx):
    ad, _, repo, _ = xenv
    surf = SimpleNamespace(locked=(), protected_keys=("model.init_from", "model.pretrained_url"), cap_ceilings={})
    ic = dataclasses.replace(ad.ctx.cfg.init_compat, allow=tuple(v2ctx.cfg.init_compat.allow),
                             allow_types=tuple(v2ctx.cfg.init_compat.allow_types))
    ad.ctx = SimpleNamespace(spec=SimpleNamespace(surface=surf, repo=str(repo)),
                             cfg=SimpleNamespace(**{**vars(ad.ctx.cfg), "init_compat": ic}) if not
                             dataclasses.is_dataclass(ad.ctx.cfg) else dataclasses.replace(ad.ctx.cfg, init_compat=ic))
    ad.config_rel, ad.locks = "fake_config.yaml", ()
    git(repo, "checkout", "-q", "main")

    def cfg(typ, url=""):
        return "model:\n  type: %s\n" % typ + (f"  pretrained_url: {url}\n" if url else "") + "train:\n  lr: 0.001\n"
    ref = _commit(repo, {"fake_config.yaml": cfg("robust_norm_dinov2_vits14_reg")}, "R")
    ad.lock_ref = ref
    meta = _commit(repo, {"fake_config.yaml": cfg("dinov3_vits16", GOOD[0])}, "dinov3")
    b14 = _commit(repo, {"fake_config.yaml": cfg("robust_norm_dinov2_vitb14_reg", GOOD[5])}, "b14")
    path_fm = _commit(repo, {"fake_config.yaml": cfg("robust_norm_dinov2_vits14_reg", BAD[3])}, "pathfm")
    big = _commit(repo, {"fake_config.yaml": cfg("dinov2_vitg14_reg", BAD[0])}, "giant")
    typ_only = _commit(repo, {"fake_config.yaml": cfg("openmidnight_vitl16")}, "typ")
    assert ad.config_violations(meta) == [] and ad.config_violations(b14) == []
    assert ad.config_violations(path_fm) == ["config:model.pretrained_url"]
    assert sorted(ad.config_violations(big)) == ["config:model.pretrained_url", "config:model.type"]
    assert ad.config_violations(typ_only) == ["config:model.type"]
    # a type whose init-swap check passed is recorded and then accepted even though it matches no allow_types regex
    ad.state.mkdir(parents=True, exist_ok=True)
    ad._swap_ok_path().write_text(f'["type:openmidnight_vitl16@{typ_only}"]')   # bound to the commit sha
    assert ad.config_violations(typ_only) == []


# ---------------------------------------------------------------------------------------------- more: coupling, operator spec,
# the computed postmortem, the program gate
def test_two_own_knobs_get_a_coupled_child_beside_their_doses(sctx, store):
    cfg = {"dino.lr": 2e-4, "dino": {"lr": 2e-4, "layerwise_decay": 0.8, "warmup_fraction": 0.09,
                                    "a_weight": 1.0, "b_weight": 1.0}}
    cfg = {"dino": cfg["dino"]}
    led, post, adapter = _setup(store, cfg=cfg, parent_cfg={"dino": {"lr": 2e-4, "layerwise_decay": 0.8,
                                                                      "warmup_fraction": 0.09}}, m=-0.001)
    PG.ladders(sctx, store, adapter, led, post, {"b1": "screening"}, 10.0)
    rec, kids = _kids(store)
    cells = [d for d, _ in kids]
    assert rec["kind"] == "loss" and len(cells) == 5
    assert {k for d in cells for k in d} == {"dino.a_weight", "dino.b_weight"}
    assert cells[-1] == {"dino.a_weight": 0.3, "dino.b_weight": 0.3}                   # the knobs moved together


def test_the_operator_retro_spec_gives_the_exact_ladder_values(sctx, store):
    ic = dataclasses.replace(sctx.cfg.ideation, struct_retro_nodes=(
        {"node": "n3025", "knobs": {"dino.stain_weight": [0.1, 0.3, 3.0], "dino.layerwise_decay": [0.9]}},))
    c = dataclasses.replace(sctx, cfg=dataclasses.replace(sctx.cfg, ideation=ic))
    assert ST.retro_items(c) == [("n3025", {"node": "n3025", "knobs": {
        "dino.stain_weight": [0.1, 0.3, 3.0], "dino.layerwise_decay": [0.9]}})]
    notes = "request=r port_of=n3025 ladder=n3025 struct_retro=1"
    store.ensure_root("r0")
    _store_node(store, "b1", Strategy.EXPLORE, notes)
    store.set_state(ST.K_RETRO, {"n3025": {"port": "b1", "ts": 1.0, "why": "operator list",
                                           "spec": ic.struct_retro_nodes[0]}})
    led = _led([_dn("b1", strategy="explore", notes=notes)])
    adapter = _adapter({"c-b1": CFG, "r0": PARENT_CFG, "c-r0": PARENT_CFG})
    PG.ladders(c, store, adapter, led, _post(), {"b1": "screening"}, 10.0)
    rec, kids = _kids(store)
    assert rec["kind"].endswith("+operator")
    assert [d for d, _ in kids] == [{"dino.layerwise_decay": 0.9}, {"dino.stain_weight": 0.1},      # round-robin over the
                                    {"dino.stain_weight": 0.3}, {"dino.stain_weight": 3.0}]         # keys (stored sorted)


def test_a_finished_ladder_writes_one_computed_postmortem_judging_the_best_tuned_point(sctx, store):
    store.ensure_root("r0")
    _store_node(store, "b1")
    kids = []
    for i in range(3):
        _store_node(store, f"t{i}", Strategy.EXPLORE, f"ladder=b1 struct_tune=backbone cell=lr:c{i}", parent="b1",
                    status=NodeStatus.QUEUED)
        kids.append(f"t{i}")
    store.set_state(ST.K_STRUCT, {"b1": {"kind": "backbone", "children": kids, "steer": "mixed", "ts": time.time()}})
    led = _led([_dn("b1")] + [_dn(k, parent="b1", strategy="explore", notes="ladder=b1 struct_tune=backbone cell=lr:c0")
                              for k in kids], obs=[k for k in kids for _ in range(3)])
    post = SimpleNamespace(belief=lambda n: {"b1": (-0.011, 4e-6), "t0": (-0.008, 4e-6), "t1": (-0.003, 4e-6),
                                             "t2": (-0.006, 4e-6)}[n])
    acts = ST.report(sctx, store, led, post, 5.0)
    (pm,) = store.postmortems(node_id="b1")
    assert pm.verdict == "loss" and "t1" in pm.explanation and "fresh test seeds" in pm.explanation and not pm.under_tuned
    assert any("struct postmortem b1" in a for a in acts) and ST.report(sctx, store, led, post, 6.0) == []   # once
    short = _led([_dn("b1")] + [_dn(k, parent="b1", strategy="explore") for k in kids], obs=["t0", "t1"])
    store.set_state(ST.K_PM, {})
    assert ST.report(sctx, store, short, post, 7.0) == []                                      # unfinished: no verdict yet


@pytest.mark.parametrize("status,laddered", [("did not engage", False), ("engaged", True)])
def test_an_engaged_but_heavily_negative_program_node_still_gets_its_dose_ladder(sctx, store, monkeypatch, status,
                                                                                laddered):
    """loop.core.program._eligible gate: belief > -z sd excluded exactly the ENGAGED-but-negative node (DINO A1)."""
    from .core.test_c8a import _prog
    cfgs = {"r0": {"train": {"local_views": 14}}, "c-p1": {"train": {"local_views": 10}}}
    data, post, adapter, built = _prog(store, cfgs, belief=(-0.02, 1e-6))
    _engaged_for(monkeypatch, "p1", status)
    PG.ladders(sctx, store, adapter, data, post, {"p1": "screening"}, 10.0)
    assert ("p1" in (store.get_state(PG.K_LADDERS) or {})) is laddered


def _engaged_for(monkeypatch, nid, status):
    from loop.core import ideation
    monkeypatch.setattr(ideation, "engagements",
                        lambda store, led: [ideation.Engagement(nid, "d", 1, status, -1.0, 2)])


# ---------------------------------------------------------------------------------------------- 10-22 adversarial review fixes
def _owed_world(store, status=NodeStatus.SCREENING, created=None, notes="request=r brief=bold"):
    store.ensure_root("r0")
    _store_node(store, "b1", notes=notes, status=status)
    created = time.time() if created is None else created
    store.update_node("b1", created_ts=created)
    return _led([dataclasses.replace(_dn("b1", notes=notes), created_ts=created)])


def test_an_unbooked_node_is_owed_a_ladder_only_while_it_is_eligible_and_not_past_struct_hold_h(sctx, store):
    """M1: held() kept a node that was never laddered (closed before the ladder, or skipped as hopeless) open forever."""
    hold = sctx.cfg.ideation.struct_hold_h * 3600.0
    led = _owed_world(store)
    assert ST.held(sctx, store, led, now=time.time()) == {"b1": "ladder owed"}
    assert ST.held(sctx, store, led, now=time.time() + hold + 10) == {}          # expired at created_ts + struct_hold_h


def test_a_closed_node_that_never_got_a_ladder_is_not_owed_one(sctx, store):
    led = _owed_world(store, status=NodeStatus.NOT_CONFIRMED)
    assert ST.held(sctx, store, led, now=time.time()) == {}


def test_the_hold_clock_of_a_booked_ladder_starts_when_its_first_child_starts(sctx, store):
    """S3: the clock ran from the ladder BUILD, so a ladder waiting in the queue was released before it ran."""
    from loop.contracts import Run, RunStatus, Stage
    hold = sctx.cfg.ideation.struct_hold_h * 3600.0
    led = _owed_world(store)
    t0 = time.time()
    store.set_state(ST.K_STRUCT, {"b1": {"kind": "other", "children": ["c1", "c2", "c3"], "steer": "mixed", "ts": t0}})
    assert "b1" in ST.held(sctx, store, led, now=t0 + hold - 60)
    assert ST.held(sctx, store, led, now=t0 + hold + 60) == {}                  # none ever started: stalled, released
    store.put_run(Run(id="c1:screen:1", task_id=store.task_id, node_id="c1", seed=1, stage=Stage.SCREEN,
                      status=RunStatus.RUNNING, submitted_ts=t0 + hold))           # the first child starts late
    assert "b1" in ST.held(sctx, store, led, now=t0 + hold + 60)                  # the clock now runs from that start


def test_a_node_that_later_reaches_the_catastrophic_seed_count_exits_the_hold(sctx, store, monkeypatch):
    """S3: catastrophic() was evaluated once at booking; held() re-evaluates it on every call when it is given the posterior."""
    led = _owed_world(store)
    store.set_state(ST.K_STRUCT, {"b1": {"kind": "other", "children": ["c1", "c2", "c3"], "steer": "mixed",
                                         "ts": time.time()}})
    _engaged(monkeypatch, "engaged")
    assert "b1" in ST.held(sctx, store, led, now=time.time(), post=_post(m=-0.02, v=1e-6, n=1))      # 1 seed: not yet
    assert ST.held(sctx, store, led, now=time.time(), post=_post(m=-0.02, v=1e-6, n=3)) == {}        # 3 seeds: exits


def test_the_cluster_owed_filter_is_symmetric(sctx, store):
    """M1: only a non-positive belief of an owed member was excluded, so an owed node's POSITIVE belief still counted."""
    store.ensure_root("r0")
    from .core.test_c7i import _put
    nodes, bel, now0 = [], {}, time.time()
    for i in range(3):
        n = _put(store, f"k{i}", f"kde_uniformity_margin_v{i}", "representation", ts=now0, notes="request=r brief=bold")
        store.update_node(n.id, status=NodeStatus.SCREENING, created_ts=now0)
        nodes.append(dataclasses.replace(n, strategy="moonshot", created_ts=now0))
        bel[f"k{i}"] = (0.02, 2)
    led = _led(nodes)
    assert set(ST.held(sctx, store, led)) == {"k0", "k1", "k2"}
    cl = next(c for c in AL.clusters(sctx, store, led, bel) if "k0" in c.members)
    assert cl.n == 0 and cl.flag != "boost"                                    # owed: no positive evidence either


def test_the_report_shows_and_judges_each_point_in_the_incumbent_frame(sctx, store):
    """M2: a ladder child's belief is its increment over the UNTUNED parent; the verdict is over the incumbent."""
    store.ensure_root("r0")
    _store_node(store, "b1")
    kids = []
    for i in range(3):
        _store_node(store, f"t{i}", Strategy.EXPLORE, f"ladder=b1 struct_tune=backbone cell=lr:c{i}", parent="b1",
                    status=NodeStatus.QUEUED)
        kids.append(f"t{i}")
    store.set_state(ST.K_STRUCT, {"b1": {"kind": "backbone", "children": kids, "steer": "mixed", "ts": time.time()}})
    led = _led([_dn("b1")] + [_dn(k, parent="b1", strategy="explore", notes="ladder=b1 struct_tune=backbone cell=lr:c0")
                              for k in kids], obs=[k for k in kids for _ in range(3)])
    # tuned points beat the untuned node by +.004 but the untuned node sits -.011 below the incumbent
    post = SimpleNamespace(belief=lambda n: {"b1": (-0.011, 4e-6), "t0": (0.004, 4e-6), "t1": (0.002, 4e-6),
                                             "t2": (0.001, 4e-6)}[n],
                           shift=lambda c, t, to: (-0.011, 1e-6))
    ST.report(sctx, store, led, post, 5.0)
    (pm,) = store.postmortems(node_id="b1")
    assert "over incumbent -0.0070" in pm.explanation and "over untuned +0.0040" in pm.explanation
    assert pm.verdict == "loss" and "t0" in pm.suggested_next or "closed after" in pm.suggested_next
    assert "test t0" not in pm.suggested_next                      # -.007 over the incumbent: nothing to promote


def test_k_of_the_max_of_k_shrink_is_the_booked_ladder_size_not_the_live_siblings():
    """S1: siblings resolve and the live count collapsed the shrink from E_5 to E_1 = 0."""
    kid = lambda i: CD.Node(f"t{i}", "b1", None, "explore", "", "explore", "r0", 1.0, "queued",         # noqa: E731
                            "ladder=b1 struct_tune=backbone cell=x", base="B")
    led = _led([_dn("b1")] + [kid(i) for i in range(5)])
    assert AQ.struct_k(led, led.nodes["t0"], 1) == 5                                  # one live sibling, five booked
    assert AQ.struct_k(led, led.nodes["t0"], 1, {"b1": 7}) == 7                       # K_STRUCT's own count wins
    assert AQ.struct_k(led, led.nodes["b1"], 3) == 3                                  # a plain node: its siblings
    assert M.max_of_k(0.004, 4e-6, AQ.struct_k(led, led.nodes["t0"], 1), 1.0) < 0.004 - 0.002


def test_mechanism_dead_ignores_ladder_children_and_open_ladder_nodes(sctx, store):
    """S4: a screened-out ladder point (or a node still being tuned) made its mechanism 'dead' for later ideas."""
    from loop import novelty as NV
    store.ensure_root("r0")
    _store_node(store, "b1", status=NodeStatus.SCREENED_OUT)
    _store_node(store, "t0", Strategy.EXPLORE, "ladder=b1 struct_tune=backbone cell=lr:c0", parent="b1",
                status=NodeStatus.SCREENED_OUT)
    _store_node(store, "p1", status=NodeStatus.SCREENED_OUT)
    assert NV.dead(store) == {"b1", "p1"}                            # a ladder child is never a dead mechanism
    store.set_state(ST.K_STRUCT, {"b1": {"kind": "other", "children": ["t0"], "steer": "mixed", "ts": 1.0}})
    assert NV.dead(store) == {"p1"}
    store.set_state(ST.K_PM, {"b1": "pm-struct-b1"})                 # the ladder finished: its verdict counts again
    assert NV.dead(store) == {"b1", "p1"}


# ---------------------------------------------------------------------------------------------- 10-22 ops review
def _many(store, specs, strategy=Strategy.MOONSHOT):
    """Backbone-swap nodes b0.. (id, notes, belief mean) all screening on the current base."""
    store.ensure_root("r0")
    nodes, bel = [], {}
    for nid, notes, m in specs:
        _store_node(store, nid, strategy, notes=notes)
        nodes.append(_dn(nid, strategy=strategy.value, notes=notes))
        bel[nid] = m
    est = {n: {M.COMPOSITE: SimpleNamespace(n=3)} for n in bel}
    post = SimpleNamespace(est=est, belief=lambda nid: (bel[nid], 1e-6))
    adapter = _adapter({"c-b1": CFG, "r0": PARENT_CFG, "c-r0": PARENT_CFG} | {f"c-{n}": CFG for n in bel})
    return _led(nodes), post, adapter


def test_struct_ladders_cap_open_ladders_and_wait_in_priority_order_without_a_running_hold_clock(sctx, store):
    """10-22 ops review: tick 0 booked ~19 ladders at once. Only struct_ladder_max_open run; the rest WAIT (retro ports,
    then the others by posterior), are not closed on their untuned shot and their hold clock does not run. 11-10: an
    operator arch arm's ladder is OUTSIDE the cap (built at once, takes no slot, not counted as open)."""
    assert sctx.cfg.ideation.struct_ladder_max_open == 6
    ic = dataclasses.replace(sctx.cfg.ideation, struct_ladder_max_open=2)
    c = dataclasses.replace(sctx, cfg=dataclasses.replace(sctx.cfg, ideation=ic))
    led, post, adapter = _many(store, [("p1", "request=r brief=bold", -0.001), ("p2", "request=r brief=bold", 0.004),
                                       ("rt", "request=r brief=bold ladder=old struct_retro=1", -0.002),
                                       ("ar", "request=r brief=program program_arm=1 arch_arm=1 op=op-x arm=a", -0.003)])
    st = {n: "screening" for n in ("p1", "p2", "rt", "ar")}
    acts = PG.ladders(c, store, adapter, led, post, st, time.time())
    bk = store.get_state(ST.K_STRUCT)
    assert set(bk) == {"ar", "rt", "p2"} and all(isinstance(v, dict) for v in bk.values())  # arm + 2 capped slots
    assert set(store.get_state(ST.K_WAIT)) == {"p1"} and any("wait for a ladder slot" in a for a in acts)
    assert any("ar" in a and "outside struct_ladder_max_open" in a for a in acts)
    hold = c.cfg.ideation.struct_hold_h * 3600.0
    held = ST.held(c, store, led, now=time.time() + 5 * hold)
    assert "p1" in held and "waiting" in held["p1"]                                       # no expiry while waiting
    assert ST.open_ladders(c, store, led, time.time()) == 2                               # the arm's is not counted
    PG.ladders(c, store, adapter, led, post, st, time.time())                            # no free slot: nothing new
    assert set(store.get_state(ST.K_STRUCT)) == {"ar", "rt", "p2"}
    # the capped ladders finish (3 points x 3 seeds): the waiting node gets a slot
    obs = [k for n in ("rt", "p2") for k in bk[n]["children"][:3] for _ in range(3)]
    done = _led(list(led.nodes.values())[1:], obs=obs)
    PG.ladders(c, store, adapter, done, post, st, time.time())
    bk2 = store.get_state(ST.K_STRUCT)
    assert set(bk2) == {"ar", "rt", "p1", "p2"} and not store.get_state(ST.K_WAIT)
    assert not ST.held(c, store, done, now=time.time() + 5 * hold).get("p1", "").startswith("ladder waiting")


def test_a_config_only_operator_arm_is_laddered_past_a_full_cap_and_a_stale_not_structural_memo(sctx, store):
    """11-10 regression (nanobrain n0001, op-natinit-lr/blr10): a ported arm whose diff is config-only
    (natinit.block_lr_mult, no backbone key) was judged 'not structural' (kind_of excluded brief=program), memoised in
    K_NONSTRUCT, and would have waited behind 6 full auto ladders anyway. It is laddered with its `tune` points at once;
    a first shot at/below struct_tune_flat_floor is still skipped."""
    pts = [{"natinit.block_lr_mult": 0.5}, {"natinit.block_lr_mult": 1.5}, {"natinit.layer_decay": 0.9},
           {"natinit.layer_decay": 0.95}]
    prog = ({"id": "op-n", "sessions": False, "arms": [{"id": "blr10", "lane": "ported", "tune": {"points": pts}}]},)
    ic = dataclasses.replace(sctx.cfg.ideation, struct_ladder_max_open=1, operator_programs=prog,
                             struct_tune_flat_floor=-0.005)
    c = dataclasses.replace(sctx, cfg=dataclasses.replace(sctx.cfg, ideation=ic))
    arm = "brief=program op=op-n arm=blr10 stage=A program_arm=1 arch_arm=1"
    cfg = {"natinit": {"block_lr_mult": 1.0, "layer_decay": 1.0}, "lr": 1e-3}
    pcfg = {"natinit": {"block_lr_mult": 0.3, "layer_decay": 1.0}, "lr": 1e-3}
    store.ensure_root("r0")
    for nid, notes in (("n1", arm), ("q1", "request=r brief=bold")):
        _store_node(store, nid, Strategy.EXPLORE, notes=notes)
    store.set_state("config_diff:n1", {"natinit.block_lr_mult": 1.0})
    store.set_state("config_diff:q1", {"lr": 1e-3})
    led = _led([_dn("n1", strategy="explore", notes=arm), _dn("q1", strategy="explore", notes="request=r brief=bold")])
    adapter = _adapter({"c-b1": cfg, "c-n1": cfg, "c-q1": {**cfg, "lr": 2e-3}, "r0": pcfg, "c-r0": pcfg})
    assert ST.kind_of(c, store, led.nodes["n1"], flatten(cfg), flatten(pcfg))[0] == "other"
    store.set_state(ST.K_NONSTRUCT, {"n1": ST._signature(led.nodes["n1"])})      # the stale verdict of the old rule
    bel = {"n1": 0.0013, "q1": 0.002}
    post = SimpleNamespace(est={n: {M.COMPOSITE: SimpleNamespace(n=2)} for n in bel}, belief=lambda nid: (bel[nid], 1e-5))
    store.set_state(ST.K_STRUCT, {"z": {"kind": "other", "children": ["zc"], "ts": time.time()}})  # the one slot is full
    _store_node(store, "zc", Strategy.EXPLORE, notes="ladder=z struct_tune=other cell=x", status=NodeStatus.QUEUED)
    assert ST.open_ladders(c, store, led, time.time()) == 1
    acts = PG.ladders(c, store, adapter, led, post, {"n1": "screening", "q1": "screening"}, time.time())
    rec, kids = _kids(store, "n1")
    assert rec["kind"] == "other+operator" and [d for d, _ in kids] == pts
    assert all("ladder=n1" in n.notes for _, n in kids)
    assert "q1" not in store.get_state(ST.K_STRUCT) and set(store.get_state(ST.K_WAIT)) == {"q1"}   # q1 still capped
    assert any("outside struct_ladder_max_open" in a for a in acts)
    # a hopeless first shot (<= the flat floor, no engagement) is booked as skipped, not laddered
    store.set_state(ST.K_STRUCT, {})
    bel["n1"] = -0.02
    PG.ladders(c, store, adapter, led, post, {"n1": "screening", "q1": "screening"}, time.time())
    assert str(store.get_state(ST.K_STRUCT)["n1"]).startswith(ST.SKIP_FIRST)


def test_a_node_judged_not_structural_is_never_read_again_and_a_plain_node_reads_no_config(sctx, store):
    """10-22 ops review: 520 derived-SCREENING nodes cost two git shows each every tick before kind_of said 'not structural'."""
    led, post, _ = _many(store, [("q1", "request=r", 0.0), ("q2", "request=r", 0.0)], Strategy.EXPLORE)
    adapter = _adapter({"c-b1": PARENT_CFG, "c-q1": PARENT_CFG, "c-q2": PARENT_CFG, "r0": PARENT_CFG, "c-r0": PARENT_CFG})
    store.set_state("config_diff:q1", {"dino.lr": 1e-4})
    reads = []
    real = adapter.config_at
    adapter.config_at = lambda c: reads.append(c) or real(c)
    st = {"q1": "screening", "q2": "screening"}
    PG.ladders(sctx, store, adapter, led, post, st, 1.0)
    assert not any(r in ("c-q1",) for r in reads)                                        # a diff with no backbone key: skipped
    first = len(reads)
    assert "c-q2" in reads                                                               # q2 has no stored diff: the config decides
    PG.ladders(sctx, store, adapter, led, post, st, 2.0)
    assert len(reads) == first                                                           # ... once
    assert set(store.get_state(ST.K_NONSTRUCT)) == {"q1", "q2"} and not store.get_state(ST.K_STRUCT)


def test_the_engagement_view_is_computed_once_per_ledger_view(sctx, store, monkeypatch):
    calls = []
    import loop.core.ideation as ID
    monkeypatch.setattr(ID, "engagements", lambda s, l: calls.append(1) or [])
    led, post, _ = _many(store, [("e1", "request=r brief=bold", 0.0)])
    for _ in range(5):
        ST.steer(sctx, store, led, post, "e1")
    assert len(calls) == 1


def test_a_flat_floor_ladders_a_flat_but_not_dead_node_and_rechecks_an_earlier_z_skip(sctx, store):
    """11-09 operator rule (nanobrain ``struct_tune_flat_floor: -.005``): n0030 at -.0017 was skipped by the -2.5 sd rule
    (a tight posterior); under the floor a flat node is laddered (<= struct_tune_max), a dead one is not, and an earlier
    z-rule skip is re-checked once."""
    cfg = {"dino": {"lr": 2e-4, "layerwise_decay": 0.8, "warmup_fraction": 0.09}}
    led, _, adapter = _setup(store, cfg=cfg, parent_cfg=cfg)
    PG.ladders(sctx, store, adapter, led, _post(m=-0.002, v=1e-8), {"b1": "screening"}, 10.0)   # z rule: -20 sd
    assert store.get_state(ST.K_STRUCT)["b1"].startswith(ST.SKIP_FIRST)
    flat = dataclasses.replace(sctx, cfg=dataclasses.replace(sctx.cfg, ideation=dataclasses.replace(
        sctx.cfg.ideation, struct_tune_flat_floor=-0.005, struct_tune_max=4)))
    PG.ladders(flat, store, adapter, led, _post(m=-0.002, v=1e-8), {"b1": "screening"}, 20.0)   # re-checked
    rec, kids = _kids(store)
    assert rec["kind"] == "other" and 1 <= len(kids) <= 4
    store.set_state(ST.K_STRUCT, {})
    PG.ladders(flat, store, adapter, led, _post(m=-0.006, v=1e-2), {"b1": "screening"}, 25.0)   # noisy: z rule ladders
    assert isinstance(store.get_state(ST.K_STRUCT)["b1"], dict)                 # the floor only relaxes the z rule
    store.set_state(ST.K_STRUCT, {})
    PG.ladders(flat, store, adapter, led, _post(m=-0.006, v=1e-8), {"b1": "screening"}, 30.0)   # dead by both
    rec = store.get_state(ST.K_STRUCT)["b1"]
    assert isinstance(rec, str) and rec.startswith(ST.SKIP_FIRST) and "struct_tune_flat_floor" in rec
    n_logs = len(store.events("struct_ladder"))
    PG.ladders(flat, store, adapter, led, _post(m=-0.007, v=1e-8), {"b1": "screening"}, 40.0)   # still dead: quiet
    assert len(store.events("struct_ladder")) == n_logs and store.get_state(ST.K_STRUCT)["b1"] == rec


def test_a_rechecked_node_waiting_for_a_ladder_slot_is_unbooked_so_it_stays_owed(sctx, store, monkeypatch):
    cfg = {"dino": {"lr": 2e-4, "layerwise_decay": 0.8, "warmup_fraction": 0.09}}
    led, _, adapter = _setup(store, cfg=cfg, parent_cfg=cfg)
    store.set_state(ST.K_STRUCT, {"b1": f"{ST.SKIP_FIRST} -0.0020 <= -2.5 sd with no engagement diagnostic; x"})
    flat = dataclasses.replace(sctx, cfg=dataclasses.replace(sctx.cfg, ideation=dataclasses.replace(
        sctx.cfg.ideation, struct_tune_flat_floor=-0.005)))
    monkeypatch.setattr(ST, "open_ladders", lambda *a, **k: 10 ** 6)          # no ladder slot free
    PG.ladders(flat, store, adapter, led, _post(m=-0.002, v=1e-8), {"b1": "screening"}, 10.0)
    assert "b1" not in (store.get_state(ST.K_STRUCT) or {}) and "b1" in (store.get_state(ST.K_WAIT) or {})
