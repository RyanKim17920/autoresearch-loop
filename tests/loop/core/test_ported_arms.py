"""10-22 operator ARCHITECTURE program: ``lane: ported`` arms (a prebuilt code port, no LLM session) -- built once as a
QUEUED node on the incumbent (code files from ``code_ref`` + config keys set in the incumbent's YAML), refused on a missing
file / a lock violation / a drifted base, tuned by an explicit multi-key ladder before they can close, and the retro of a
closed current-epoch moonshot."""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from loop.contracts import ROOT_NODE_ID, NodeStatus, Strategy
from loop.core import ideation as ID
from loop.core import program as PG
from loop.core import struct_tune as ST

from ..test_struct_tune import (CFG, PARENT_CFG, _adapter, _dn, _kids, _led, _post, _setup, _store_node,  # noqa: F401
                                real_ideation, sctx)
from .test_fnf import env  # noqa: F401  (the fixture)

CONFIG = """data:
  mean: [0.485, 0.456, 0.406]   # imagenet
  std: [0.229, 0.224, 0.225]
model:
  type: robust_norm_dinov2_vits14_reg
train:
  local_size: 128
dino:
  lr: 0.0002344
"""
BASE, REF = "base-sha", "ref-sha"


class Ad:
    config_rel = "configs/main.yaml"

    def __init__(self, files=None, locked=()):
        self.files = {("inc", "model.py"): "M0", ("inc", "train.py"): "T0", (BASE, "model.py"): "M0",
                      (BASE, "train.py"): "T0", (REF, "model.py"): "M1", (REF, "train.py"): "T1",
                      ("inc", self.config_rel): CONFIG, **(files or {})}
        self.built, self.locked = [], list(locked)

    def file_at(self, commit, rel):
        return self.files.get((commit, rel))

    def commit_files(self, parent, files, *, message, branch=""):
        self.built.append((parent, dict(files)))
        return f"sha{len(self.built)}"

    def lock_violations(self, commit):
        return list(self.locked)


def _arm(aid="eva", **kw):
    return {"id": aid, "lane": "ported", "family": "architecture", "code_ref": REF, "code_base": BASE,
            "code_files": ["model.py", "train.py"], "stage": "B", "config": {
                "model.type": "timm:eva02_small_patch14_224.mim_in22k", "train.local_size": 126,
                "data.mean": [0.48145466, 0.4578275, 0.40821073]}, **kw}


def _icfg(ctx, **kw):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(ctx.cfg.ideation, **kw)))


def _inc(store):
    store.ensure_root("r0")
    return store.update_node(ROOT_NODE_ID, commit="inc")


def test_a_ported_arm_builds_one_queued_node_with_its_files_config_and_notes_once(ctx, store):
    inc, ad = _inc(store), Ad()
    c = _icfg(ctx, operator_programs=({"id": "op-a", "sessions": False, "arms": [_arm()]},))
    acts = PG.arms(c, store, ad, inc, 1.0)
    nid = store.get_state(PG.K_ARMS)["op-a"]["eva"]
    n = store.get_node(nid)
    assert n.status is NodeStatus.QUEUED and n.parent_id == inc.id and n.family == "architecture"
    assert n.notes == "brief=program op=op-a arm=eva stage=B program_arm=1 arch_arm=1" and acts and "ported node" in acts[0]
    assert store.get_state(f"config_diff:{nid}") == _arm()["config"]
    (parent, files), = ad.built
    assert parent == "inc" and files["model.py"] == "M1" and files["train.py"] == "T1"
    got = yaml.safe_load(files[ad.config_rel])                      # dotted key, flow list, comment kept, `timm:` string
    assert got["model"]["type"] == "timm:eva02_small_patch14_224.mim_in22k" and got["train"]["local_size"] == 126
    assert got["data"]["mean"] == [0.48145466, 0.4578275, 0.40821073] and got["data"]["std"] == [0.229, 0.224, 0.225]
    assert "# imagenet" in files[ad.config_rel]                                           # comments kept
    PG.arms(c, store, ad, inc, 2.0)
    PG.arms(c, store, ad, inc, 3.0)
    assert len(ad.built) == 1 and len(store.nodes()) == 2                                 # idempotent across ticks


@pytest.mark.parametrize("what,ad,arm,why", [
    ("missing file", Ad(), {"code_files": ["model.py", "aug.py"]}, "aug.py missing at code_ref"),
    ("lock violation", Ad(locked=["train.py::loss"]), {}, "lock violation"),
    ("base drift", Ad(files={("inc", "train.py"): "T-moved"}), {}, "code_ref base drifted"),
    ("bad config key", Ad(), {"config": {"dino.nope": 1}}, "config:")])
def test_a_ported_arm_is_refused_with_a_booked_reason(ctx, store, what, ad, arm, why):
    inc = _inc(store)
    c = _icfg(ctx, operator_programs=({"id": "op-a", "arms": [_arm(**arm)]},))
    PG.arms(c, store, ad, inc, 1.0)
    rec = store.get_state(PG.K_ARMS)["op-a"]["eva"]
    assert rec.startswith("-:") and why in rec and len(store.nodes()) == 1, rec
    PG.arms(c, store, ad, inc, 2.0)
    assert store.get_state(PG.K_ARMS)["op-a"]["eva"] == rec                              # booked: never rebuilt


def test_without_code_base_the_drift_check_is_off(ctx, store):
    inc = _inc(store)
    ad = Ad(files={("inc", "train.py"): "T-moved"})
    arm = _arm()
    del arm["code_base"]
    c = _icfg(ctx, operator_programs=({"id": "op-a", "arms": [arm]},))
    PG.arms(c, store, ad, inc, 1.0)
    assert not store.get_state(PG.K_ARMS)["op-a"]["eva"].startswith("-") and len(ad.built) == 1


def test_arm_problem_validates_ported_arms():
    assert PG.arm_problem(_arm()) == ""
    for k in ("code_ref", "code_files", "config"):
        bad = {x: v for x, v in _arm().items() if x != k}
        assert k in PG.arm_problem(bad)
    assert "code_files" in PG.arm_problem(_arm(code_files="model.py"))


def test_after_makes_a_ported_arm_wait_for_the_booking_of_those_arms(ctx, store):
    inc, ad = _inc(store), Ad()
    c = _icfg(ctx, operator_programs=({"id": "op-a", "arms": [_arm("second", after=["first"]), _arm("first")]},))
    PG.arms(c, store, ad, inc, 1.0)
    book = store.get_state(PG.K_ARMS)["op-a"]
    assert set(book) == {"first"}                                                         # `second` waits for `first`
    PG.arms(c, store, ad, inc, 2.0)
    assert set(store.get_state(PG.K_ARMS)["op-a"]) == {"first", "second"}


def test_after_done_waits_for_the_other_programs_arm_nodes_to_be_terminal_or_screened_on_three_seeds(ctx, store):
    """10-22 ops review: `after` waits only for the BOOKING (one tick), so it cannot order two programs' builds."""
    from loop.contracts import Run, RunStatus, Stage
    inc, ad = _inc(store), Ad()
    c = _icfg(ctx, operator_programs=(
        {"id": "op-core", "arms": [_arm("core1")]},
        {"id": "op-x", "arms": [_arm("late", after_done=["core1", "nope"]), _arm("late2", after_done=["core1"])]}))
    PG.arms(c, store, ad, inc, 1.0)
    book = store.get_state(PG.K_ARMS)
    assert set(book["op-core"]) == {"core1"} and "op-x" not in book                       # booked, but not done
    nid = book["op-core"]["core1"]
    for sd in (1, 2):
        store.put_run(Run(id=f"{nid}:screen:{sd}", task_id=store.task_id, node_id=nid, seed=sd, stage=Stage.SCREEN,
                          status=RunStatus.DONE, submitted_ts=1.0))
    PG.arms(c, store, ad, inc, 2.0)
    assert "op-x" not in store.get_state(PG.K_ARMS)                                       # 2 finished seeds: still waiting
    store.put_run(Run(id=f"{nid}:screen:3", task_id=store.task_id, node_id=nid, seed=3, stage=Stage.SCREEN,
                      status=RunStatus.DONE, submitted_ts=1.0))
    PG.arms(c, store, ad, inc, 3.0)
    assert set(store.get_state(PG.K_ARMS)["op-x"]) == {"late2"}                           # `nope` is not booked anywhere
    assert PG.after_done_ready(store, store.get_state(PG.K_ARMS), ["core1"])
    store.update_node(nid, status=NodeStatus.SCREENED_OUT)
    assert PG.after_done_ready(store, {"p": {"a": nid}}, ["a"])                           # terminal also counts
    assert PG.after_done_ready(store, {"p": {"a": "-: refused"}}, ["a"])                  # given up counts as done


def test_the_yaml_arch_explore_arms_wait_for_the_core_s_arms(real_ideation):
    from tests.loop.conftest import example_programs
    progs = {p["id"]: p for p in example_programs()}
    core, expl = progs["op-arch-core"], progs["op-arch-explore"]
    assert core["budget_gpu_h"] == 180 and expl["budget_gpu_h"] == 90
    assert {a["id"] for a in expl["arms"]} == {"eva02_s14", "eva02_b14", "siglip2_b16"}
    ids = {a["id"] for a in core["arms"]}
    for a in expl["arms"]:
        assert set(a["after_done"]) == {"v3_s16", "v3_sp16", "port_id"} <= ids


# ---------------------------------------------------------------------------------------------- the tuning ladder
POINTS = [{"dino.lr": 1.172e-4}, {"dino.lr": 4.688e-4}, {"dino.layerwise_decay": 0.65},
          {"dino.lr": 1.172e-4, "dino.layerwise_decay": 0.65}, {"dino.warmup_fraction": 0.1818}]
ARCH = "brief=program op=op-a arm=eva stage=B program_arm=1 arch_arm=1"


def test_explicit_points_give_one_multi_key_child_each_typed_like_the_config():
    cfg = {"dino.lr": 2.344e-4, "dino.layerwise_decay": 0.8, "dino.warmup_fraction": 0.0909, "dino.n": 4}
    got = ST.explicit_points({"points": POINTS + [{"dino.n": 4}, {"dino.lr": 1e-4, "dino.absent": 1.0}, {}]}, cfg, 9)
    assert [d for d, _ in got] == POINTS                        # unchanged / unsettable points are dropped
    assert got[3][1] == "lr=0.0001172+layerwise_decay=0.65"
    assert len(ST.explicit_points({"points": POINTS}, cfg, 2)) == 2


def _arch_program(**kw):
    return ({"id": "op-a", "sessions": False, "arms": [{**_arm("eva"), "tune": {"points": POINTS}}]},)


def test_an_arch_arm_is_held_until_tuned_then_gets_its_five_explicit_points(sctx, store):
    c = _icfg(sctx, operator_programs=_arch_program())
    store.ensure_root("r0")
    _store_node(store, "b1", notes=ARCH, strategy=Strategy.EXPLORE)
    led = _led([_dn("b1", strategy="explore", notes=ARCH)])
    assert ST.held(c, store, led, 11.0)["b1"] == "ladder owed"
    cfg = {"model": {"type": "timm:x"}, "dino": {"lr": 2.344e-4, "layerwise_decay": 0.8, "warmup_fraction": 0.0909}}
    adapter = _adapter({"c-b1": cfg, "r0": PARENT_CFG, "c-r0": PARENT_CFG})
    PG.ladders(c, store, adapter, led, _post(m=-0.03, v=1e-6), {"b1": "screening"}, 10.0)
    rec, kids = _kids(store)
    assert rec["kind"] == "backbone+operator" and [d for d, _ in kids] == POINTS
    assert all("ladder=b1" in n.notes and n.status is NodeStatus.QUEUED for _, n in kids)
    assert "0/" in ST.held(c, store, led, 11.0)["b1"]                  # held until its points finish
    PG.ladders(c, store, adapter, led, _post(m=-0.03, v=1e-6), {"b1": "screening"}, 20.0)
    assert len(store.get_state(ST.K_STRUCT)["b1"]["children"]) == 5          # once


def test_retro_spec_reads_the_arms_tune_dict_and_nothing_else(sctx, store):
    c = _icfg(sctx, operator_programs=_arch_program())
    node = _dn("b1", strategy="explore", notes=ARCH)
    assert ST.retro_spec(c, store, node) == {"points": POINTS}
    assert ST.retro_spec(c, store, _dn("b1", strategy="explore", notes=ARCH.replace("arm=eva", "arm=zzz"))) == {}
    assert ST.retro_spec(c, store, _dn("b1", notes="request=r brief=bold")) == {}


# ---------------------------------------------------------------------------------------------- sessions: false
def test_a_program_with_sessions_false_is_never_briefed(env):  # noqa: F811
    ctx, store, led, ide = env
    prog = {"id": "op-x", "brief": "do it", "arms": []}
    ide.ctx = _icfg(ctx, operator_programs=(prog,))
    assert ide._op_next() is not None and ide._op_find("op-x")["id"] == "op-x"
    ide.ctx = _icfg(ctx, operator_programs=({**prog, "sessions": False},))
    assert ide._op_next() is None and ide._op_find("op-x") is None


# ---------------------------------------------------------------------------------------------- C7: retro in place
def test_a_closed_current_epoch_moonshot_is_retro_tuned_in_place_without_a_port(sctx, store):
    c = _icfg(sctx, struct_retro_nodes=("b1",))
    led, post, adapter = _setup(store)
    acts = ST.retro(c, store, adapter, led, 5.0)
    assert any("tuned in place" in a for a in acts)
    rec = store.get_state(ST.K_RETRO)["b1"]
    assert rec["port"] == "b1" and ST.RETRO_TAG in store.get_node("b1").notes and "ladder=b1" in store.get_node("b1").notes
    assert len(store.nodes()) == 2                                                    # root + b1: nothing ported
    led = _led([_dn("b1", notes=store.get_node("b1").notes)])                         # the next tick's ledger carries the tag
    assert ST.held(c, store, led, 6.0)["b1"] == "ladder owed"
    PG.ladders(c, store, adapter, led, post, {"b1": "screened_out"}, 10.0)           # closed, yet laddered (retro tag)
    assert store.get_state(ST.K_STRUCT)["b1"]["children"]
    assert ST.retro(c, store, adapter, led, 6.0) == []                                # once


# ---------------------------------------------------------------------------------------------- C6: timm weights source
ALLOW = r"^(?:hf-hub:)?(?:timm/)?(?:eva02_(?:small|base)_patch14_224\.mim_in22k|vit_base_patch16_siglip_224\.v2_webli)$"
ALLOW_TYPES = (r"^(?:robust_norm_)?dinov3_vit(?:s16(?:plus)?|sp16|splus16|s_plus16|b16)$",
               r"^timm:(?:eva02_(?:small|base)_patch14_224\.mim_in22k|vit_base_patch16_siglip_224\.v2_webli)$")


def test_a_timm_trunk_reports_a_timm_source_judged_against_the_allowlist():
    import re

    from loop.adapters import initcompat as IC
    assert 'getattr(model, "timm_name", "")' in IC.SCRIPT and '"timm/"' in IC.SCRIPT
    for ok in ("timm/eva02_small_patch14_224.mim_in22k", "timm/vit_base_patch16_siglip_224.v2_webli"):
        assert IC.allowed(ok, (ALLOW,))
    for bad in ("timm/eva02_large_patch14_224.mim_in22k", "timm/vit_base_patch16_224.augreg"):
        assert not IC.allowed(bad, (ALLOW,))
    assert any(re.search(p, "timm:eva02_base_patch14_224.mim_in22k") for p in ALLOW_TYPES)
    assert any(re.search(p, "dinov3_vits16plus") for p in ALLOW_TYPES)
    assert not any(re.search(p, "timm:vit_large_patch14_clip_224") for p in ALLOW_TYPES)
