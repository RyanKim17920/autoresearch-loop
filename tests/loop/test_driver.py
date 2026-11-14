"""Driver, seeds and calibrate tests (docs/loop/DESIGN.md sections 5-7).

Run: uv run pytest tests/loop -q --confcutdir=tests/loop
Everything is offline: SimAdapter + FakeLLMClient + an inline executor.
"""

from __future__ import annotations

import dataclasses
import json
import re
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from loop import remint as RM
from loop import seeds as S
from loop.calibrate import calibrate, calibration_seeds, compute, rewrite_yaml
from loop.contracts import (PANEL, ROOT_NODE_ID, HypothesisKind, Node, NodePosterior, NodeStatus,
                            Posterior, Run, RunStatus, Stage, Strategy, VoidType)
from loop.spec import LoopConfig, PromotionCfg, ReviewCfg, SeedPlan
from loop.driver import Driver
from tests.loop.conftest import InlineExecutor
from loop.spec import load_context
from loop.store import SqliteStore

REPO = Path(__file__).resolve().parents[2]
FAKE = REPO / "tasks" / "fake_loop.yaml"


# ---------------------------------------------------------------------------
# seeds.py — pure transition table
# ---------------------------------------------------------------------------

def _cfg(**kw: Any) -> LoopConfig:
    return LoopConfig(execution_adapter="x:Y", axes=("a",), weights={"a": 1.0},
                      panel_sd=0.004, axis_sd={"a": 0.004}, **kw)


def _node(status: NodeStatus, parent: str = ROOT_NODE_ID, nid: str = "n1") -> Node:
    return Node(id=nid, task_id="t", commit="c", parent_id=parent,
                strategy=Strategy.EXPLORE, operator=None, family="f", mechanism_tag="m",
                diff_hash="h", status=status)


def _run(node: str, stage: Stage, seed: int, status: RunStatus = RunStatus.DONE,
         void: VoidType | None = None, attempt: int = 0) -> Run:
    return Run(id=Run.make_id(node, stage, seed), task_id="t", node_id=node, seed=seed,
               stage=stage, status=status, void_type=void, scores={"a": 0.5}, attempt=attempt)


def _post(p_pos: float = 0.9, p_inc: float = 0.9, inc: float = 0.01, vetoed: bool = False
          ) -> NodePosterior:
    q = Posterior(inc, 0.002, 2)
    return NodePosterior(node_id="n1", increment=q, utility=q, axes={"a": q},
                         p_positive=p_pos, p_beats_incumbent=p_inc, vetoed=vetoed,
                         veto_axes=("a",) if vetoed else ())


def _roots(seeds: list[int]) -> list[Run]:
    return [_run(ROOT_NODE_ID, Stage.BASELINE, s) for s in seeds]


def _step(node: Node, own: list[Run], roots: list[Run], post: NodePosterior | None,
          cfg: LoopConfig | None = None, parent: list[Run] | None = None,
          confirm: tuple[float, int] | None = None, frozen: tuple[float, float] | None = None) -> S.Decision:
    control = roots if node.parent_id == ROOT_NODE_ID else (parent or [])
    return S.step(node, own_runs=own, control_runs=control, post=post, cfg=cfg or _cfg(),
                  sigma_panel=0.004, infra_retry=2, confirm=confirm, frozen=frozen)


def test_screen_needs_root_first_and_own_runs():
    d = _step(_node(NodeStatus.SCREENING), [], [], None)
    ids = {n.run_id for n in d.needs}
    assert ids == {"n1:screen:11", "n1:screen:12", "root:baseline:11", "root:baseline:12"}
    assert d.status is None


def test_screen_waits_for_root_even_when_own_runs_done():
    own = [_run("n1", Stage.SCREEN, 11), _run("n1", Stage.SCREEN, 12)]
    roots = _roots([11]) + [_run(ROOT_NODE_ID, Stage.BASELINE, 12, RunStatus.RUNNING)]
    assert _step(_node(NodeStatus.SCREENING), own, roots, _post()).status is None


@pytest.mark.parametrize("post,expected", [
    (_post(p_pos=0.9), NodeStatus.PROMOTING),
    (_post(p_pos=0.5), NodeStatus.SCREENED_OUT),
    (_post(p_pos=0.99, vetoed=True), NodeStatus.VETOED),
])
def test_screen_outcomes(post, expected):
    own = [_run("n1", Stage.SCREEN, 11), _run("n1", Stage.SCREEN, 12)]
    assert _step(_node(NodeStatus.SCREENING), own, _roots([11, 12]), post).status is expected


def test_no_promote_stage_screen_seeds_gate_straight_to_confirm():
    """seeds.promote_stage false: the 2 screen seeds' posterior is the promotion gate (screen gate AND P(U>U_inc)),
    a passing node confirms on 3 fresh seeds (5 in all); a node already PROMOTING keeps its promote seed + 2."""
    cfg = _cfg(seeds=SeedPlan(promote_stage=False, max_seeds_per_node=5),
               promotion=PromotionCfg(confirm_max_seeds=3, alpha_confirm=0.2))
    own = [_run("n1", Stage.SCREEN, 11), _run("n1", Stage.SCREEN, 12)]
    d = _step(_node(NodeStatus.SCREENING), own, _roots([11, 12]), _post(), cfg)
    assert d.status is NodeStatus.CONFIRMING and "P(U>U_inc)" in d.reason
    assert d.fields["confirm_alt"] == pytest.approx(0.004 * math.sqrt(2 * math.log(5) / 3))
    assert _step(_node(NodeStatus.SCREENING), own, _roots([11, 12]), _post(p_inc=0.2), cfg).status \
        is NodeStatus.SCREENED_OUT
    assert _step(_node(NodeStatus.SCREENING), own, _roots([11, 12]), _post(p_pos=0.5), cfg).status \
        is NodeStatus.SCREENED_OUT
    promo = own + [_run("n1", Stage.PROMOTE, 13), _run("n1", Stage.PROMOTE, 14)]
    assert S.planned_confirm(cfg, own) == 3 and S.planned_confirm(cfg, own + promo[2:3]) == 2


def test_screen_futility_stops_after_the_first_seed():
    own = [_run("n1", Stage.SCREEN, 11), _run("n1", Stage.SCREEN, 12, RunStatus.RUNNING)]
    d = _step(_node(NodeStatus.SCREENING), own, _roots([11, 12]), _post(inc=-0.004))     # P(gain>0) ~ 0.02
    assert d.status is NodeStatus.SCREENED_OUT and d.reason.startswith("futile at screen")
    assert _step(_node(NodeStatus.SCREENING), own, _roots([11, 12]), _post(inc=-0.002)).status is None  # 0.16
    assert _step(_node(NodeStatus.SCREENING), own, _roots([11, 12]), _post(inc=-0.004),
                 cfg=_cfg(promotion=PromotionCfg(screen_futility_p=0.0))).status is None


def test_screen_gate_needs_the_nodes_own_gain_not_the_carriers():
    own = [_run("n1", Stage.SCREEN, 11), _run("n1", Stage.SCREEN, 12)]
    carried = _post(p_pos=0.99, inc=0.0)                  # U > 0 only through the parent's lead
    assert carried.p_gain == 0.5
    d = _step(_node(NodeStatus.SCREENING, parent="p1"), own, [], carried,
              parent=[_run("p1", Stage.PARENT, 11), _run("p1", Stage.PARENT, 12)])
    assert d.status is NodeStatus.SCREENED_OUT and "P(gain>0)=0.500" in d.reason


def test_regate_once_screens_out_carrier_only_promoting_and_confirming_nodes(tmp_path):
    from types import SimpleNamespace
    ctx = _world(tmp_path)
    driver, store, _ = _make(ctx, Scripts())
    store.ensure_root("base")
    for nid, st in (("pz", NodeStatus.PROMOTING), ("pg", NodeStatus.PROMOTING),
                    ("cz", NodeStatus.CONFIRMING), ("cg", NodeStatus.CONFIRMING)):
        store.put_node(dataclasses.replace(_node(st, nid=nid), task_id=ctx.task_id))
    posts = {"pz": _post(p_pos=0.99, inc=0.0), "pg": _post(p_pos=0.99, inc=0.01),
             "cz": _post(p_inc=0.5), "cg": _post(p_inc=0.9)}
    driver.snapshot = SimpleNamespace(nodes=posts)
    acts: list[str] = []
    driver._regate(acts)
    st = {n.id: n.status for n in store.nodes() if n.id != ROOT_NODE_ID}
    assert st == {"pz": NodeStatus.SCREENED_OUT, "pg": NodeStatus.PROMOTING,
                  "cz": NodeStatus.SCREENED_OUT, "cg": NodeStatus.CONFIRMING}
    assert len(acts) == 2 and all("regate" in a for a in acts)
    from loop.driver import K_REGATE
    assert store.get_state(K_REGATE) is True


def _SIG(ctx):
    from loop.world import noise as WN
    return {t: 0.004 for t in WN.targets(ctx)}


def test_refreeze_resets_delta1_once_and_confirms_seed_spent_nodes_that_now_clear(tmp_path):
    """Once (K_REFREEZE): every CONFIRMING / NOT_CONFIRMED node's frozen delta_1 becomes confirm_delta at its frozen
    sigma bound; a NOT_CONFIRMED node closed on spent confirm seeds whose E now clears 1/alpha is CONFIRMED, one
    demoted by a remeasure never is."""
    from types import SimpleNamespace
    from loop.driver import K_REFREEZE
    from loop.world.evalues import CONFIRM_KEY
    ctx = _world(tmp_path)
    driver, store, _ = _make(ctx, Scripts())
    store.ensure_root("base")
    for nid, st, why in (("a", NodeStatus.NOT_CONFIRMED, "seeds spent E=2.05 n=2"),
                         ("b", NodeStatus.NOT_CONFIRMED, "remeasure: duplicate"), ("c", NodeStatus.CONFIRMING, "")):
        store.put_node(dataclasses.replace(_node(st, nid=nid), task_id=ctx.task_id, confirm_alt=0.0016))
        store.log("node_transition", node_id=nid, frm="confirming", to=st.value, reason=why)
    store.set_state(CONFIRM_KEY, {n: [0.0016, 0.0034] for n in "abc"})
    driver.snapshot = SimpleNamespace(nodes={n: _post() for n in "abc"}, n_scored=0, sigma=_SIG(ctx))
    driver.world.confirm_evalue = lambda snap, nid, st, led=None: (1.0 / ctx.cfg.promotion.alpha_confirm, 2)
    acts: list[str] = []
    driver._refreeze(acts)
    pr = ctx.cfg.promotion
    d1 = 0.0034 * math.sqrt(2 * math.log(1 / pr.alpha_confirm) / S.planned_confirm(ctx.cfg, []))
    assert store.get_state(CONFIRM_KEY)["c"] == pytest.approx([d1, 0.0034]) and store.get_node("c").confirm_alt == pytest.approx(d1)
    assert store.get_node("a").status is NodeStatus.CONFIRMED and store.get_node("b").status is NodeStatus.NOT_CONFIRMED
    assert len(acts) == 1 and "(refreeze: " in acts[0] and store.get_state(K_REFREEZE) is True
    from loop.world import K_REFROZEN, held_back
    assert store.get_state(K_REFROZEN) == ["a"] and held_back(store) == {"a"}      # audit first


def test_mechanism_pooled_confirmation_multiplies_same_tag_confirm_evalues_once(tmp_path):
    """Same normalized tag (whitened / whiten): confirm E 3 x 3 x 0.5 = 4.5 < 20, then a third member's 10 makes 45: the
    best confirm-only estimate graduates 'mechanism-pooled'; the spent members never pool again."""
    from types import SimpleNamespace
    from loop.driver import K_MECH
    from loop.contracts import Posterior
    ctx = _world(tmp_path)
    driver, store, _ = _make(ctx, Scripts())
    store.ensure_root("base")
    es = {"a": 3.0, "b": 3.0, "c": 0.5, "x": 50.0}
    for nid, tag in (("a", "whitened_jepa_target"), ("b", "whiten_jepa_target"), ("c", "whiten_jepa_targets"),
                     ("x", "other")):
        store.put_node(dataclasses.replace(_node(NodeStatus.NOT_CONFIRMED, nid=nid), task_id=ctx.task_id,
                                           mechanism_tag=tag))
    est = {"a": 0.002, "b": 0.004, "c": -0.001, "x": 0.01}
    posts = {n: dataclasses.replace(_post(), confirm_estimate=Posterior(m, 0.002, 2)) for n, m in est.items()}
    driver.snapshot = SimpleNamespace(nodes=posts, n_scored=0, sigma=_SIG(ctx))
    driver.world.confirm_evalue = lambda snap, nid, st, led=None: (es[nid], 2)
    acts: list[str] = []
    driver._mech_pool(acts)
    assert not acts                                  # 4.5 < 1/alpha; a lone tag never pools
    es["c"] = 2.0 / ctx.cfg.promotion.alpha_confirm / 9.0
    driver._mech_pool(acts)
    assert store.get_node("b").status is NodeStatus.CONFIRMED and "mechanism-pooled" in acts[0]
    assert sorted(store.get_state(K_MECH)["whiten_jepa_target"]) == ["a", "b", "c"]
    driver._mech_pool(acts)
    assert len(acts) == 1


def test_offset_regate_screens_out_only_nodes_the_robust_offset_lowered(tmp_path):
    """Once (K_REGATE_OFFSET): a node below its gate is screened out only when the robust control offset
    (noise.offset_ok) raised its offset estimate; one below the gate for other reasons is left to its stage."""
    from types import SimpleNamespace
    from loop.driver import K_REGATE_OFFSET
    ctx = _world(tmp_path)
    driver, store, _ = _make(ctx, Scripts())
    store.ensure_root("base")
    for nid, st in (("pz", NodeStatus.PROMOTING), ("cz", NodeStatus.CONFIRMING)):
        store.put_node(dataclasses.replace(_node(st, nid=nid), task_id=ctx.task_id))
    driver.snapshot = SimpleNamespace(nodes={"pz": _post(p_pos=0.99, inc=0.0), "cz": _post(p_inc=0.5)})
    acts: list[str] = []
    driver._regate(acts, K_REGATE_OFFSET, "robust control offsets", lambda nid: nid == "cz")
    assert store.get_node("cz").status is NodeStatus.SCREENED_OUT and store.get_node("pz").status is NodeStatus.PROMOTING
    assert len(acts) == 1 and "robust control offsets" in acts[0] and store.get_state(K_REGATE_OFFSET) is True


def test_crash_and_void_typing():
    own = [_run("n1", Stage.SCREEN, 11, RunStatus.CRASH), _run("n1", Stage.SCREEN, 12)]
    assert _step(_node(NodeStatus.SCREENING), own, _roots([11, 12]), _post()).status \
        is NodeStatus.CRASHED
    own = [_run("n1", Stage.SCREEN, 11, RunStatus.VOID, VoidType.EVAL), _run("n1", Stage.SCREEN, 12)]
    d = _step(_node(NodeStatus.SCREENING), own, _roots([11, 12]), _post())
    assert d.status is NodeStatus.VOID and d.reason == "void:eval"


def test_infra_void_is_retried_not_failed_until_cap():
    retry = _run("n1", Stage.SCREEN, 11, RunStatus.VOID, VoidType.INFRA, attempt=1)
    assert S.is_retryable(retry, 2) and not S.run_failed(retry, 2)
    d = _step(_node(NodeStatus.SCREENING), [retry, _run("n1", Stage.SCREEN, 12)],
              _roots([11, 12]), _post())
    assert d.status is None
    spent = dataclasses.replace(retry, attempt=2)
    d = _step(_node(NodeStatus.SCREENING), [spent, _run("n1", Stage.SCREEN, 12)],
              _roots([11, 12]), _post())
    assert d.status is NodeStatus.VOID


def test_promote_to_confirm_freezes_alternative():
    own = [_run("n1", Stage.PROMOTE, 13), _run("n1", Stage.PROMOTE, 14)]
    d = _step(_node(NodeStatus.PROMOTING), own, _roots([13, 14]), _post(inc=0.001))
    assert d.status is NodeStatus.CONFIRMING
    assert d.fields["confirm_alt"] == pytest.approx(0.004 * math.sqrt(2 * math.log(20) / 6))  # sigma, alpha, n
    d = _step(_node(NodeStatus.PROMOTING), own, _roots([13, 14]), _post(inc=0.01))
    assert d.fields["confirm_alt"] == pytest.approx(0.004 * math.sqrt(2 * math.log(20) / 6))  # never the estimate
    capped = _cfg(seeds=SeedPlan(max_seeds_per_node=5), promotion=PromotionCfg(confirm_max_seeds=3, alpha_confirm=0.2))
    d = S.step(_node(NodeStatus.PROMOTING), own_runs=own + [_run("n1", Stage.SCREEN, 11), _run("n1", Stage.SCREEN, 12)],
               control_runs=_roots([13, 14]), post=_post(), cfg=capped, sigma_panel=0.004, infra_retry=2,
               sigma_ucb=0.0034)
    assert d.fields["confirm_alt"] == pytest.approx(0.0034 * math.sqrt(2 * math.log(5) / 1))  # 5 - 4 selection seeds
    d = _step(_node(NodeStatus.PROMOTING), own, _roots([13, 14]), _post(p_inc=0.2))
    assert d.status is NodeStatus.SCREENED_OUT


def test_confirm_is_sequential_on_pool_seeds_paired_to_parent():
    node = _node(NodeStatus.CONFIRMING, parent="p1")
    d = _step(node, [], [], _post())
    assert [n.run_id for n in d.needs] == ["n1:confirm:101", "p1:parent:101"]   # parent-paired only
    # the parent's own done run at a pool seed is reused as the control, and such a seed goes first (conservative)
    d = _step(node, [], [], _post(), parent=[_run("p1", Stage.CONFIRM, 103)])
    assert [n.run_id for n in d.needs] == ["n1:confirm:103"]
    own = [_run("n1", Stage.CONFIRM, 101)]
    both = dict(parent=[_run("p1", Stage.PARENT, 101)])
    cfg = _cfg()
    assert _step(node, own, _roots([101]), _post(), cfg, confirm=(25.0, 1), **both).status \
        is NodeStatus.CONFIRMED                          # 1 / alpha_confirm = 20
    assert _step(node, own, _roots([101]), _post(), cfg, confirm=(0.05, 1), **both).status \
        is NodeStatus.NOT_CONFIRMED
    d = _step(node, own, _roots([101]), _post(), cfg, confirm=(2.0, 1), **both)
    assert d.status is None and [n.run_id for n in d.needs][0] == "n1:confirm:102"
    assert d.fields["confirm_e"] == 2.0
    for need in d.needs:
        assert need.seed in cfg.seeds.confirm_pool
    capped = _cfg(promotion=PromotionCfg(confirm_max_seeds=1))
    assert _step(node, own, _roots([101]), _post(), capped, confirm=(2.0, 1), **both).status \
        is NodeStatus.NOT_CONFIRMED


def test_confirm_stops_as_futile_when_even_favourable_remaining_seeds_cannot_reach_the_bar():
    """Live: 0 GPU-h reached confirm while confirm seeds were spent on hopeless nodes. After each confirm seed: if E
    could not reach 1/alpha_confirm with every remaining planned seed at delta_1 + 1 sigma, NOT_CONFIRMED now."""
    import math
    from loop.world.evalues import confirm_delta
    node = _node(NodeStatus.CONFIRMING, parent="p1")             # live: alpha 0.2, 3 confirm seeds
    cfg = _cfg(promotion=PromotionCfg(alpha_confirm=0.2, confirm_max_seeds=3, futility_e=0.0))
    own, both = [_run("n1", Stage.CONFIRM, 101)], dict(parent=[_run("p1", Stage.PARENT, 101)])
    sig = 0.006
    d1 = confirm_delta(sig, cfg.promotion.alpha_confirm, S.planned_confirm(cfg, own))
    left = S.planned_confirm(cfg, own) - 1
    gain = left * (d1 * (d1 + sig) - d1 * d1 / 2) / sig ** 2              # log E the remaining seeds add at best
    bar = math.log(1 / cfg.promotion.alpha_confirm)
    lo, hi = math.exp(bar - gain) * 0.9, math.exp(bar - gain) * 1.1
    d = _step(node, own, [], _post(), cfg, confirm=(lo, 1), frozen=(d1, sig), **both)
    assert d.status is NodeStatus.NOT_CONFIRMED and d.reason.startswith("futile: ") and "delta_1+1 sigma" in d.reason
    d = _step(node, own, [], _post(), cfg, confirm=(hi, 1), frozen=(d1, sig), **both)
    assert d.status is None and d.needs[0].run_id == "n1:confirm:102"     # still reachable: next seed
    assert _step(node, own, [], _post(), cfg, confirm=(lo, 1), **both).status is None   # no frozen alternative yet


def test_confirming_siblings_on_one_control_rotate_their_confirm_seeds(tmp_path):
    """n0405 / n0408 / n0410 all confirmed on seed 101 against one n0255 run (correlated verdicts): the k-th
    confirming sibling of a control starts at the k-th unused pool seed; a lone node is unchanged."""
    node = _node(NodeStatus.CONFIRMING, parent="p1")
    pool = _cfg().seeds.confirm_pool
    firsts = [S.step(node, own_runs=[], control_runs=[], post=_post(), cfg=_cfg(), sigma_panel=0.004, infra_retry=2,
                     confirm_rot=k).needs[0].seed for k in range(3)]
    assert firsts == list(pool[:3]) and len(set(firsts)) == 3
    ctx = _world(tmp_path)
    driver, store, _ = _make(ctx, Scripts())
    store.ensure_root("base")
    store.put_node(dataclasses.replace(_node(NodeStatus.CONFIRMED, nid="p1"), task_id=ctx.task_id))
    for i, nid in enumerate(("n0405", "n0408", "n0410")):
        store.put_node(dataclasses.replace(_node(NodeStatus.CONFIRMING, parent="p1", nid=nid), task_id=ctx.task_id,
                                           created_ts=float(i), confirm_alt=0.004))
    seeds = [driver._decide(store.get_node(n), driver._runs_by_node(), {}).needs[0].seed
             for n in ("n0405", "n0408", "n0410")]
    assert len(set(seeds)) == 3, seeds


def test_confirm_never_uses_selection_seeds():
    with pytest.raises(ValueError):
        from loop.spec import SeedPlan
        SeedPlan(screen=(1, 2), promote=(3,), confirm_pool=(3, 4))


def test_seed_sets_must_be_disjoint_and_unique():
    from loop.spec import SeedPlan
    SeedPlan()
    for bad in (dict(screen=(11, 11)), dict(screen=(11, 12), promote=(12, 13)),
                dict(confirm_pool=(101, 201))):                 # 201 is a default audit seed
        with pytest.raises(ValueError):
            SeedPlan(**bad)


def test_failed_partner_is_retried_then_voids_the_dependent():
    own = [_run("n1", Stage.SCREEN, 11), _run("n1", Stage.SCREEN, 12)]
    crashed = _run(ROOT_NODE_ID, Stage.BASELINE, 12, RunStatus.CRASH)
    assert S.is_retryable(crashed, 8)                   # a control crash is retried ONCE...
    d = _step(_node(NodeStatus.SCREENING), own, _roots([11]) + [crashed], _post())
    assert d.status is None and not d.needs             # waiting on the retry
    spent = dataclasses.replace(crashed, attempt=1)
    assert not S.is_retryable(spent, 8)                 # ...never up to the infra cap: it is deterministic
    assert S.is_retryable(dataclasses.replace(spent, status=RunStatus.VOID, void_type=VoidType.INFRA), 8)
    d = _step(_node(NodeStatus.SCREENING), own, _roots([11]) + [spent], _post())
    assert d.status is NodeStatus.VOID and d.reason == "void:partner root:baseline:12"


def test_own_eval_void_on_a_confirm_seed_ends_confirmation_not_skips_it():
    node = _node(NodeStatus.CONFIRMING)
    own = [_run("n1", Stage.CONFIRM, 101, RunStatus.VOID, VoidType.EVAL)]
    d = _step(node, own, _roots([101]), _post())
    assert d.status is NodeStatus.NOT_CONFIRMED and "void:eval" in d.reason
    spent = [_run("n1", Stage.CONFIRM, 101, RunStatus.VOID, VoidType.INFRA, attempt=2)]
    d = _step(node, spent, _roots([101, 102]), _post())       # exhausted infra: skip to the next seed
    assert d.status is None and [n.run_id for n in d.needs] == ["n1:confirm:102"]


def test_infra_void_is_retryable_but_own_crash_or_eval_void_is_not():
    assert S.is_retryable(_run("n1", Stage.SCREEN, 11, RunStatus.VOID, VoidType.INFRA), 2)
    assert not S.is_retryable(_run("n1", Stage.SCREEN, 11, RunStatus.CRASH), 2)
    assert not S.is_retryable(_run("n1", Stage.SCREEN, 11, RunStatus.VOID, VoidType.EVAL), 2)
    assert S.is_retryable(_run("p1", Stage.PARENT, 11, RunStatus.VOID, VoidType.EVAL), 2)


def test_screen_blocks_rotate_and_children_pair_against_parent_controls():
    from loop.spec import SeedPlan
    plan = SeedPlan()
    kids = [dataclasses.replace(_node(NodeStatus.QUEUED, nid=f"n{i:04d}"), created_ts=i)
            for i in range(1, 13)]
    grandkid = dataclasses.replace(_node(NodeStatus.QUEUED, parent="n0001", nid="n0013"),
                                   created_ts=13)
    blocks = S.assign_blocks(kids + [grandkid], plan)
    assert [blocks[k.id] for k in kids] == [0] * 4 + [1] * 4 + [2] * 4    # block_children = 4
    assert blocks["n0013"] == 3                          # a fresh block, never the parent's
    fixed = set(plan.promote) | set(plan.confirm_pool) | set(plan.audit)
    b1, b2 = S.screen_block(plan, 1), S.screen_block(plan, 2)
    assert S.screen_block(plan, 0) == plan.screen and len(b1) == len(plan.screen)
    assert not (set(b1) | set(b2)) & (fixed | set(plan.screen)) and not set(b1) & set(b2)
    node = _node(NodeStatus.SCREENING, parent="p1")
    d = S.step(node, own_runs=[], control_runs=[_run("p1", Stage.SCREEN, b2[0])],
               post=None, cfg=_cfg(), sigma_panel=0.004, infra_retry=2, screen=b2)
    ids = {n.run_id for n in d.needs}
    assert {f"n1:screen:{s}" for s in b2} | {f"p1:parent:{b2[1]}"} == ids   # the parent's done run at b2[0] reused


def test_a_cancelled_or_duplicate_parent_control_never_voids_a_child_on_the_parents_own_seeds():
    """Live 2026-10-03 (ladders t17/t31/t34): the dose points were children of the 1x design point, screening on
    ITS seeds; a duplicate PARENT run was launched beside its in-flight screen run, the 1x closed on its screen and
    its close cancelled that run -> both dose points void:partner. The parent's own screen run is the control."""
    node = _node(NodeStatus.SCREENING, parent="p1")
    own = [_run("n1", Stage.SCREEN, 11), _run("n1", Stage.SCREEN, 12)]
    flying = [_run("p1", Stage.SCREEN, s, RunStatus.RUNNING) for s in (11, 12)]
    d = _step(node, own, [], _post(), parent=flying)
    assert d.status is None and not d.needs                     # waits on the parent's own runs: no duplicate
    cancelled = [_run("p1", Stage.PARENT, s, RunStatus.VOID, VoidType.CANCELLED) for s in (11, 12)]
    d = _step(node, own, [], _post(), parent=cancelled + [_run("p1", Stage.SCREEN, s) for s in (11, 12)])
    assert d.status is not NodeStatus.VOID and "partner" not in d.reason
    d = _step(node, own, [], _post(), parent=cancelled)          # no reusable run: still a partner void
    assert d.status is NodeStatus.VOID and d.reason == "void:partner p1:parent:11"


def test_blocks_stick_to_screen_runs_and_promote_seeds_rotate():
    """Nodes screened under an old block size keep their block; later ones fill fresh blocks
    of ``block_children``; promote seeds rotate through the pool per block."""
    from loop.spec import SeedPlan
    plan = SeedPlan(promote=(13, 14, 15), promote_per_node=1)
    kids = [dataclasses.replace(_node(NodeStatus.QUEUED, nid=f"n{i:04d}"), created_ts=i)
            for i in range(1, 16)]
    old = S.screen_block(plan, 1)
    runs = ([_run(f"n{i:04d}", Stage.SCREEN, 11) for i in range(1, 11)]         # old block 0 of 10
            + [_run(f"n{i:04d}", Stage.SCREEN, old[0]) for i in (11, 12)])      # old block 1
    blocks = S.assign_blocks(kids, plan, runs)
    assert [blocks[k.id] for k in kids] == [0] * 10 + [1] * 4 + [2]      # block 1 fills to 4
    assert [S.promote_block(plan, b) for b in range(4)] == [(13,), (14,), (15,), (13,)]
    assert S.promote_block(SeedPlan(), 5) == SeedPlan().promote                 # 0 = every seed
    node = _node(NodeStatus.PROMOTING)
    d = S.step(node, own_runs=[], control_runs=[], post=None, cfg=_cfg(seeds=plan),
               sigma_panel=0.004, infra_retry=2, promote=S.promote_block(plan, 1))
    assert {n.run_id for n in d.needs} == {"n1:promote:14", "root:baseline:14"}


def test_audit_step_needs_then_verdict():
    cfg = _cfg()
    seeds = cfg.seeds.audit
    verdict, _, needs = S.audit_step("n1", [], [], cfg=cfg, delta=0.004, sigma=0.004,
                                     infra_retry=2)
    assert verdict == "" and {n.run_id for n in needs} == \
        {f"{i}:audit:{s}" for i in ("n1", ROOT_NODE_ID) for s in seeds}

    def runs(node: str, score: float) -> list[Run]:
        return [dataclasses.replace(_run(node, Stage.AUDIT, s), scores={"a": score})
                for s in seeds]
    good = S.audit_step("n1", runs("n1", 0.51), runs(ROOT_NODE_ID, 0.5), cfg=cfg, delta=0.01,
                        sigma=0.004, infra_retry=2)
    assert good[0] == "passed" and good[1] >= 1 / cfg.promotion.alpha_audit
    # the audit's bar is its own key: a lenient alpha_confirm never loosens it
    strict = dataclasses.replace(cfg, promotion=dataclasses.replace(
        cfg.promotion, alpha_confirm=0.9, alpha_audit=1.0 / (good[1] * 2)))
    assert S.audit_step("n1", runs("n1", 0.51), runs(ROOT_NODE_ID, 0.5), cfg=strict, delta=0.01,
                        sigma=0.004, infra_retry=2)[0] == "failed"
    bad = S.audit_step("n1", runs("n1", 0.5), runs(ROOT_NODE_ID, 0.5), cfg=cfg, delta=0.01,
                       sigma=0.004, infra_retry=2)
    assert bad[0] == "failed"


# ---------------------------------------------------------------------------
# end to end on tasks/fake_loop.yaml with the real modules
# ---------------------------------------------------------------------------

#: (config key, multiplier, family, mechanism tag, lever tags). Idea 0 is the
#: planted winner (lower LR: the sim's "train weaker" truth), idea 1 breaks init.
IDEAS = [
    ("train.lr", 0.3, "optimization", "lower_lr", {"effective_lr": -1}),
    ("model.depth", 16 / 12, "architecture", "deeper", {"capacity": 1}),
    ("aug.strength", 1.6, "augmentation", "stronger_aug", {"augmentation_strength": 1}),
    ("loss.temperature", 1.05, "objective", "temp", {"objective_difficulty": 1}),
    ("train.weight_decay", 0.5, "regularization", "less_wd", {"regularization": -1}),
    ("train.warmup", 1.4, "optimization", "longer_warmup", {"training_strength": -1}),
    ("train.lr", 1.5, "optimization", "higher_lr", {"effective_lr": 1}),
    ("aug.strength", 0.7, "augmentation", "weaker_aug", {"augmentation_strength": -1}),
]


def _set(cfg: dict, key: str, value: Any) -> None:
    d = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = value


def _get(cfg: dict, key: str) -> Any:
    d = cfg
    for p in key.split("."):
        d = d[p]
    return d


class Scripts:
    """FakeLLMClient scripts for every role; records what each role saw."""

    def __init__(self) -> None:
        self.research_calls = 0
        self.reviews = 0

    def researcher(self, **call: Any) -> Any:
        k = self.research_calls
        self.research_calls += 1
        key, mult, family, tag, levers = IDEAS[k % len(IDEAS)]
        cfg_path = Path(call["cwd"]) / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        old = _get(cfg, key)
        # later rounds nudge further so diffs stay unique
        new = old * (mult ** (1 + k // len(IDEAS)))
        new = int(round(new)) if isinstance(old, int) else round(new, 8)
        _set(cfg, key, new)
        finish = {"hypothesis": {"claim": f"{tag}: {key} {old}->{new} helps",
                                 "mechanism_tag": tag, "family": family,
                                 "predictions": {PANEL: {"sign": 1, "size": 0.005}},
                                 "falsification": "panel delta <= 0 on screen seeds"},
                  "levers": levers,
                  "handoff": {"actions": f"set {key}={new}", "learnings": "n/a",
                              "concerns": "none"}}
        return {"tool_calls": [("write_file", {"path": "config.yaml",
                                               "content": yaml.safe_dump(cfg, sort_keys=True)}),
                               ("smoke", {})],
                "finish": finish}

    def json_role(self, **call: Any) -> Any:
        req = set(call["schema"].get("required", ()))
        if "tests_hypothesis" in req:
            return {"tests_hypothesis": True, "rationale": "the diff moves exactly the knob"}
        if "winner" in req:
            return {"winner": "A", "rationale": "first is simpler"}
        if "assumptions" in req:                         # a first-principles review
            return {"summary": "the loop's assumptions hold", "assumptions": []}
        if "rules" in req:                               # the meta-learning consolidation (loop.meta)
            self.consolidations = getattr(self, "consolidations", 0) + 1
            return {"rules": []}
        if "meta_hypotheses" in req:
            self.reviews += 1
            return {"summary": "LR-down helped; strong aug hurt: the model is over-driven",
                    "findings": ["lower lr won", "stronger augmentation lost"],
                    "meta_hypotheses": [{
                        "claim": "the model is over-regularised/over-driven: train weaker",
                        "mechanism_tag": "train_weaker", "family": "optimization",
                        "levers": {"effective_lr": -1, "regularization": -1},
                        "predicted_sign": 1, "predicted_size": 0.005,
                        "falsification": "lever-down trials do not beat their parents",
                        "probes": ["halve weight decay"]}],
                    "probe_briefs": ["halve weight decay on the incumbent"],
                    "explore_bias": {"regularization": 0.01}}
        raise AssertionError(f"unexpected json role call: {sorted(req)}")

    def client(self) -> Any:
        from loop.agents.llm import FakeLLMClient
        return FakeLLMClient({"researcher": self.researcher,
                              "review": self.json_role,
                              "brief": self.json_role})


class Killed(BaseException):
    """Simulates the driver process dying mid-submit."""


def _world(tmp_path: Path, *, review_every: int = 3):
    ctx = load_context(FAKE, state_dir=tmp_path / "state")
    cfg = dataclasses.replace(ctx.cfg, review=ReviewCfg(stall_evals=review_every, stall_min_hours=0.0))
    return dataclasses.replace(ctx, cfg=cfg)


def _make(ctx, scripts: Scripts, adapter=None, seed: int = 7):
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    adapter = adapter or SimAdapter(ctx)
    driver, store = build(ctx, llm=scripts.client(), adapter=adapter, executor=InlineExecutor,
                          rng=random.Random(seed))
    driver.sleep = lambda s: None
    return driver, store, adapter


def _run_ticks(driver: Driver, n: int, occupancy: list[int] | None = None) -> list[str]:
    log: list[str] = []
    for _ in range(n):
        log += driver.tick()
        if occupancy is not None:
            occupancy.append(len(driver.book.active()))
    return log


def test_end_to_end_fake_loop(tmp_path):
    ctx = _world(tmp_path)
    scripts = Scripts()
    # the scripted researcher has a handful of ideas; past them every request is a near-dup reject, so on some rng
    # paths the queue runs dry late in the 40 ticks (rng 7 after the carrier rotation: 27/36 warm ticks full; rng 1-8
    # average 99.6%): this test checks slot filling given supply, on a path that keeps supply
    # (tuning passes are part of that supply here: the fresh-confirm-seed tuning gate is off, loop.tune.gate)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, moonshot_tune=dataclasses.replace(
        ctx.cfg.moonshot_tune, min_confirm_sigmas=0.0)))
    driver, store, adapter = _make(ctx, scripts, seed=5)
    adapter.inject_fault("n0003:screen:*", "crash", at_poll=1, times=1)
    flaky = {"big_review": 0}

    def fail_once(name: str) -> Any:
        real = getattr(driver.analyst, name)

        def call(*a: Any, **k: Any) -> Any:
            flaky[name] += 1
            if flaky[name] == 1:
                raise RuntimeError("llm down")
            return real(*a, **k)
        return call
    for name in flaky:
        setattr(driver.analyst, name, fail_once(name))
    occupancy: list[int] = []
    log = _run_ticks(driver, 40, occupancy)
    runs = store.runs()
    nodes = {n.id: n for n in store.nodes()}

    # slots stay full once the loop is warm
    warm = occupancy[4:]
    assert sum(o == ctx.cfg.capacity for o in warm) >= 0.8 * len(warm), occupancy

    # every compared run has its control (root's baseline, or the parent's parent-stage run, or the parent's own
    # DONE run at that seed reused as the control: loop.seeds.REUSE_STAGES)
    by_id = {r.id: r for r in runs}
    for r in runs:
        if r.stage in (Stage.SCREEN, Stage.PROMOTE, Stage.CONFIRM) and r.status is RunStatus.DONE:
            pid, stage = S.control_of(nodes[r.node_id])
            assert any(Run.make_id(pid, st, r.seed) in by_id for st in (stage, *S.REUSE_STAGES)), r.id

    # the planted winner (idea 0 -> n0001) is CONFIRMED, and only on its fresh confirm block (loop.remint)
    win = nodes["n0001"]
    assert win.status is NodeStatus.CONFIRMED, (win.status, log[-30:])
    confirm = [r for r in runs if r.node_id == win.id and r.stage is Stage.CONFIRM]
    assert confirm and all(r.seed in RM.confirm_seeds(store, S.control_of(win)[0]) for r in confirm)
    selection = set(ctx.cfg.seeds.screen) | set(ctx.cfg.seeds.promote)
    assert not selection & {r.seed for r in confirm}
    assert win.confirm_e >= 1 / ctx.cfg.promotion.alpha_confirm
    assert win.confirm_alt is not None and win.confirm_alt > 0

    # every research session's node records whether it had web tools and how many searches it made
    assert all(re.search(r" web=(0|1 searches=\d+ fetches=\d+)\b", n.notes) for n in nodes.values()
               if n.notes.startswith("request=")), [n.notes for n in nodes.values()]

    # the init-breaking architecture change (idea 1 -> n0002) is rejected before any run
    arch = nodes["n0002"]
    assert arch.family == "architecture" and arch.status is NodeStatus.REJECTED
    assert "init" in arch.reject_reason
    assert not store.runs(node_id=arch.id)

    # the crash costs no LLM postmortem (loop.explain computes explanations on demand)
    assert nodes["n0003"].status is NodeStatus.CRASHED
    assert not store.postmortems(node_id="n0003")

    # every closed node was paid a bandit reward exactly once
    pulls = store.pulls()
    assert len({p.node_id for p in pulls}) == len(pulls)
    closed = [n for n in nodes.values() if n.status.value in
              {"rejected", "screened_out", "confirmed", "not_confirmed", "vetoed", "void", "crashed",
               "audit_failed"}]
    # (a moonshot tuning pass charges the arm separately under '<base>#tune': loop.tune)
    assert {p.node_id for p in pulls if not p.node_id.endswith("#tune")} == {n.id for n in closed}
    assert all(p.node_id.removesuffix("#tune") in nodes for p in pulls)

    # a stall review fired (cadence reviews are gone) and its meta-hypothesis has an e-value from linked trials
    reviews = [r for r in store.reviews() if r.trigger != "first_principles"]
    assert reviews and reviews[0].trigger == "stall"
    metas = store.hypotheses(kind=HypothesisKind.META)
    assert metas and metas[0].review_id == reviews[0].id

    # no run id launched twice
    assert len({r.id for r in runs}) == len(runs)
    # a failed review is not marked done: it was re-asked and stored
    assert flaky["big_review"] >= 2

    # Labless needs the held-out audit (the planted winner vs root on the audit seeds), which
    # starts by itself once the winner is the incumbent: nobody asks
    assert win.id in (a := driver._audit_book())["queue"] + a["passed"], a

    def meta_scored() -> bool:
        ev = driver.snapshot.hypotheses
        return any(ev[m.id].n_trials >= 1 and ev[m.id].e_value != 1.0 for m in metas)
    for _ in range(120):
        driver.tick()
        a = driver._audit_book()
        if win.id in a["passed"] + a["failed"] and meta_scored():
            break
    assert meta_scored(), [driver.snapshot.hypotheses[m.id] for m in metas]
    nodes = {n.id: n for n in store.nodes()}
    linked = store.links(metas[0].id)
    assert linked, "no trial linked to the meta-hypothesis"
    # trials link by sign(L . M): every link agrees with the lever dot product; an explicit / mechanistic
    # link with L . M = 0 (e.g. a tuned moonshot whose tuning moved a meta key, loop.tune) is +1 (D3.9)
    m_dirs = metas[0].levers.directions
    for nid, sign in linked:
        dot = sum(d * m_dirs.get(l, 0) for l, d in nodes[nid].levers.directions.items())
        assert sign == (1 if dot >= 0 else -1), (nid, sign, dot)
    assert driver.labless_ready(win.id), driver._audit_book()
    audit = {r.seed for r in store.runs(node_id=win.id) if r.stage is Stage.AUDIT}
    gate = driver._audit_book()["gate"][win.id]          # seed rotation (loop.remint): its own never-used block
    assert audit == set(gate["audit_seeds"]) and min(audit) >= ctx.cfg.seeds.fresh_base
    assert len(audit) == len(ctx.cfg.seeds.audit) and not audit & set(ctx.cfg.seeds.audit)
    assert not audit & (selection | set(ctx.cfg.seeds.confirm_pool))
    drawn = [set(d["seeds"]) for d in store.get_state("seeds.drawn")]
    assert all(not a & b for i, a in enumerate(drawn) for b in drawn[i + 1:])     # no block is ever reused


def test_failed_audit_demotes_to_last_audited_ancestor(tmp_path):
    ctx = _world(tmp_path, review_every=1000)
    driver, store, adapter = _make(ctx, Scripts())
    _run_ticks(driver, 2)
    root = store.get_node(ROOT_NODE_ID)
    for nid, parent in (("a1", ROOT_NODE_ID), ("a2", "a1")):
        store.put_node(dataclasses.replace(root, id=nid, parent_id=parent, strategy=Strategy.EXPLOIT,
                                           status=NodeStatus.CONFIRMED, notes=""))
    store.set_state("driver.audit", {"passed": ["a1"], "current": {
        "node": "a2", "delta": 0.01, "sigma": 0.004}})
    for _ in range(12):                                   # same commit as root: zero gain
        driver.tick()
        if "a2" in driver._audit_book()["failed"]:
            break
    assert driver._audit_book()["failed"] == ["a2"]
    g = driver._audit_book()["gate"]["a2"]                           # the verdict's own numbers, frozen
    assert (g["verdict"], g["n"], g["delta"], g["sigma"], g["source"]) == ("failed", 4, 0.01, 0.004, "gate")
    assert g["alpha"] == ctx.cfg.promotion.alpha_audit and g["mean"] == pytest.approx(0.0, abs=1e-9)
    e = math.exp(sum(c * 0.01 - 0.01 ** 2 / 2 for c in g["deltas"].values()) / 0.004 ** 2)
    assert g["e"] == pytest.approx(e) and store.events("audit")[-1]["e_value"] == pytest.approx(e)
    assert store.get_node("a2").status is not NodeStatus.CONFIRMED
    assert store.get_node("a1").status is NodeStatus.CONFIRMED        # audited ancestor kept


def test_failed_run_is_cancelled_and_retried_and_keeps_first_submit_ts(tmp_path):
    ctx = _world(tmp_path, review_every=1000)
    driver, store, adapter = _make(ctx, Scripts())
    adapter.inject_fault("root:baseline:11", "infra", at_poll=1, times=1)
    cancelled: list[str] = []
    real = adapter.cancel
    adapter.cancel = lambda h: (cancelled.append(h.run_id), real(h))
    while store.get_run("root:baseline:11") is None:
        driver.tick()
    ts0 = store.get_run("root:baseline:11").submitted_ts
    _run_ticks(driver, 6)
    r = store.get_run("root:baseline:11")
    assert "root:baseline:11" in cancelled and r.attempt == 1 and r.submitted_ts == ts0


def test_orphan_worktrees_are_reaped_on_restart(tmp_path):
    from loop.adapters.sim import SimAdapter
    ctx = _world(tmp_path, review_every=1000)
    driver, store, adapter = _make(ctx, Scripts())
    wt = adapter.prepare_worktree(parent_commit=adapter.root_commit(), name="loop-dead")
    assert Path(wt.path).exists()                       # the process "dies" here
    store.close()
    driver2, _, _ = _make(ctx, Scripts(), adapter=SimAdapter(ctx))
    log = driver2.tick()
    assert f"reaped orphan worktree {wt.path}" in log and not Path(wt.path).exists()
    assert not any("reaped" in line for line in driver2.tick())          # once per process


def test_single_instance_lock(tmp_path):
    from loop.__main__ import _lock
    ctx = _world(tmp_path)
    held = _lock(ctx)
    assert held is not None and _lock(ctx) is None
    held.close()
    assert _lock(ctx) is not None


def test_kill_and_resume_produces_no_duplicate_runs(tmp_path):
    ctx = _world(tmp_path, review_every=1000)
    scripts = Scripts()
    driver, store, adapter = _make(ctx, scripts)
    _run_ticks(driver, 8)
    # the next submit reaches the adapter but the process dies before the row gets its job id
    real = adapter.submit
    state = {"killed": None}

    def dying_submit(*, run, commit):
        real(run=run, commit=commit)
        state["killed"] = run.id
        raise Killed(run.id)

    adapter.submit = dying_submit
    with pytest.raises(Killed):
        _run_ticks(driver, 20)
    killed = state["killed"]
    assert store.get_run(killed).status is RunStatus.SUBMITTING
    assert store.get_run(killed).job_id == ""
    store.close()

    # a brand-new driver (new store connection, new adapter) on the same state
    from loop.adapters.sim import SimAdapter
    driver2, store2, adapter2 = _make(ctx, scripts, adapter=SimAdapter(ctx))
    submits: list[str] = []
    real2 = adapter2.submit

    def counting(*, run, commit):
        submits.append(run.id)
        return real2(run=run, commit=commit)

    adapter2.submit = counting
    log = _run_ticks(driver2, 20)
    assert any("resume: found job" in line and killed in line for line in log), log[:5]
    assert killed not in submits              # found via adapter.find, never resubmitted
    assert store2.get_run(killed).job_id
    runs = store2.runs()
    assert len({r.id for r in runs}) == len(runs)
    # one sim job per (run, attempt): nothing double-launched
    jobs = [json.loads(p.read_text()) for p in (Path(ctx.state_dir) / "sim" / "jobs").glob("*.json")]
    per_run: dict[str, int] = {}
    for j in jobs:
        per_run[j["run_id"]] = per_run.get(j["run_id"], 0) + 1
    by_id = {r.id: r for r in runs}
    for rid, n in per_run.items():
        assert n <= by_id[rid].attempt + 1, (rid, n)


def test_dry_run_submits_nothing_and_calls_no_llm(tmp_path):
    ctx = _world(tmp_path)
    scripts = Scripts()
    driver, store, adapter = _make(ctx, scripts)
    plan = driver.tick(dry_run=True)
    assert any("would" in p for p in plan)
    assert scripts.research_calls == 0 and not store.runs() and store.get_node("root") is None


def test_infra_void_resubmits_same_run(tmp_path):
    ctx = _world(tmp_path, review_every=1000)
    scripts = Scripts()
    driver, store, adapter = _make(ctx, scripts)
    adapter.inject_fault("root:baseline:11", "infra", at_poll=1, times=1)
    _run_ticks(driver, 8)
    r = store.get_run("root:baseline:11")
    assert r.attempt == 1 and r.status in (RunStatus.RUNNING, RunStatus.DONE)


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

def _unmeasured_copy(tmp_path: Path) -> Path:
    text = FAKE.read_text()
    text = text.replace("seed_transfer_sd: 0.004", "seed_transfer_sd: UNMEASURED  # to measure")
    text = text.replace("panel_sd: 0.004", "panel_sd: UNMEASURED")
    text = text.replace("axis_sd: {alpha: 0.005, beta: 0.006, gamma: 0.007}",
                        "axis_sd: {alpha: UNMEASURED, beta: UNMEASURED, gamma: UNMEASURED}")
    p = tmp_path / "fake_loop.yaml"
    p.write_text(text)
    return p


def test_calibrate_measures_and_rewrites_only_those_lines(tmp_path):
    from loop.adapters.sim import SimAdapter
    from loop.contracts import UnmeasuredLoopNoise
    spec = _unmeasured_copy(tmp_path)
    with pytest.raises(UnmeasuredLoopNoise, match="calibrate"):
        load_context(spec, state_dir=tmp_path / "state")
    ctx = load_context(spec, state_dir=tmp_path / "state", require_measured=False)
    store = SqliteStore(tmp_path / "state.db", ctx.task_id, weights=ctx.cfg.weights)
    before = spec.read_text().splitlines()
    seeds = calibration_seeds(ctx)
    assert seeds == (11, 12, 13, 14, 101, 102, 103, 104)     # 8 by default (df 7)
    # the simulator draws noise from declared axis sds: hand it the hidden truth
    truth = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, panel_sd=0.004, axis_sd={"alpha": 0.005, "beta": 0.006, "gamma": 0.007}))
    res = calibrate(ctx, store, SimAdapter(truth), sleep=lambda s: None, echo=lambda s: None)
    assert res is not None and res.written
    after = spec.read_text().splitlines()
    assert len(after) == len(before)
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    # seed_transfer_sd, panel_sd, axis_sd (+ replicate_sd when replicates differ; the sim's
    # replicates are bit-identical, so a zero replicate sd is NOT written)
    assert res.replicate_sd[PANEL] == 0.0 and ("noise", "replicate_sd") not in res.updates
    assert len(changed) == 3
    for i in changed:
        assert "# measured" in after[i] and "by loop calibrate" in after[i]
    ctx2 = load_context(spec, state_dir=tmp_path / "state")      # now runnable
    assert ctx2.cfg.panel_sd == pytest.approx(res.paired_sd[PANEL], rel=1e-5)
    assert ctx2.cfg.axis_sd["beta"] == pytest.approx(math.sqrt(2) * res.seed_sd["beta"], rel=1e-5)
    # the sim's per-axis noise is the declared axis sd, so the estimate is in the right range
    assert 0.001 < res.seed_sd["alpha"] < 0.02
    # replicate-1 runs are root's baseline runs; sixteen runs total, nothing duplicated
    runs = store.runs(node_id=ROOT_NODE_ID)
    assert len(runs) == 16 and {r.stage for r in runs} == {Stage.BASELINE, Stage.CALIBRATE}
    # re-running is idempotent (no new runs)
    calibrate(ctx, store, SimAdapter(truth), sleep=lambda s: None, echo=lambda s: None, write=False)
    assert len(store.runs(node_id=ROOT_NODE_ID)) == 16


def test_rewrite_yaml_block_mapping_and_comments():
    text = ("# header\nnoise:\n  # why\n  seed_transfer_sd: UNMEASURED  # old\n"
            "  replicate_sd: UNMEASURED\nloop:\n  panel_sd: UNMEASURED\n  axis_sd:\n"
            "    a: UNMEASURED\n    b: UNMEASURED\n  other: 1\n")
    out = rewrite_yaml(text, {("noise", "seed_transfer_sd"): 0.0031,
                              ("noise", "replicate_sd"): 0.001,
                              ("loop", "panel_sd"): 0.0044,
                              ("loop", "axis_sd", "a"): 0.005,
                              ("loop", "axis_sd", "b"): 0.006}, "measured X")
    data = yaml.safe_load(out)
    assert data["noise"]["seed_transfer_sd"] == 0.0031
    assert data["loop"]["axis_sd"] == {"a": 0.005, "b": 0.006}
    assert out.startswith("# header\nnoise:\n  # why\n") and "  other: 1\n" in out
    with pytest.raises(KeyError):
        rewrite_yaml(text, {("loop", "missing"): 1.0}, "x")


def test_compute_needs_two_seeds():
    ctx = load_context(FAKE, state_dir="/nonexistent-unused")
    one = [Run(id="root:baseline:11", task_id=ctx.task_id, node_id="root", seed=11,
               stage=Stage.BASELINE, status=RunStatus.DONE,
               scores={"alpha": 0.6, "beta": 0.6, "gamma": 0.6})]
    with pytest.raises(RuntimeError):
        compute(ctx, one, (11, 12))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "loop", *args], cwd=REPO,
                          capture_output=True, text=True, timeout=120)


def test_cli_dry_run_exits_zero(tmp_path):
    cp = _cli("run-v2", "--task", "tasks/fake_loop.yaml", "--dry-run", "--fake-llm",
              "--state-dir", str(tmp_path / "st"))
    assert cp.returncode == 0, cp.stderr
    assert "dry run" in cp.stdout


def test_cli_refuses_unmeasured_nanopath_naming_calibrate(tmp_path):
    # `review` never launches anything: on unmeasured sigma it refuses (exit 2), naming
    # calibrate, and writes no state. (`run` here would calibrate on the REAL cluster.)
    import re
    spec = tmp_path / "nanopath_v2.yaml"          # the live spec is measured now: an unmeasured copy
    spec.write_text(re.sub(r"(\n  panel_sd:) [^\n]*", r"\1 UNMEASURED",
                           (REPO / "tasks" / "nanopath_v2.yaml").read_text()))
    cp = _cli("review", "--task", str(spec), "--fake-llm", "--state-dir", str(tmp_path / "st"))
    assert cp.returncode == 2, (cp.stdout, cp.stderr)
    assert "calibrate" in cp.stderr
    assert not (tmp_path / "st").exists()


def test_cli_run_on_unmeasured_sigma_calibrates_then_loops_in_one_process(tmp_path, monkeypatch):
    import loop.__main__ as M
    from loop.adapters.sim import SimAdapter
    spec = _unmeasured_copy(tmp_path)
    truth = load_context(spec, state_dir=tmp_path / "st", require_measured=False)
    truth = dataclasses.replace(truth, cfg=dataclasses.replace(
        truth.cfg, panel_sd=0.004, axis_sd={"alpha": 0.005, "beta": 0.006, "gamma": 0.007}))
    monkeypatch.setattr(M, "resolve_adapter", lambda ctx: SimAdapter(truth))
    monkeypatch.setattr(M, "_wait", lambda s: None)
    rc = M.main(["run-v2", "--task", str(spec), "--fake-llm", "--max-ticks", "6",
                 "--state-dir", str(tmp_path / "st")])
    assert rc == 0
    assert "# measured" in spec.read_text()                  # calibration wrote the sigmas
    ctx = load_context(spec, state_dir=tmp_path / "st")     # ... and the spec is now runnable
    store = SqliteStore(tmp_path / "st" / f"{ctx.task_id}.db", ctx.task_id, weights=ctx.cfg.weights)
    runs = store.runs()
    assert any(r.id.startswith("root:calibrate:") for r in runs)
    assert any(e["event"] == "research_started" for e in store.events())   # then the loop ran
    store.close()


def test_seed_cap_rejects_a_spec_over_max_seeds_per_node(tmp_path):
    text = FAKE.read_text().replace("loop:\n", "loop:\n  seeds: {screen: [11, 12], promote: [13, 14], "
                                    "max_seeds_per_node: 5}\n  promotion: {confirm_max_seeds: 2}\n", 1)
    p = tmp_path / "capped.yaml"
    p.write_text(text)
    with pytest.raises(ValueError, match="exceeds max_seeds_per_node 5"):
        load_context(p, state_dir=tmp_path / "st")
    p.write_text(text.replace("max_seeds_per_node: 5", "max_seeds_per_node: 8"))
    assert load_context(p, state_dir=tmp_path / "st").cfg.seeds.max_seeds_per_node == 8   # 2+2+2 fits


def test_too_slow_is_a_non_retryable_verdict_that_earns_a_postmortem():
    from loop.contracts import VoidType
    slow = _run("n1", Stage.SCREEN, 11, RunStatus.VOID, VoidType.TOO_SLOW)
    assert not S.is_retryable(slow, 8) and S.run_failed(slow, 8)      # a verdict, never retried


def test_no_config_import_from_driver_modules():
    code = ("import sys, loop.driver, loop.seeds, loop.calibrate, loop.__main__; "
            "assert 'config' not in sys.modules, 'config imported'")
    cp = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr


def test_a_closed_node_has_its_running_jobs_cancelled(tmp_path):
    ctx = _world(tmp_path, review_every=1000)
    driver, store, adapter = _make(ctx, Scripts())
    adapter.inject_fault("n0001:screen:11", "crash", at_poll=1, times=1)
    adapter.inject_fault("n0001:screen:12", "hang", times=1)       # would hold its GPU forever
    for _ in range(12):
        driver.tick()
        if store.get_node("n0001") and store.get_node("n0001").status is NodeStatus.CRASHED:
            break
    driver.tick()
    assert store.get_node("n0001").status is NodeStatus.CRASHED
    run = store.get_run("n0001:screen:12")
    assert run.status is RunStatus.VOID and run.void_type is VoidType.CANCELLED   # the loop's own cancel
    assert not S.is_retryable(run, 9)
    assert "n0001:screen:12" not in {e["run_id"] for e in store.events("run_finished")}   # never an infra void
    assert [e["run_id"] for e in store.events("run_cancelled")] == ["n0001:screen:12"]


def test_closing_a_node_keeps_the_parent_runs_its_active_children_pair_against(tmp_path):
    ctx = _world(tmp_path, review_every=1000)
    driver, store, adapter = _make(ctx, Scripts())
    for nid, par, st in (("p1", ROOT_NODE_ID, NodeStatus.SCREENED_OUT), ("k1", "p1", NodeStatus.SCREENING)):
        store.put_node(dataclasses.replace(_node(st, parent=par, nid=nid), task_id=ctx.task_id))
    for stage, seed in ((Stage.PARENT, 21), (Stage.SCREEN, 12)):
        store.put_run(dataclasses.replace(_run("p1", stage, seed, RunStatus.RUNNING), task_id=ctx.task_id, job_id="9"))
    adapter.cancel = lambda handle: None
    driver.book.cancel("p1", "screened out")
    assert store.get_run("p1:parent:21").status is RunStatus.RUNNING        # k1's control at 21: kept
    assert store.get_run("p1:screen:12").void_type is VoidType.CANCELLED
    store.update_node("k1", status=NodeStatus.SCREENED_OUT)
    driver.book.cancel("p1", "screened out")
    assert store.get_run("p1:parent:21").void_type is VoidType.CANCELLED   # no child left waiting on it


def test_past_cancels_are_relabelled_once(tmp_path):
    ctx = _world(tmp_path, review_every=1000)
    driver, store, _ = _make(ctx, Scripts())
    store.ensure_root("base")
    store.put_node(dataclasses.replace(_node(NodeStatus.SCREENED_OUT, nid="a"), task_id=ctx.task_id))
    store.log("run_finished", run_id="a:screen:2", status="void", void_type="infra")       # before the close
    store.log("node_transition", node_id="a", frm="screening", to="screened_out", reason="x")
    store.log("run_finished", run_id="a:screen:1", status="void", void_type="infra")       # the cancel
    for seed in (1, 2):
        store.put_run(dataclasses.replace(_run("a", Stage.SCREEN, seed, RunStatus.VOID), task_id=ctx.task_id,
                                          void_type=VoidType.INFRA))
    acts: list[str] = []
    driver._relabel_cancels(acts)
    assert store.get_run("a:screen:1").void_type is VoidType.CANCELLED
    assert store.get_run("a:screen:2").void_type is VoidType.INFRA and len(acts) == 1


def _first_node(driver: Driver, store: SqliteStore) -> Node:
    for _ in range(5):
        driver.tick()
        if store.get_node("n0001") is not None:
            return store.get_node("n0001")
    raise AssertionError("no node produced")


def test_debug_child_pairs_against_the_crashed_nodes_own_control(tmp_path):
    from loop.contracts import CandidateRequest, Operator
    ctx = _world(tmp_path)
    driver, store, adapter = _make(ctx, Scripts())
    crashed = store.update_node(_first_node(driver, store).id, status=NodeStatus.CRASHED)
    req = CandidateRequest(id="req-dbg", task_id=ctx.task_id, strategy=Strategy.REVISIT,
                           operator=Operator.DEBUG, island="main", parent_ids=(crashed.id,),
                           brief="fix it", brief_kind="failure")
    res = driver.researcher.research(req, store, driver.snapshot)
    assert res.candidate is not None
    assert res.candidate.parent_commit == store.get_node(ROOT_NODE_ID).commit   # diff vs the control
    driver._intake(req, res)
    child = next(n for n in store.nodes() if n.notes.startswith("request=req-dbg"))
    assert child.parent_id == crashed.parent_id == ROOT_NODE_ID
    assert child.second_parent_id == crashed.id and child.debug_depth == crashed.debug_depth + 1
    assert S.control_of(child) == (ROOT_NODE_ID, Stage.BASELINE)   # never the crashed node's runs


def test_heartbeat_line_every_log_every_min_with_a_fake_clock(ctx, store, capsys):
    from loop.adapters.sim import SimAdapter
    from loop.driver import hb_line
    now = [0.0]
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, heartbeat=dataclasses.replace(ctx.cfg.heartbeat, log_every_min=10.0)))
    d = Driver(ctx, store, SimAdapter(ctx, clock=lambda: now[0]), None, None, None, None,
               clock=lambda: now[0], executor=InlineExecutor)
    d.tick = lambda dry_run=False: []
    d.sleep = lambda s: now.__setitem__(0, now[0] + 10.0 * s)     # poll_s 30 in 1 s naps: 5 fake min/tick
    d.run(max_ticks=7)                                            # t = 0, 5, ..., 30 min
    hbs = [ln for ln in capsys.readouterr().out.splitlines() if " hb phase=loop " in ln]
    assert len(hbs) == 4                                          # t = 0, 10, 20, 30 min
    assert hbs[1].startswith("1970-01-01T00:10:00Z hb phase=loop slots=0/")
    assert hb_line(ctx, store, "calibrate").endswith(
        "incumbent=root(+0.0000) best_post=- best_fresh=- confirmed=0 resumed=0/0 done labless: DISABLED proposals: 0 awaiting operator")


def test_calibrate_honours_stop_within_a_second_of_fake_time(ctx, store):
    # the SIGTERM bug: calibration slept poll_s at a time; now it naps <= 1 s and checks stop()
    from loop.adapters.sim import SimAdapter
    t, naps = [0.0], []

    def sleep(s: float) -> None:
        naps.append(s)
        t[0] += s
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, heartbeat=dataclasses.replace(ctx.cfg.heartbeat, poll_s=30.0)))
    res = calibrate(ctx, store, SimAdapter(ctx, duration_ticks=10**6), clock=lambda: t[0],
                    sleep=sleep, echo=lambda m: None, stop=lambda: t[0] >= 42.5)
    assert res is None and 42.5 <= t[0] <= 43.5 and max(naps) <= 1.0


def test_stop_flag_is_lock_free_and_driver_run_exits_mid_wait(ctx, store):
    import loop.__main__ as M
    from loop.adapters.sim import SimAdapter
    stop = M._Stop()
    stop.set()
    stop.set()                                    # a second SIGTERM (uv + systemd) is harmless
    assert stop.is_set() and not hasattr(stop, "_cond")
    t = [0.0]
    d = Driver(ctx, store, SimAdapter(ctx), None, None, None, None, clock=lambda: t[0],
               executor=InlineExecutor)
    d.tick = lambda dry_run=False: []
    d.sleep = lambda s: t.__setitem__(0, t[0] + s)
    d.run(stop=lambda: t[0] >= 5.0)               # poll_s 30: returns after ~5 s, not 30
    assert 5.0 <= t[0] <= 6.0


def test_in_flight_requests_are_published_with_distinct_explore_families(tmp_path):
    """4 concurrent researchers: each request sees the earlier ones (K_INFLIGHT), so no two
    explore requests share a target family; a finished request leaves the list."""
    from concurrent.futures import Future
    from loop.contracts import K_INFLIGHT, ResearchResult, Strategy
    from loop.spec import ResearcherCfg

    class Pending:                                   # research never finishes by itself
        def __init__(self, **_: Any) -> None:
            self.futs: list[Future] = []

        def submit(self, fn, *a, **k) -> Future:
            self.futs.append(Future())
            return self.futs[-1]

        def shutdown(self, **_: Any) -> None:
            pass
    ctx = _world(tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, researcher=ResearcherCfg(max_concurrent=4),
        intake=dataclasses.replace(ctx.cfg.intake, queue_low_water=4),      # a target that wants all 4
        bandit=dataclasses.replace(ctx.cfg.bandit, explore_floor=1.0)))     # force the explore arm
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    driver, store = build(ctx, llm=Scripts().client(), adapter=SimAdapter(ctx), executor=Pending,
                          rng=random.Random(7))
    driver.tick()
    driver.tick()                                    # starts are staggered: at most 2 per tick (research_starts)
    inflight = store.get_state(K_INFLIGHT)
    fams = [r["family"] for r in inflight if r["strategy"] == Strategy.EXPLORE.value]
    assert len(inflight) == 4 and len(fams) >= 3 and len(set(fams)) == len(fams), inflight
    next(iter(driver._research)).set_result(ResearchResult("failed", None, "boom"))   # the first request
    driver.tick()
    assert inflight[0]["id"] not in {r["id"] for r in store.get_state(K_INFLIGHT)}


def test_restart_reissues_wait_for_a_slot_and_a_stop_drains_in_flight_research(tmp_path):
    """Live 09-29..10-06: 217 requests were re-issued at once after restarts. Re-issues now queue under
    researcher.max_concurrent (persisted as queued, no extra restart counted); a stop drains: no new LLM job, the
    loop ticks until in-flight research is harvested or drain_min passes."""
    from loop.contracts import K_INFLIGHT, ResearchResult
    from loop.spec import ResearcherCfg
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    ctx = _world(tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, researcher=ResearcherCfg(max_concurrent=4),
        intake=dataclasses.replace(ctx.cfg.intake, queue_low_water=4)))
    driver, store = build(ctx, llm=Scripts().client(), adapter=SimAdapter(ctx), executor=_Pending,
                          rng=random.Random(7))
    for _ in range(3):
        driver.tick()
    assert len(driver._research) == 4
    lost = store.get_state(K_INFLIGHT)
    small = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, researcher=ResearcherCfg(max_concurrent=2)))
    d2, _ = build(small, llm=Scripts().client(), adapter=SimAdapter(small), executor=_Pending, rng=random.Random(7))
    d2.tick()                                                    # the restart: 4 lost, 2 slots
    assert len(d2._research) == 2 and len(d2._requeue) == 2
    book = d2.store.get_state(K_INFLIGHT)
    assert [e["id"] for e in book] == [e["id"] for e in lost] and [bool(e.get("queued")) for e in book] == [0, 0, 1, 1]
    assert all(e["restarts"] == 1 for e in book)
    d3, _ = build(small, llm=Scripts().client(), adapter=SimAdapter(small), executor=_Pending, rng=random.Random(7))
    d3.tick()                                                    # restarted again: queued ones are no new restart
    assert [e["restarts"] for e in d3.store.get_state(K_INFLIGHT)] == [2, 2, 1, 1]
    now = [1.0e9]
    d3.clock, d3.sleep = (lambda: now[0]), (lambda s: now.__setitem__(0, now[0] + s))
    (first, *_rest) = list(d3._research)
    first.set_result(ResearchResult("failed", None, "boom"))
    d3.run(stop=lambda: True)                                     # SIGTERM: drain, never start new work
    assert d3._draining and d3._requeue and len(d3._research) == 1   # one harvested; the queued never started
    assert now[0] >= 1.0e9 + 60 * small.cfg.researcher.drain_min    # waited out the drain


class _Pending:                                        # research never finishes by itself
    def __init__(self, **_: Any) -> None:
        from concurrent.futures import Future
        self.new, self.futs = Future, []

    def submit(self, fn, *a, **k):
        self.futs.append(self.new())
        return self.futs[-1]

    def shutdown(self, **_: Any) -> None:
        pass


def test_a_restart_logs_and_reissues_in_flight_research_up_to_max_restarts(tmp_path):
    """Researchers die with the process: on startup every request K_INFLIGHT still lists is logged
    research_failed status=restart and re-issued (same id, arm and brief) -- a restart is not the request's
    attempt -- up to MAX_RESTARTS times; lost once more, it is dropped."""
    from loop.contracts import K_INFLIGHT
    from loop.driver import MAX_RESTARTS
    from loop.spec import ResearcherCfg
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    ctx = _world(tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, researcher=ResearcherCfg(max_concurrent=2)))

    def boot():
        return build(dataclasses.replace(ctx), llm=Scripts().client(), adapter=SimAdapter(ctx),
                     executor=_Pending, rng=random.Random(7))
    driver, store = boot()
    driver.tick()
    first = {r["id"]: r for r in store.get_state(K_INFLIGHT)}
    assert len(first) == 2 and all(r["restarts"] == 0 and r["req"]["brief"] for r in first.values())
    store.close()
    for k in range(1, MAX_RESTARTS + 1):
        driver2, store2 = boot()
        log = driver2.tick()
        lost = [e for e in store2.events("research_failed") if e.get("status") == "restart"]
        assert len(lost) == 2 * k and all(e["reissued"] for e in lost)
        again = {r.id: r for r in driver2._research.values()}
        assert set(first) <= set(again), log
        assert all(again[i].strategy.value == first[i]["strategy"] and again[i].brief == first[i]["req"]["brief"]
                   for i in first)
        assert all(r["restarts"] == k for r in store2.get_state(K_INFLIGHT) if r["id"] in first)
        store2.close()
    driver3, store3 = boot()
    driver3.tick()
    lost = [e for e in store3.events("research_failed") if e.get("request_id") in first]
    assert len(lost) == 2 * (MAX_RESTARTS + 1) and sum(not e["reissued"] for e in lost) == 2
    assert not set(first) & {r.id for r in driver3._research.values()}


def test_a_closed_node_costs_no_llm_call_and_is_explained_by_the_hook(tmp_path, monkeypatch):
    """The Opus postmortem is gone: a close starts no LLM job; loop.explain asks loop.diagnostics.explain (None
    while it does not exist) and falls back to a stored historical postmortem."""
    import sys
    import types
    from loop import explain as X
    from loop.contracts import LeverTags, Operator, Postmortem
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    ctx = _world(tmp_path, review_every=1000)
    driver, store = build(ctx, llm=Scripts().client(), adapter=SimAdapter(ctx), executor=_Pending,
                          rng=random.Random(7))
    driver.tick()
    n = Node(id="n9001", task_id=ctx.task_id, commit="c", parent_id=ROOT_NODE_ID, strategy=Strategy.MOONSHOT,
             operator=Operator.DRAFT, family="f", mechanism_tag="m", diff_hash="h9001", status=NodeStatus.VETOED)
    store.put_node(n)
    driver._refit()
    acts: list[str] = []
    driver._triggers(acts)
    assert not any("postmortem" in a for a in acts), acts
    monkeypatch.setitem(sys.modules, "loop.diagnostics", None)          # no diagnostics module: None
    assert X.explain(n, store) is None and X.block(store, [n]) == "(none)"
    store.put_postmortem(Postmortem(id="pm-old", task_id=ctx.task_id, node_id=n.id, verdict="loss", explanation="old",
                                    mechanism_reasoning="", levers=LeverTags({}), created_ts=1.0))
    assert "old (stored postmortem)" in X.block(store, [n])
    monkeypatch.setitem(sys.modules, "loop.diagnostics",
                        types.SimpleNamespace(explain=lambda node, st, snap: f"residual of {node.id}"))
    assert X.explain(n, store) == "residual of n9001" and "residual of n9001" in X.block(store, [n])


def test_audit_offsets_remove_the_shared_root_audit_run():
    """Every audit pairs against the same root audit runs: another audited node that beat its own
    estimate by +0.004 there says root's run was low, so the audited node's delta is lowered."""
    from loop.world.noise import shared_offset
    cfg = _cfg()
    seed = cfg.seeds.audit[0]

    def run(node, score):
        return dataclasses.replace(_run(node, Stage.AUDIT, seed), scores={"a": score})
    by_node = {ROOT_NODE_ID: [run(ROOT_NODE_ID, 0.5)], "n1": [run("n1", 0.51)], "n2": [run("n2", 0.505)]}
    off = S.audit_offsets("n1", by_node, cfg, {"n2": (0.001, 0.0)}, sigma=0.004)
    assert off == {seed: pytest.approx(shared_offset([(0.004, 0.0)], 0.004)[0])} and off[seed] > 0
    alone = S.audit_offsets("n1", {k: v for k, v in by_node.items() if k != "n2"}, cfg, {}, 0.004)
    assert alone == {seed: 0.0}                          # no other audited node: no correction


def test_slots_fill_the_primary_pool_first_then_the_preemptible_one(ctx, store):
    from types import SimpleNamespace
    from loop.contracts import JobHandle
    from loop.driver import RunBook, pool_use
    from loop.spec import PoolCfg, SlurmCfg
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=(
        PoolCfg("primary", slots=1), PoolCfg("idle", slots=2, preemptible=True)))))
    assert ctx.cfg.capacity == 3
    book = RunBook(ctx, store, SimpleNamespace(submit=lambda run, commit: JobHandle(run.id, "1", "", 0.0)))
    root = store.ensure_root("c0")
    got = [book.launch(root, Stage.BASELINE, s).pool for s in (11, 12, 13)]
    assert got == ["primary", "idle", "idle"] and pool_use(ctx, store.runs()) == \
        "primary 1/1, idle 2/2 (preemptible)"
    store.update_run("root:baseline:11", status=RunStatus.VOID)            # a primary slot frees up
    assert book.retry(store.get_run("root:baseline:11"), root).pool == "primary"


def test_pools_replace_loop_slots(tmp_path):
    text = FAKE.read_text().replace("  slots: 4\n", "  slots: 4\n  slurm: {pools: [{name: a, slots: 2}]}\n")
    (spec := tmp_path / "s.yaml").write_text(text)
    with pytest.raises(ValueError, match="no loop.slots beside them"):
        load_context(spec, state_dir=tmp_path / "st")
    (spec := tmp_path / "t.yaml").write_text(text.replace("  slots: 4\n", "", 1))
    assert load_context(spec, state_dir=tmp_path / "st").cfg.capacity == 2


# --------------------------------------------- run importance routing + cluster courtesy
def _pools_ctx(ctx, primary=2, idle=2, **slurm):
    from loop.spec import PoolCfg, SlurmCfg
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, slurm=SlurmCfg(pools=(
        PoolCfg("primary", slots=primary), PoolCfg("idle", slots=idle, preemptible=True)), **slurm)))


def _pnode(store, nid, strategy=None, operator=None):
    from loop.contracts import Operator, Strategy
    n = Node(id=nid, task_id=store.task_id, commit="c", parent_id=ROOT_NODE_ID,
             strategy=strategy or Strategy.EXPLORE, operator=operator or Operator.DRAFT, family="f",
             mechanism_tag="m", diff_hash=f"h{nid}", status=NodeStatus.SCREENING)
    store.put_node(n)
    return n


class _PoolAdapter:
    """Fake SLURM: submit / poll (every job PENDING) / cancel / repool, recorded."""

    def __init__(self):
        self.jobs, self.cancelled, self.moved = 0, [], []

    def submit(self, *, run, commit):
        from loop.contracts import JobHandle
        self.jobs += 1
        return JobHandle(run.id, str(self.jobs), f"/o/{run.id}", 0.0)

    def poll(self, handle):
        from loop.contracts import JobStatus
        return JobStatus(RunStatus.RUNNING, detail="PENDING")

    def cancel(self, handle):
        self.cancelled.append(handle.job_id)

    def repool(self, handle, run):
        from loop.contracts import JobHandle
        self.cancel(handle)
        self.jobs += 1
        self.moved.append((run.id, run.pool))
        return JobHandle(run.id, str(self.jobs), handle.output_dir, handle.submitted_ts)


def test_high_runs_take_primary_first_low_runs_preemptible_first(ctx, store):
    from loop.contracts import Operator, Strategy
    from loop.driver import RunBook
    ctx = _pools_ctx(ctx)
    book = RunBook(ctx, store, _PoolAdapter())
    low, stacked, likely = _pnode(store, "n1"), _pnode(store, "n2", Strategy.DESIGN, Operator.DRAFT), _pnode(store, "n3")
    point = _pnode(store, "n4", Strategy.DESIGN)
    store.update_node("n4", operator=None)
    point = store.get_node("n4")
    book.p_positive = lambda nid: 0.9 if nid == "n3" else 0.2
    assert [book.high(n, Stage.SCREEN) for n in (low, stacked, likely, point)] == [False, True, True, False]
    assert all(book.high(low, st) for st in (Stage.PROMOTE, Stage.CONFIRM, Stage.AUDIT, Stage.PARENT, Stage.BASELINE))
    assert book.pool(low, Stage.SCREEN) == "idle" and book.pool(stacked, Stage.SCREEN) == "primary"
    assert book.pool(likely, Stage.SCREEN) == "primary" and book.pool(low, Stage.PROMOTE) == "primary"
    assert [book.launch(low, Stage.SCREEN, s).pool for s in (1, 2)] == ["idle", "idle"]
    book.high_demand = 1                                   # a HIGH run waits: LOW stays off primary
    assert book.pool(low, Stage.SCREEN) == "" and book.launch(low, Stage.SCREEN, 3) is None
    book.high_demand = 0                                   # no HIGH demand: LOW may use idle primary
    assert book.launch(low, Stage.SCREEN, 3).pool == "primary"
    assert book.launch(stacked, Stage.SCREEN, 1).pool == "primary"
    assert book.pool(point, Stage.SCREEN) == "" and book.pool(likely, Stage.SCREEN) == "primary"  # all full
    store.update_run("n1:screen:1", status=RunStatus.DONE)                    # an idle slot frees
    assert book.launch(likely, Stage.SCREEN, 1).pool == "idle"               # HIGH overflows to preemptible


def test_eval_only_runs_go_to_the_preemptible_pool_only(ctx, store):
    """An eval-only run launches as PROMOTE / CONFIRM / AUDIT (HIGH stages) but is a ~23 min refit on a
    trained checkpoint: it runs on idle, waits when idle is full, and is never repooled to primary."""
    from loop.driver import RunBook
    from loop.evalonly import EO_SEED
    ctx = _pools_ctx(ctx, primary=2, idle=1)
    book = RunBook(ctx, store, _PoolAdapter())
    n = _pnode(store, "n1")
    for st in (Stage.PROMOTE, Stage.CONFIRM, Stage.AUDIT):
        assert book.pool(n, st, EO_SEED + 1) == "idle" and book.pool(n, st) == "primary"
    assert book.launch(n, Stage.PROMOTE, EO_SEED).pool == "idle"
    assert book.pool(n, Stage.PROMOTE, EO_SEED + 1) == "" and book.launch(n, Stage.PROMOTE, EO_SEED + 1) is None
    book.poll()
    assert book.promote_stuck() == []                                       # primary free, but not for it


def test_a_pending_high_run_on_preemptible_moves_to_a_free_primary_slot(ctx, store):
    """n0104's case: the stacked node's screens sat PENDING on idle behind cheap screens; once a
    primary slot frees they are scancelled and relaunched there (same row, same attempt)."""
    from loop.contracts import Operator, Strategy
    from loop.driver import RunBook
    ctx = _pools_ctx(ctx, primary=1, idle=3)
    adapter = _PoolAdapter()
    book = RunBook(ctx, store, adapter)
    root, low, stacked = store.ensure_root("c0"), _pnode(store, "n1"), _pnode(store, "n2", Strategy.DESIGN, Operator.DRAFT)
    assert book.launch(root, Stage.BASELINE, 9).pool == "primary"
    assert book.launch(stacked, Stage.SCREEN, 1).pool == "idle"             # primary full: overflow
    assert book.launch(low, Stage.SCREEN, 1).pool == "idle"
    book.poll()
    assert book.pending == {"root:baseline:9", "n2:screen:1", "n1:screen:1"}
    assert book.promote_stuck() == [] and adapter.moved == []               # no primary slot free
    store.update_run("root:baseline:9", status=RunStatus.DONE)
    acts = book.promote_stuck()
    run = store.get_run("n2:screen:1")
    assert adapter.moved == [("n2:screen:1", "primary")] and adapter.cancelled == ["2"]
    assert run.pool == "primary" and run.job_id == "4" and run.attempt == 0 and run.status is RunStatus.RUNNING
    assert store.get_run("n1:screen:1").pool == "idle" and len(acts) == 1   # LOW stays where it is
    assert book.promote_stuck() == []                                      # primary is full again


def test_a_high_run_preempted_again_and_again_moves_to_primary_even_when_full_and_nothing_is_niced(ctx, store):
    """n0358: a promote run on idle requeued 5x under the same job id (invisible before: 0 preemptions logged)."""
    from loop.contracts import JobStatus, Operator, Strategy
    from loop.driver import RunBook
    ctx = _pools_ctx(ctx, primary=1, idle=3)
    adapter = _PoolAdapter()
    adapter.niced = []
    adapter.nice = lambda handle, n: adapter.niced.append((handle.run_id, n))
    book = RunBook(ctx, store, adapter)
    root, low, stacked = store.ensure_root("c0"), _pnode(store, "n1"), _pnode(store, "n2", Strategy.DESIGN, Operator.DRAFT)
    assert book.launch(root, Stage.BASELINE, 9).pool == "primary"
    assert book.launch(stacked, Stage.SCREEN, 1).pool == "idle" and book.launch(low, Stage.SCREEN, 1).pool == "idle"
    assert adapter.niced == []          # nice ranks a job below EVERY user's (live: 390 vs 400): off by default
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING" if h.run_id.startswith("root") else "PENDING",
                                       restarts=1 + (h.run_id == "n2:screen:1"),          # primary's queue: empty
                                       requeued=1, gpu_h=0.5)
    book.poll()
    ev = store.events("run_preempted")
    assert len(ev) == 3 and sum(e["gpu_h"] for e in ev) == pytest.approx(1.5)
    assert book.restarts == {"root:baseline:9": 1, "n2:screen:1": 2, "n1:screen:1": 1}
    acts = book.promote_stuck()                                              # primary is full, but n2 keeps
    assert adapter.moved == [("n2:screen:1", "primary")] and len(acts) == 1  # getting preempted: it queues there
    assert "n2:screen:1" not in book.restarts and store.get_run("n1:screen:1").pool == "idle"
    assert adapter.niced == []


def test_a_preemption_wave_queues_at_most_the_overflow_cap_on_a_full_primary(ctx, store):
    """Live 10-08: one wave requeued 8 HIGH screens at once and all 8 queued on the full primary (16/8, AssocGrpGRES).
    At most max(1, slots // 4) queue there; the rest stay due on idle and take primary slots as they free."""
    from loop.contracts import JobStatus, Operator, Strategy
    from loop.driver import RunBook
    ctx = _pools_ctx(ctx, primary=8, idle=16)
    adapter = _PoolAdapter()
    book = RunBook(ctx, store, adapter)
    root = store.ensure_root("c0")
    for seed in range(8):
        assert book.launch(root, Stage.BASELINE, seed).pool == "primary"
    hi = [_pnode(store, f"n{i}", Strategy.DESIGN, Operator.DRAFT) for i in range(6)]
    assert all(book.launch(n, Stage.SCREEN, 1).pool == "idle" for n in hi)
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="PENDING" if h.run_id.startswith("n") else "RUNNING",
                                       restarts=2, requeued=1)
    book.poll()
    acts = book.promote_stuck()
    assert len(acts) == 2 and len(adapter.moved) == 2 and all("again and again" in a for a in acts)
    assert book.promote_stuck() == []                                       # primary's queue is at the cap
    for seed in range(3):                                                   # 3 primary slots free: the 2 queued
        store.update_run(f"root:baseline:{seed}", status=RunStatus.DONE)    # take 2, a third run moves into the last
    assert len(book.promote_stuck()) == 1 and len(adapter.moved) == 3
    assert sum(r.pool == "primary" for r in book.active()) == 8 and book.promote_stuck() == []


def test_at_most_max_pending_per_pool_of_our_runs_wait_pending_the_rest_stay_unsubmitted(ctx, store):
    """Smokes go first among our own jobs by submitting at most slurm.max_pending_per_pool runs beyond those SLURM
    runs (not by nice); jobs already pending are never cancelled, smoke placement ignores the gate."""
    from loop.driver import RunBook
    ctx = _pools_ctx(ctx, primary=8, idle=8, max_pending_per_pool=2, smoke_reserve=0)
    book = RunBook(ctx, store, _PoolAdapter())
    root = store.ensure_root("c0")
    assert book.limits() == {"primary": 8, "idle": 8}                     # before the first poll: no gate
    for seed in range(5):
        book.launch(root, Stage.BASELINE, seed)                           # 5 on primary
    book.pool_states = {"primary": (3, 2), "idle": (0, 0)}                # 2 of them PENDING: the cap is reached
    assert book.limits() == {"primary": 5, "idle": 8}                     # running + 2; idle keeps up: its slots
    assert book.free() == 8 and book._open(False) == []                  # primary takes no more
    assert book.limits(gate=False)["primary"] == 8
    book.pool_states = {"primary": (3, 6), "idle": (0, 0)}                # more pending than the cap: kept
    assert book.limits()["primary"] == 5 and book.free() == 8


def test_only_pending_runs_count_against_max_pending_per_pool_a_pool_that_keeps_up_fills_in_one_tick(ctx, store):
    """Live 2026-10-05: 35/64 running, 0 pending, 19 nodes queued. The gate cut every pool to running + 2 at the
    last poll, so 2 jobs started per tick -- about what finished -- and 29 GPUs sat idle for hours. Only our
    PENDING jobs count against the cap: a pool SLURM keeps up with fills to its slots at once."""
    from loop.driver import RunBook
    ctx = _pools_ctx(ctx, primary=8, idle=56, max_pending_per_pool=2, smoke_reserve=0)
    book = RunBook(ctx, store, _PoolAdapter())
    root = store.ensure_root("c0")
    for seed in range(35):
        book.launch(root, Stage.BASELINE, seed)
    assert sorted(r.pool for r in book.active()).count("idle") == 27
    book.pool_states = {"primary": (8, 0), "idle": (27, 0)}
    assert book.free() == 29                                             # was 2: running + 2 on idle
    book.pool_states = {"primary": (8, 0), "idle": (25, 2)}               # SLURM stopped keeping up on idle
    assert book.free() == 0


def test_the_idea_queue_is_sized_on_idle_gpus_not_on_the_pending_gate(ctx, store):
    """Live 2026-10-04: 29/64 running, 0 pending; free() (pending-gated: running + 2 per pool) said 4, so the queue
    target was 2 and ~3 researchers ran instead of ~16. headroom() counts the idle GPUs of every pool SLURM keeps
    up with, and none of a pool where max_pending_per_pool of our runs already wait."""
    from loop.driver import RunBook, research_starts
    ctx = _pools_ctx(ctx, primary=8, idle=56, max_pending_per_pool=2)
    book = RunBook(ctx, store, _PoolAdapter())
    root = store.ensure_root("c0")
    for seed in range(29):
        book.launch(root, Stage.BASELINE, seed)
    book.pool_states = {"primary": (8, 0), "idle": (21, 0)}
    assert book.free() == 35 and book.headroom() == 35                    # 64 - 29
    assert research_starts(0, 3, book.headroom() // 2, 40, 22) == 6     # was research_starts(0, 3, 2, ..) == 0
    book.pool_states = {"primary": (8, 0), "idle": (19, 2)}               # SLURM stopped keeping up on idle
    assert book.headroom() == 0


def test_the_queue_target_holds_through_a_short_pending_spell(ctx, store):
    """Live 2026-10-05 15:23-16:13: 2+ of our runs PENDING on idle zeroed its headroom, the queue target fell to
    queue_low_water and researchers drained 12 -> 0; SLURM then granted the runs and 40 slots waited ~45 min on a
    research pipeline restarted from zero. The target is sized on the max headroom over one research lead time
    (researcher.moonshot_timeout_min), so a short spell keeps researchers going and a long one still shrinks it."""
    from loop.driver import research_starts
    ctx = _pools_ctx(ctx, primary=8, idle=56, max_pending_per_pool=2)
    t = [0.0]
    d = Driver(ctx, store, _PoolAdapter(), None, None, None, None, clock=lambda: t[0], executor=InlineExecutor)
    root = store.ensure_root("c0")
    for seed in range(24):
        d.book.launch(root, Stage.BASELINE, seed)
    d.book.pool_states = {"primary": (8, 0), "idle": (16, 0)}
    target, _ = d._queue_marks()
    assert target == 20                                                  # (64 - 24) // 2
    d.book.pool_states = {"primary": (8, 0), "idle": (14, 2)}             # a pending spell: idle headroom gated to 0
    t[0] = 600.0
    assert d.book.headroom() == 0
    target, qmax = d._queue_marks()
    assert target == 20                                                  # was queue_low_water (2): no researcher
    assert research_starts(3, 4, target, qmax, 16) > 0
    t[0] = 60.0 * ctx.cfg.researcher.moonshot_timeout_min + 601.0        # pending past the lead time
    assert d._queue_marks()[0] == ctx.cfg.intake.queue_low_water


def test_a_high_run_requeued_while_running_moves_on_the_requeue_event(ctx, store):
    """n0366:screen:377 (HIGH) was requeued 15x and stayed on idle: a poll rarely caught it PENDING. The requeue
    event itself (restarts >= slurm.repool_restarts) moves it; a LOW run only at slurm.repool_restarts_any."""
    from loop.contracts import JobStatus, Operator, Strategy
    from loop.driver import RunBook
    ctx = _pools_ctx(ctx, primary=1, idle=3)
    adapter = _PoolAdapter()
    book = RunBook(ctx, store, adapter)
    root, low, stacked = store.ensure_root("c0"), _pnode(store, "n1"), _pnode(store, "n2", Strategy.DESIGN, Operator.DRAFT)
    book.launch(root, Stage.BASELINE, 9)
    book.launch(stacked, Stage.SCREEN, 1)
    book.launch(low, Stage.SCREEN, 1)
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", restarts=1, requeued=1)
    book.poll()
    assert book.promote_stuck() == [] and book.repool_due == set()          # once is bad luck
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", restarts=2, requeued=1)
    book.poll()
    assert book.pending == set() and book.repool_due == {"root:baseline:9", "n2:screen:1", "n1:screen:1"}
    assert len(book.promote_stuck()) == 1 and adapter.moved == [("n2:screen:1", "primary")]
    assert "n2:screen:1" not in book.repool_due and all(e.get("seeded") for e in store.events("run_preempted"))
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", restarts=5, requeued=1)
    book.poll()                                  # n0366:screen:377 (LOW) 37x: repool_restarts_any moves ANY run
    assert book.promote_stuck() == [] and ("n1:screen:1", "primary") not in adapter.moved  # LOW: never onto a
    for rid in ("n2:screen:1", "root:baseline:9"):                           # full primary (it held a slot
        store.update_run(rid, status=RunStatus.DONE)
    acts = book.promote_stuck()                                              # PENDING there); a free one: yes
    assert ("n1:screen:1", "primary") in adapter.moved and any("n1:screen:1" in a and "(LOW" in a for a in acts)


def test_preemption_rate_ignores_the_telemetry_first_poll_backfill(ctx, store):
    from loop import selftune as T
    ctx = _pools_ctx(ctx, primary=1, idle=3)
    store.log("run_preempted", run_id="a", pool="idle", gpu_h=0.1, detail="slurm requeue (restarts 6)")
    for _ in range(5):                                                       # the first poll's backfill
        store.log("run_preempted", run_id="a", pool="idle", gpu_h=0.1, detail="slurm requeue (restarts 6)")
    t = store.events("run_preempted")[-1]["ts"]
    assert T.real_preemptions(store.events("run_preempted"), {"idle"}) == []
    late = [{"ts": t + 600, "pool": "idle", "detail": "slurm requeue (restarts 7)"},
            {"ts": t, "pool": "idle", "detail": "slurm requeue (restarts 1)", "seeded": True}]
    assert len(T.real_preemptions(store.events("run_preempted") + late, {"idle"})) == 2


def test_fill_keeps_low_runs_off_primary_while_high_demand_waits(ctx, store):
    from loop.adapters.sim import SimAdapter
    ctx = _pools_ctx(ctx, primary=1, idle=1)
    d = Driver(ctx, store, SimAdapter(ctx), None, None, None, None, executor=InlineExecutor)
    root = store.ensure_root("c0")
    d.book.launch(root, Stage.BASELINE, 9)                                   # HIGH on primary
    d.book.pending.add("root:baseline:9")                                   # ...and SLURM holds it
    d._needs = lambda nodes, by_node: []
    d.selector = type("Sel", (), {"rank_queue": lambda *a: [], "next_request": lambda *a: None})()
    d._fill([])
    assert d.book.high_demand == 1


def test_courtesy_backoff_caps_preemptible_while_a_protected_job_pends(ctx, store):
    from loop.driver import COURTESY_RELEASE_S, K_COURTESY, hb_line
    ctx = _pools_ctx(ctx, primary=1, idle=12, protect_jobs=("qwen-serve-low-pri",))
    t, probes, waiting = [0.0], [], [["qwen-serve-low-pri"]]
    adapter = _PoolAdapter()
    adapter.pending_jobs = lambda names: (probes.append(tuple(names)), waiting[0])[1]
    d = Driver(ctx, store, adapter, None, None, None, None, clock=lambda: t[0], executor=InlineExecutor)
    for s in range(10):                                                     # 10 held preemptible
        d.book.launch(_pnode(store, f"n{s}"), Stage.SCREEN, 1)
    assert d.book.used()["idle"] == 10
    acts = []
    d._courtesy(acts)
    assert d.book.courtesy_cap == 6 and d.book.limits() == {"primary": 1, "idle": 6}
    assert acts and acts[0].startswith("courtesy_backoff on: qwen-serve-low-pri PENDING -> preemptible cap 6")
    assert d.book.pool(store.get_node("n0"), Stage.SCREEN) in ("", "primary")      # never a NEW idle job
    assert d.book.smoke_pool() == "primary" and d.book.free() == 1
    assert "courtesy_backoff=on(cap 6; qwen-serve-low-pri pending)" in hb_line(ctx, store, "loop")
    assert store.get_state(K_COURTESY)["on"] and store.events("courtesy_backoff")[-1]["on"]
    waiting[0] = []
    t[0] = 60.0
    d._courtesy(acts := [])                                                 # probed at most every 2 min
    assert len(probes) == 1 and d.book.courtesy_cap == 6 and acts == []
    t[0] = 121.0
    d._courtesy(acts := [])                                                 # out of PENDING: a grace first
    assert len(probes) == 2 and d.book.courtesy_cap == 6 and acts[0].startswith("courtesy_backoff grace")
    t[0] = 121.0 + COURTESY_RELEASE_S
    d._courtesy(acts := [])
    assert len(probes) == 3 and d.book.courtesy_cap is None and acts == ["courtesy_backoff off"]
    assert d.book.limits() == {"primary": 1, "idle": 12} and "courtesy" not in hb_line(ctx, store, "loop")
    assert adapter.cancelled == []                                          # never cancels for courtesy
    t[0] = 121.0 + COURTESY_RELEASE_S + 200                                 # a failed squeue keeps the state
    adapter.pending_jobs = lambda names: (_ for _ in ()).throw(RuntimeError("squeue down"))
    d._courtesy(acts := [])
    assert d.book.courtesy_cap is None and acts == [] and store.events("courtesy_probe_failed")


def test_courtesy_cap_has_a_floor_of_four(ctx, store):
    ctx = _pools_ctx(ctx, primary=1, idle=12, protect_jobs=("q",))
    adapter = _PoolAdapter()
    adapter.pending_jobs = lambda names: ["q"]
    d = Driver(ctx, store, adapter, None, None, None, None, executor=InlineExecutor)
    d.book.launch(_pnode(store, "n1"), Stage.SCREEN, 1)
    d._courtesy([])
    assert d.book.courtesy_cap == 4 and d.book.pool(store.get_node("n1"), Stage.SCREEN) == "idle"


def _court_driver(ctx, store, idle=32, held=32):
    ctx = _pools_ctx(ctx, primary=1, idle=idle, protect_jobs=("qwen-serve-low-pri",))
    t, waiting, adapter = [0.0], [["qwen-serve-low-pri"]], _PoolAdapter()
    adapter.pending_jobs = lambda names: list(waiting[0])
    d = Driver(ctx, store, adapter, None, None, None, None, clock=lambda: t[0], executor=InlineExecutor)
    for s in range(held):
        d.book.launch(_pnode(store, f"n{s}"), Stage.SCREEN, 1)
    return d, t, waiting, adapter


def _idle_free(d):
    return max(0, d.book.limits()["idle"] - d.book.used()["idle"])


def test_courtesy_backoff_survives_a_protected_job_start_that_is_preempted_minutes_later(ctx, store):
    """2026-09-30: qwen-serve-low-pri (QOS bottom) was started and preempted again 45 times; each start
    took it out of PENDING for one probe, the backoff lifted ('courtesy_backoff off'), every idle slot
    was refilled at once (12:33:20: 17 submits) and the re-armed cap was held-4 of the NEW held
    (cap 17 at 12:30:54 -> cap 27 at 12:41:18). Now the cap never rises while qwen keeps waiting."""
    d, t, waiting, _ = _court_driver(ctx, store)
    d._courtesy(acts := [])
    assert d.book.courtesy_cap == 28 and acts[0].startswith("courtesy_backoff on")
    for i in range(11):                                      # 11 preemptible jobs finish over 20 min
        store.update_run(f"n{i}:screen:1", status=RunStatus.DONE)
        t[0] += 110
        d._courtesy([])
    assert d.book.used()["idle"] == 21 and d.book.courtesy_cap == 21 and _idle_free(d) == 0
    waiting[0] = []                                          # qwen starts ... (one probe out of PENDING)
    t[0] += 130
    d._courtesy(acts := [])
    assert d.book.courtesy_cap == 21 and _idle_free(d) == 0 and d.book.smoke_pool() == "primary"
    assert acts and "grace" in acts[0] and [e["on"] for e in store.events("courtesy_backoff")] == [True]
    low = store.get_node("n0")
    assert d.book.pool(low, Stage.SCREEN) in ("", "primary")             # never a NEW idle job
    waiting[0] = ["qwen-serve-low-pri"]                      # ... and is preempted: PENDING again
    t[0] += 130
    d._courtesy(acts := [])
    assert d.book.courtesy_cap == 21 and "stays on" in acts[0]
    assert [e["on"] for e in store.events("courtesy_backoff")] == [True]  # never flipped off
    store.update_run("n11:screen:1", status=RunStatus.DONE)
    t[0] += 130
    d._courtesy([])
    assert d.book.courtesy_cap == 20 and _idle_free(d) == 0             # follows held down, never up


def test_courtesy_cap_never_rises_while_on_and_steps_down_after_an_hour(ctx, store):
    d, t, waiting, _ = _court_driver(ctx, store, held=20)
    d._courtesy([])
    assert d.book.courtesy_cap == 16
    caps = []
    for _ in range(int(3 * 3600 / 120)):                     # 3 h pending, nothing finishes
        t[0] += 120
        d._courtesy(acts := [])
        caps.append(d.book.courtesy_cap)
    assert all(a >= b for a, b in zip(caps, caps[1:]))                  # monotone
    steps = store.events("courtesy_step")
    assert [(e["cap_from"], e["cap_to"]) for e in steps] == [(16, 12), (12, 8), (8, 4)]
    assert [round(e["pending_min"]) for e in steps] == [60, 90, 120]
    assert d.book.courtesy_cap == 4 and d.book.pool(store.get_node("n0"), Stage.SCREEN) in ("", "primary")


def test_courtesy_backoff_resumes_its_cap_across_a_driver_restart(ctx, store):
    d, t, waiting, adapter = _court_driver(ctx, store, held=10)
    d._courtesy([])
    store.update_run("n0:screen:1", status=RunStatus.DONE)
    t[0] = 200.0
    d._courtesy([])
    assert d.book.courtesy_cap == 6
    for i in range(1, 5):                                    # 4 more finish while the driver is down
        store.update_run(f"n{i}:screen:1", status=RunStatus.DONE)
    d2 = Driver(d.ctx, store, adapter, None, None, None, None, clock=lambda: t[0], executor=InlineExecutor)
    d2._courtesy(acts := [])
    assert d2.book.courtesy_cap == 5 and acts == []                     # same backoff, no re-arm at held-4


def test_calibrate_honours_the_courtesy_cap(ctx, store):
    from loop.driver import Courtesy, RunBook
    ctx = _pools_ctx(ctx, primary=1, idle=12, protect_jobs=("q",))
    adapter = _PoolAdapter()
    adapter.pending_jobs = lambda names: ["q"]
    book = RunBook(ctx, store, adapter)
    for s in range(6):
        book.launch(_pnode(store, f"n{s}"), Stage.SCREEN, 1)
    Courtesy(ctx, store, adapter, book).step([])
    assert book.courtesy_cap == 4 and book.limits()["idle"] == 4 and book.free() == 1


def test_probe_node_links_explicitly_to_its_meta(tmp_path):
    """A node researched from a meta's own probe brief is EXPLICITLY linked (auto=False) even when
    its levers are orthogonal to the meta's (the lever rule would have logged probe_unlinked)."""
    from loop.contracts import AxisPrediction, CandidateRequest, HypothesisRecord, LeverTags, Operator
    ctx = _world(tmp_path)
    driver, store, adapter = _make(ctx, Scripts())
    _first_node(driver, store)
    meta = HypothesisRecord(id="rv-9-abc-m1", task_id=ctx.task_id, kind=HypothesisKind.META,
                            claim="data curation is the lever", mechanism_tag="curate",
                            family="optimization", predictions={"panel": AxisPrediction(1, 0.004)},
                            falsification="curated trials do not gain",
                            levers=LeverTags({"data_curation": 1}), source="review",
                            created_ts=driver.clock() - 1.0)
    store.put_hypothesis(meta)
    req = CandidateRequest(id="req-probe", task_id=ctx.task_id, strategy=Strategy.EXPLORE,
                           operator=Operator.IMPROVE, island="main", parent_ids=(ROOT_NODE_ID,),
                           brief="PROBE rv-9-abc-m1", brief_kind="probe",
                           meta_hypothesis_id=meta.id)
    res = driver.researcher.research(req, store, driver.snapshot)
    assert res.candidate is not None
    driver._intake(req, res)
    child = next(n for n in store.nodes() if n.notes.startswith("request=req-probe"))
    assert child.status is not NodeStatus.REJECTED, child.reject_reason
    assert (child.id, 1) in store.links(meta.id, explicit_only=True)
    assert not store.events("probe_unlinked")


def test_measure_book_and_one_time_remeasure_of_a_confirmed_copy(tmp_path):
    """The driver books each node's contained ancestors (git ancestry, nearest effective config first); once on
    deploy a CONFIRMED copy is NOT_CONFIRMED, its audit runs cancelled (void/cancelled), its audit and unsent
    Labless entries dropped, its bandit credit re-paid, the stall clock put back, the measurement epoch bumped."""
    from loop.contracts import ArmPull, K_EPOCH
    from loop.driver import K_LAST_CONFIRM
    from loop.labless import EXP, K_AUDIT, K_LABLESS
    from loop.world.noise import MEASURE_KEY
    ctx = _world(tmp_path, review_every=1000)
    driver, store, adapter = _make(ctx, Scripts())
    store.ensure_root("c0")
    for nid, parent, commit, st in (("a", ROOT_NODE_ID, "ca", NodeStatus.CONFIRMED),
                                    ("b", "a", "cb", NodeStatus.NOT_CONFIRMED),
                                    ("x", ROOT_NODE_ID, "cb", NodeStatus.CONFIRMED)):      # x: b's commit re-rooted
        store.put_node(dataclasses.replace(_node(st, parent=parent, nid=nid), task_id=ctx.task_id, commit=commit,
                                           created_ts=float(len(store.nodes()))))
    adapter.ancestry = lambda commit, base: {("cb", "c0"): ["cb", "ca"], ("ca", "c0"): ["ca"],
                                             ("cb", "ca"): ["cb"]}[(commit, base)]
    driver._measure()
    assert store.get_state(MEASURE_KEY) == {"a": [], "b": [], "x": ["b", "a"]}
    for e in ({"node_id": "a", "to": "confirmed"}, {"node_id": "q", "to": "screened_out"},
              {"node_id": "x", "to": "confirmed"}):
        store.log("node_transition", **e)
    store.set_state(K_LAST_CONFIRM, 2)
    store.put_run(dataclasses.replace(_run("x", Stage.AUDIT, 201, RunStatus.RUNNING), task_id=ctx.task_id, job_id="9"))
    store.set_state(K_AUDIT, {"passed": [], "failed": [], "queue": ["x"], "current": {"node": "x"}})
    store.set_state(K_LABLESS, {"x" + EXP: {"tries": 1}, "a": {"submission_id": "s1"}})
    store.record_pull(ArmPull(ctx.task_id, Strategy.CROSSOVER, "x", 0.12, 1.0, 0.0))
    from types import SimpleNamespace
    driver.snapshot = SimpleNamespace(measured={"x": ("b", 0.01, 0.0)}, nodes={}, sigma={PANEL: 0.003})
    driver.world = SimpleNamespace(confirm_evalue=lambda snap, nid, st: (0.4, 1))
    adapter.cancel = lambda handle: None
    acts: list[str] = []
    driver._remeasure(acts)
    assert store.get_node("x").status is NodeStatus.NOT_CONFIRMED and "duplicate of b" in acts[0]
    assert store.get_node("a").status is NodeStatus.CONFIRMED
    assert store.get_run("x:audit:201").void_type is VoidType.CANCELLED
    a = store.get_state(K_AUDIT)
    assert a["queue"] == [] and a["current"] is None
    assert store.get_state(K_LABLESS) == {"a": {"submission_id": "s1"}}
    assert [p.node_id for p in store.pulls()] == [] and store.get_state(K_LAST_CONFIRM) == 1   # at a's confirm
    assert store.get_state(K_EPOCH)["epoch"] == 1


def test_measurement_epoch_withdraws_playbook_and_overrides_built_on_the_old_numbers(tmp_path):
    from types import SimpleNamespace
    from loop import meta as ML, selftune as T
    from loop.contracts import K_EPOCH, current_reviews
    from loop.driver import K_FP_FORCE
    ctx = _world(tmp_path, review_every=1000)
    driver, store, _ = _make(ctx, Scripts())
    ML.save_playbook(store, {"version": 2, "ts": 0.0, "reviews_at": 0, "rules": [{"component": "researcher",
                     "rule": "sample JEPA only on the n0255 carrier", "evidence": "1"}], "retired": [], "dropped": []})
    store.set_state(T.K_OVERRIDES, {"meta.crossover_bias": {"value": 0.05, "spec": 0.0, "reason": "n0308 clears the bar"},
                                    "bandit.floor": {"value": 0.08, "spec": 0.1, "reason": "n0255 is real"}})
    post = lambda m: SimpleNamespace(increment=Posterior(m, 0.001, 2))      # noqa: E731
    driver.snapshot = SimpleNamespace(nodes={"n0308": post(0.0027), "n0255": post(0.0064)}, sigma={PANEL: 0.00326})
    acts: list[str] = []
    driver._epoch(acts)
    assert acts == []                                                      # no bump: nothing to do
    ML.bump_epoch(store, "test", now=driver.clock())
    driver._epoch(acts)
    assert ML.playbook(store)["rules"] == [] and ML.playbook(store)["version"] == 3
    book = store.get_state(T.K_OVERRIDES)                                   # kept as history, but expired:
    assert set(book) == {"bandit.floor", "meta.crossover_bias"} and T.live(ctx.cfg, book, T.epoch_ts(store)) == {}
    assert T.effective(ctx.cfg, store).bandit.floor == ctx.cfg.bandit.floor    # reviews re-tune on the new numbers
    store.set_state(T.K_OVERRIDES, {**book, "bandit.floor": {**book["bandit.floor"], "ts": driver.clock() + 1}})
    assert T.effective(ctx.cfg, store).bandit.floor == 0.08                    # tuned after the bump: live
    assert store.get_state(K_FP_FORCE) is True and driver._fp_due()
    assert store.get_state(K_EPOCH)["handled"] == 1 and current_reviews(store) == []



@pytest.mark.parametrize("final,verdict", [(0.5005, "passed"), (0.503, "failed")])
def test_resume_calibration_interrupts_a_root_rerun_and_turns_resume_on_only_within_tol(ctx, store, final, verdict):
    """slurm.resume=calibrate: root re-run (stage AUDIT, outside the world model) at a finished baseline seed,
    interrupted past FRAC and resumed; resume_ok iff |final_score delta| <= resume_tol; else it stays off."""
    from loop import resume as RS
    from loop.driver import RunBook

    class _Resumable(_PoolAdapter):
        def __init__(self):
            super().__init__()
            self.resume_ok, self.resume_runs, self.hits = False, set(), []
            self.state = {"armed": False, "resumed": 0, "fraction": 0.2}

        def resume_state(self, handle):
            return dict(self.state)

        def interrupt(self, handle):
            self.hits.append(handle.run_id)
            self.state["resumed"] += 1
            return "99"

    c, adapter = _pools_ctx(ctx, primary=1, idle=2, resume="calibrate"), _Resumable()
    book = RunBook(c, store, adapter)
    store.ensure_root("c0")
    store.put_run(dataclasses.replace(_run(ROOT_NODE_ID, Stage.BASELINE, 11), task_id=store.task_id, aggregate=0.5))
    acts = RS.tick(c, store, book, adapter)
    run = store.get_run("root:audit:11")
    assert run.pool == "idle" and run.status is RunStatus.RUNNING and adapter.resume_runs == {run.id}, acts
    assert RS.tick(c, store, book, adapter) == [] and not adapter.resume_ok        # not armed / not past FRAC
    adapter.state.update(armed=True, fraction=0.6)
    assert "interrupted" in RS.tick(c, store, book, adapter)[0] and adapter.hits == [run.id]
    assert RS.tick(c, store, book, adapter) == [] and adapter.hits == [run.id]       # once
    store.update_run(run.id, status=RunStatus.DONE, aggregate=final)
    acts = RS.tick(c, store, book, adapter)
    assert store.get_state(RS.K_RESUME)["status"] == verdict, acts
    RS.tick(c, store, book, adapter)
    assert adapter.resume_ok is (verdict == "passed")
    off = dataclasses.replace(c, cfg=dataclasses.replace(c.cfg, slurm=dataclasses.replace(c.cfg.slurm, resume="off")))
    assert RS.tick(off, store, book, adapter) == [] and not adapter.resume_ok and adapter.resume_runs == set()


def test_resume_on_records_resumed_runs_and_reports_their_share(ctx, store):
    """slurm.resume=on (the default): no calibration, every run resumable; a poll that reports resumed starts
    marks the run row (the world model gives it seed-level noise) and the heartbeat shows the resumed share."""
    from loop import resume as RS
    from loop.contracts import JobStatus
    from loop.driver import RunBook, hb_line
    c = _pools_ctx(ctx, primary=1, idle=2)
    assert c.cfg.slurm.resume == "on"
    adapter = _PoolAdapter()
    adapter.resume_ok, adapter.resume_runs, adapter.interrupt = False, set(), lambda h: "x"
    book = RunBook(c, store, adapter)
    assert RS.tick(c, store, book, adapter) == [] and adapter.resume_ok and store.get_state(RS.K_RESUME) is None
    run = book.launch(_pnode(store, "n1"), Stage.SCREEN, 1)
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING, detail="RUNNING", resumed=1)
    book.poll()
    assert store.get_run(run.id).resumed == 1
    adapter.poll = lambda h: JobStatus(RunStatus.DONE, scores={"a": 0.5}, aggregate=0.5, resumed=2)
    book.poll()
    assert store.get_run(run.id).resumed == 2 and "resumed=1/1 done" in hb_line(c, store, "loop")


def test_stall_review_waits_stall_min_hours_since_the_last_review(tmp_path):
    """rv-34 / rv-35: two stall reviews 1.5 h apart. A stall needs stall_evals closes AND stall_min_hours since the
    last review (its marker's ts; an old marker falls back to the newest stored review)."""
    from loop.driver import K_REVIEW
    ctx = _world(tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, review=ReviewCfg(
        stall_evals=2, stall_min_hours=4.0)))
    driver, store, _ = _make(ctx, Scripts())
    store.ensure_root("base")
    for i in range(3):
        store.put_node(dataclasses.replace(_node(NodeStatus.SCREENED_OUT, nid=f"s{i}"), task_id=ctx.task_id))
    store.set_state(K_REVIEW, {"evals": 0, "confirmations": 0, "ts": 1000.0})
    driver.clock = lambda: 1000.0 + 3600
    assert driver._review_due() == ""
    driver.clock = lambda: 1000.0 + 5 * 3600
    assert driver._review_due() == "stall"


def test_a_smoke_queue_timeout_reissues_the_request_with_no_node_and_no_reward(tmp_path):
    """Live n0521-n0526: a smoke that never got a slot was rejected smoke_failed (+0 reward). Now the request is
    re-issued (same id, the previous diff in its brief) up to MAX_SMOKE_REISSUES times; no node is created."""
    from loop.contracts import Candidate, HandoffNote, HypothesisRecord, LeverTags, ResearchResult
    from loop.driver import MAX_SMOKE_REISSUES, REISSUE_MARK
    from loop.spec import ResearcherCfg
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    ctx = _world(tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, researcher=ResearcherCfg(max_concurrent=1)))
    driver, store = build(ctx, llm=Scripts().client(), adapter=SimAdapter(ctx), executor=_Pending,
                          rng=random.Random(7))
    driver.tick()
    (fut, req), = driver._research.items()
    h = HypothesisRecord(id="h-x", task_id=ctx.task_id, kind=HypothesisKind.TRIAL, claim="the claim",
                         mechanism_tag="m", family="optim", predictions={}, falsification="", created_ts=0.0,
                         levers=LeverTags({}))
    cand = Candidate(req.id, ctx.task_id, "", "base", "+DIFF LINE", "dh", ("a.py",), {}, h,
                     HandoffNote("ho", ctx.task_id, req.id, "", "", "", 0.0), None)
    nodes = len(store.nodes())
    for k in range(1, MAX_SMOKE_REISSUES + 1):
        fut.set_result(ResearchResult("backend_unavailable", cand, "smoke_unavailable: queue timeout"))
        driver._harvest(acts := [])
        assert any("re-issued" in a for a in acts), acts
        (fut, again), = driver._research.items()
        assert again.id == req.id and again.brief.count(REISSUE_MARK) == 1 and "+DIFF LINE" in again.brief
        assert again.brief.startswith(req.brief) and len(store.nodes()) == nodes
    fut.set_result(ResearchResult("backend_unavailable", cand, "smoke_unavailable: queue timeout"))
    driver._harvest(acts := [])
    assert not driver._research and len(store.nodes()) == nodes and "produced nothing" in acts[0]


def test_no_researcher_starts_while_every_pool_is_full_with_our_own_runs_pending(tmp_path):
    """Live 2026-10-02: 15 researchers' smokes queued an hour each behind 23 of our own pending runs. While
    RunBook.saturated() no new researcher starts (logged once); the next fill with a free slot resumes."""
    from loop.spec import ResearcherCfg
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    ctx = _world(tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, researcher=ResearcherCfg(max_concurrent=1)))
    driver, store = build(ctx, llm=Scripts().client(), adapter=SimAdapter(ctx), executor=_Pending,
                          rng=random.Random(7))
    driver.book.saturated = lambda: True
    driver.tick()
    driver.tick()
    assert not driver._research and len(store.events("research_paused")) == 1
    driver.book.saturated = lambda: False
    driver.tick()
    assert len(driver._research) == 1 and len(store.events("research_resumed")) == 1


def test_saturated_needs_no_free_slot_and_every_pool_backed_up(ctx, store):
    from loop.driver import RunBook
    ctx = _pools_ctx(ctx, primary=1, idle=1)
    book = RunBook(ctx, store, _PoolAdapter())
    assert not book.saturated()                                   # before the first poll: unknown
    book.free = lambda: 0
    book.pool_states = {"primary": (1, 4), "idle": (0, 2)}        # idle runs none of ours while 2 wait
    assert book.saturated()
    book.pool_states = {"primary": (1, 3), "idle": (0, 2)}        # primary: 3 < pause_pending
    assert not book.saturated()
    book.pool_states = {"primary": (1, 4), "idle": (1, 2)}        # idle still runs ours
    assert not book.saturated()
    book.pool_states, book.free = {"primary": (1, 4), "idle": (0, 2)}, lambda: 1
    assert not book.saturated()                                   # a free slot: a smoke can start


def test_nodes_rejected_on_a_smoke_queue_timeout_are_reissued_once_and_never_paid(tmp_path):
    """The one-time repair: a smoke_failed node without a commit whose smoke dir has no out/ (the job never
    started) is relabelled smoke_unavailable, its bandit credit dropped, and its request re-issued; a node whose
    smoke ran (out/ exists) is left alone."""
    from loop.contracts import ArmPull, Strategy, Operator
    from loop.driver import K_SMOKE_REPAIR, REISSUE_MARK
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    ctx = dataclasses.replace(_world(tmp_path), state_dir=str(tmp_path / "st"))
    driver, store = build(ctx, llm=Scripts().client(), adapter=SimAdapter(ctx), executor=_Pending,
                          rng=random.Random(7))
    store.ensure_root("base")
    for nid, req, ran in (("n7", "req-7", False), ("n8", "req-8", True)):
        d = Path(ctx.state_dir) / "smoke" / f"loop-{req}"
        d.mkdir(parents=True)
        (d / "smoke.yaml").write_text("x: 1\n")
        if ran:
            (d / "out").mkdir()
        store.put_node(dataclasses.replace(_node(NodeStatus.REJECTED, nid=nid), task_id=ctx.task_id, commit="", strategy=Strategy.EXPLORE,
                                           operator=Operator.DRAFT, reject_reason="smoke_failed",
                                           notes=f"request={req} brief=failure backend=x web=0"))
        store.record_pull(ArmPull(task_id=ctx.task_id, arm=Strategy.EXPLORE, node_id=nid, reward=0.0, gpu_h=0.0, ts=1.0))
    acts: list[str] = []
    driver._repair_smoke_timeouts(acts)
    assert store.get_node("n7").reject_reason == "smoke_unavailable" and store.get_node("n8").reject_reason == "smoke_failed"
    assert [p.node_id for p in store.pulls()] == ["n8"] and store.get_state(K_SMOKE_REPAIR)
    (req,), = [[r for r in driver._research.values() if r.id == "req-7"]]
    assert REISSUE_MARK in req.brief and req.strategy is Strategy.EXPLORE
    driver._research.clear()
    driver._repair_smoke_timeouts(acts)                 # once only
    assert not driver._research
    driver.tick()
    assert "n7" not in {p.node_id for p in store.pulls()}  # never paid again


def test_a_finished_researchers_queued_smoke_jobs_are_cancelled(tmp_path):
    """Live: smoke jobs whose researcher had ended started later on the reaped worktree (`.venv/bin/activate: No
    such file`) and wasted a slot; cancel_smokes ran only on restart. Now every harvested request's smokes go."""
    from loop.contracts import ResearchResult
    from loop.spec import ResearcherCfg
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    ctx = _world(tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, researcher=ResearcherCfg(max_concurrent=1)))
    adapter, cancelled = SimAdapter(ctx), []
    adapter.cancel_smokes = lambda name="": (cancelled.append(name), ["99"])[1]
    driver, store = build(ctx, llm=Scripts().client(), adapter=adapter, executor=_Pending, rng=random.Random(7))
    driver.tick()
    (fut, req), = driver._research.items()
    fut.set_result(ResearchResult("failed", None, "boom"))
    driver._harvest(acts := [])
    assert cancelled == ["", f"loop-{req.id}"] and any("scancelled smoke job(s) 99" in a for a in acts)
    driver.tick()                                      # the orphan sweep runs at start, then every SMOKE_SWEEP_S
    assert cancelled.count("") == 1


def test_orphaned_smoke_jobs_are_scancelled_at_start_and_every_sweep(tmp_path):
    """Live job 509746 (tune-t10-5's smoke): preempted after training, requeued under the same id, it pended 5 h on
    the starved idle pool owned by no thread; a restart cancelled only in-flight requests' smokes. Every smoke job
    no live smoke() call waits on is scancelled at start (deploy) and every SMOKE_SWEEP_S."""
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    from loop.driver import SMOKE_SWEEP_S
    ctx, t = _world(tmp_path), [1000.0]
    adapter, cancelled = SimAdapter(ctx), []
    adapter.cancel_smokes = lambda name="": (cancelled.append(name), ["509746"] if not name else [])[1]
    driver, store = build(ctx, llm=Scripts().client(), adapter=adapter, executor=_Pending, rng=random.Random(7))
    driver.clock = lambda: t[0]
    acts = driver.tick()
    assert cancelled.count("") == 1 and any("orphaned smoke job(s) 509746" in a for a in acts)
    t[0] += SMOKE_SWEEP_S + 1
    driver.tick()
    assert cancelled.count("") == 2


def test_a_starved_idle_pool_takes_no_new_jobs_and_its_low_screens_move_to_primary_controls_done_first(ctx, store):
    """Live 2026-10-01: 41 of 43 jobs PENDING; idle (low QoS) ran none. After STARVE_S with 0 running there, idle
    keeps at most STARVED_PENDING pending, and its LOW screens move to primary as slots open, the one whose block
    control is already done first (its verdict is one run away)."""
    from loop.contracts import Run
    from loop.driver import K_STARVED, STARVE_S, STARVED_PENDING, RunBook, pool_use
    ctx = _pools_ctx(ctx, primary=1, idle=8, max_pending_per_pool=0)      # the starved cut alone
    adapter, t = _PoolAdapter(), [1000.0]
    book = RunBook(ctx, store, adapter, clock=lambda: t[0])
    root, a, b = store.ensure_root("c0"), _pnode(store, "n1"), _pnode(store, "n2")
    assert book.launch(root, Stage.BASELINE, 9).pool == "primary"
    assert book.launch(a, Stage.SCREEN, 1).pool == "idle" and book.launch(b, Stage.SCREEN, 9).pool == "idle"
    book.poll()                                                    # every job PENDING
    assert book.pool_states == {"primary": (0, 1), "idle": (0, 2)} and store.get_state(K_STARVED) == {"idle": 1000.0}
    assert not book.starved("idle") and book.limits()["idle"] == 8
    assert "(0 running, 2 pending)" in pool_use(ctx, store.runs(), book.pending)
    t[0] += STARVE_S + 1
    assert book.starved("idle") and book.limits()["idle"] == STARVED_PENDING
    store.put_run(Run(id="root:baseline:5", task_id=store.task_id, node_id=ROOT_NODE_ID, seed=5,
                      stage=Stage.BASELINE, status=RunStatus.DONE))
    store.update_run("root:baseline:9", status=RunStatus.DONE)    # a primary slot opens; n2's control (seed 9) done
    acts = book.promote_stuck()
    assert adapter.moved == [("n2:screen:9", "primary")] and "(LOW" in acts[0]
    assert book.promote_stuck() == [] and store.get_run("n1:screen:1").pool == "idle"   # primary full, its queue too


def test_control_runs_hold_at_most_half_of_primary_while_candidate_runs_wait(ctx, store):
    """Live: primary held 8/8 parent/baseline controls while every candidate screen sat PENDING on idle. While a
    candidate run waits PENDING elsewhere, controls take at most half of primary; the rest is the candidates'."""
    from loop.driver import RunBook
    ctx = _pools_ctx(ctx, primary=4, idle=2)
    adapter = _PoolAdapter()
    book = RunBook(ctx, store, adapter)
    book.p_positive = lambda nid: 0.9                                   # HIGH screens (they still overflow)
    root, parent, a = store.ensure_root("c0"), _pnode(store, "n1"), _pnode(store, "n2")
    assert [book.launch(root, Stage.BASELINE, s).pool for s in (1, 2)] == ["primary", "primary"]
    assert book.pool(parent, Stage.PARENT, 3) == "primary"              # no candidate waits: no reserve
    book.launch(root, Stage.BASELINE, 4)
    book.launch(root, Stage.BASELINE, 5)                                # primary full of controls
    assert book.launch(a, Stage.SCREEN, 1).pool == "idle" and book.launch(a, Stage.SCREEN, 2).pool == "idle"
    book.poll()                                                         # the candidate screens wait PENDING
    assert book.controls_capped("primary") and not book.controls_capped("idle")
    assert book.pool(parent, Stage.PARENT, 3) == ""                     # idle full: the control waits
    store.update_run("n2:screen:2", status=RunStatus.DONE)
    assert book.pool(parent, Stage.PARENT, 3) == "idle"                 # a control may still use idle
    store.update_run("root:baseline:5", status=RunStatus.DONE)          # a primary slot opens: not for a control
    assert book.pool(parent, Stage.PARENT, 3) == "idle"
    acts = book.promote_stuck()                                         # the candidate takes it
    assert ("n2:screen:1", "primary") in adapter.moved and len(acts) == 1


def test_a_tuning_trial_pending_on_a_starved_pool_is_never_moved_to_primary(ctx, store):
    """Live 2026-10-02: tuning trials n0554 (t10) / n0555 (t12) were repooled from the starved idle pool onto primary:
    the low-screen move skipped only designed-batch points. A tuning pass's trials stay preemptible-only."""
    from loop.contracts import Strategy
    from loop.driver import STARVE_S, RunBook
    ctx = _pools_ctx(ctx, primary=2, idle=8, max_pending_per_pool=0)
    adapter, t = _PoolAdapter(), [1000.0]
    book = RunBook(ctx, store, adapter, clock=lambda: t[0])
    store.ensure_root("c0")
    _pnode(store, "n1", Strategy.DESIGN)
    store.update_node("n1", operator=None, notes="request=tune-t10-4 tune=t10")
    plain = _pnode(store, "n2")
    assert book.launch(store.get_node("n1"), Stage.SCREEN, 1).pool == "idle"
    assert book.launch(plain, Stage.SCREEN, 1).pool == "idle"
    book.poll()
    t[0] += STARVE_S + 1
    book.poll()
    assert book.starved("idle")
    book.promote_stuck()
    assert adapter.moved == [("n2:screen:1", "primary")] and store.get_run("n1:screen:1").pool == "idle"


def test_scarce_mode_follows_the_2h_mean_of_running_gpus_and_pauses_design_points(ctx, store):
    """Live 2026-10-02: ~2.6 GPUs effective over 12 h, 20% of them on tune / design points. SCARCE MODE is on while the
    GPUs SLURM runs for us average < slurm.scarce_gpus over 2 h (or a preemptible pool is STARVED): logged
    scarce_on / scarce_off, shown in the heartbeat; a designed batch's points then wait (never launched)."""
    from loop.contracts import JobStatus, Strategy
    from loop.driver import K_SCARCE, SCARCE_SAMPLE_S, SCARCE_WINDOW_S, RunBook, scarce_note
    ctx = _pools_ctx(ctx, primary=4, idle=4, scarce_gpus=2)
    adapter, t = _PoolAdapter(), [1000.0]
    book = RunBook(ctx, store, adapter, clock=lambda: t[0])
    store.ensure_root("c0")
    point = _pnode(store, "n9", Strategy.DESIGN)
    store.update_node("n9", operator=None, mechanism_tag="d1pt01")
    a = _pnode(store, "n1")
    for s in (1, 2, 3):
        book.launch(a, Stage.SCREEN, s)
    book.poll()                                                   # all PENDING: 0 running
    assert book.scarce() and [e["event"] for e in store.events() if e["event"].startswith("scarce")] == ["scarce_on"]
    assert any("scarce mode ON" in x for x in book.acts) and scarce_note(store).startswith("scarce=on(0.0 GPUs")
    assert book.pool(store.get_node("n9"), Stage.SCREEN, 1) == ""   # a design point waits
    adapter.poll = lambda h: JobStatus(RunStatus.RUNNING)          # SLURM runs all 3
    t[0] += SCARCE_SAMPLE_S
    book.poll()
    assert book.scarce() and store.get_state(K_SCARCE)["avg"] == 1.5       # (0 + 3) / 2 < 2
    t[0] += 10
    book.poll()
    assert len(store.get_state(K_SCARCE)["samples"]) == 2                   # at most one sample per SCARCE_SAMPLE_S
    t[0] += SCARCE_WINDOW_S                                       # the 0-sample ages out
    book.poll()
    assert not book.scarce() and scarce_note(store) == ""
    assert [e["event"] for e in store.events() if e["event"].startswith("scarce")] == ["scarce_on", "scarce_off"]
    assert book.pool(store.get_node("n9"), Stage.SCREEN, 1) == "idle"


def test_scarce_mode_starts_no_design_batch_and_no_tuning_pass(ctx, store, monkeypatch):
    from loop import design as D, tune as MT
    store.set_state(D.K_DESIGN, {"request": ["review asks"]})
    assert D.tick(ctx, store, None, None, evals=0, evals_at_confirm=0, new_id=None, paused=True) == []
    assert store.get_state(D.K_DESIGN)["request"] == ["review asks"]           # it waits, not dropped
    monkeypatch.setattr(MT, "eligible", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no new pass")))
    assert MT.tick(ctx, store, None, object(), new_id=None, now=0.0, scarce=True) == []


def test_a_control_pairing_a_confirming_childs_run_is_placed_like_that_high_run(ctx, store):
    """Controls of HIGH runs (a confirming child's) take primary like their children: the controls' half-of-primary
    cap never holds them back (live 2026-10-02: 0 GPU-h reached confirm in 12 h)."""
    from loop.driver import RunBook
    ctx = _pools_ctx(ctx, primary=4, idle=2)
    book = RunBook(ctx, store, _PoolAdapter())
    book.p_positive = lambda nid: 0.9                               # HIGH screens
    root, parent, a = store.ensure_root("c0"), _pnode(store, "n1"), _pnode(store, "n2")
    for s in (1, 2):
        book.launch(root, Stage.BASELINE, s)
    assert book.launch(a, Stage.SCREEN, 5).pool == "primary" and book.launch(a, Stage.SCREEN, 6).pool == "primary"
    book.launch(a, Stage.SCREEN, 7)
    book.launch(a, Stage.SCREEN, 8)
    book.poll()                                                     # candidate screens wait PENDING
    store.update_run("n2:screen:5", status=RunStatus.DONE)          # one primary slot free
    assert book.controls_capped("primary")
    seed = int(ctx.cfg.seeds.confirm_pool[-1])
    assert book.pool(parent, Stage.PARENT, seed) == ""              # no confirming child: held to the half
    _pnode(store, "n3")
    store.update_node("n3", parent_id="n1", status=NodeStatus.CONFIRMING)
    assert book.serves_high(parent, seed) and not book.serves_high(parent, 999)
    assert book.pool(parent, Stage.PARENT, seed) == "primary"


def test_an_account_refusal_pauses_every_new_llm_job_until_its_reset_then_probes_with_backoff(tmp_path):
    """Live 2026-10-03 22:00Z: the weekly limit refused every Opus session for 2 h while the loop kept issuing
    ~1500 requests. A refusal (loop.agents.llm.note_outage) now pauses all new LLM work until resets_at (+1 min);
    with no known reset 15 min, doubling while the resumed loop is refused again. Status / heartbeat show it."""
    from loop.agents import llm as LA
    from loop.driver import K_LLM_PAUSE, LLM_PROBE_S, hb_line
    from loop.spec import ResearcherCfg
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    ctx = _world(tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, researcher=ResearcherCfg(max_concurrent=1)))
    driver, store = build(ctx, llm=Scripts().client(), adapter=SimAdapter(ctx), executor=_Pending,
                          rng=random.Random(7))
    now = [1.0e9]
    driver.clock = lambda: now[0]
    LA.note_outage("quota", now[0] + 7200)
    acts = driver.tick()
    assert not driver._research and any(a.startswith("llm_paused quota until") for a in acts), acts
    assert store.get_state(K_LLM_PAUSE)["until"] == now[0] + 7260 and "PAUSED:quota" in hb_line(ctx, store, "loop")
    now[0] += 7200
    driver.tick()
    assert not driver._research and not store.events("llm_resumed")
    now[0] += 61
    acts = driver.tick()
    assert any(a.startswith("llm_resumed") for a in acts) and len(driver._research) == 1, acts
    LA.note_outage("auth", None)                                   # the resumed loop is refused again: probe x2
    driver.tick()
    assert store.get_state(K_LLM_PAUSE)["until"] == now[0] + 2 * LLM_PROBE_S
    assert store.get_state(K_LLM_PAUSE)["probes"] == 1 and len(store.events("llm_paused")) == 2


def test_an_sdk_failure_storm_pauses_then_one_probe_researcher_until_a_call_works(tmp_path):
    """The circuit breaker (loop.agents.llm.note_failure) trips on 'error result: success'; after the pause ONE
    researcher probes (no review / scout / other source) until a harvested result shows the backend works."""
    from loop.agents import llm as LA
    from loop.driver import K_LLM_PAUSE, LLM_PROBE_S
    from loop.spec import ResearcherCfg
    from loop.__main__ import build
    from loop.adapters.sim import SimAdapter
    ctx = _world(tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, researcher=ResearcherCfg(max_concurrent=4)))
    driver, store = build(ctx, llm=Scripts().client(), adapter=SimAdapter(ctx), executor=_Pending,
                          rng=random.Random(7))
    now = [1.0e9]
    driver.clock = lambda: now[0]
    LA.take_outage()
    assert LA.note_failure("RuntimeError: sdk_session other: Claude Code returned an error result: success")
    acts = driver.tick()
    assert not driver._research and any(a.startswith("llm_paused sdk until") for a in acts), acts
    now[0] += LLM_PROBE_S + 1
    acts = driver.tick()
    assert any(a.startswith("llm_resumed") for a in acts) and store.get_state(K_LLM_PAUSE)["probing"]
    assert len(driver._research) == 1 and any("LLM probe after a pause" in a for a in acts), acts
    driver.tick()
    assert len(driver._research) == 1 and driver._review is None            # still probing: nothing more
    (fut,) = list(driver._research)
    from loop.contracts import ResearchResult
    fut.set_result(ResearchResult("failed", None, "boom"))                 # an answer, not an outage
    driver.tick()
    assert not store.get_state(K_LLM_PAUSE)["probing"] and store.events("llm_probe_ok")
    assert len(driver._research) > 1                                         # back to normal


def test_the_outage_repair_reopens_ladders_whose_exposures_only_failed_on_outages_once(tmp_path):
    """One-time repair of 2026-10-03: t18/t19/t21/t22 closed 'knob exposure failed 3x' on backend_unavailable
    requests only -> re-opened with fails 0; a real failure keeps its ladder closed; the probes they served return."""
    import types as _t
    from loop import tune as MT
    from loop.driver import Driver, K_OUTAGE_REPAIR
    from loop.select import PortfolioSelector
    ctx = _world(tmp_path)
    driver, store, _ = _make(ctx, Scripts())
    lad = lambda b, why, fails: {"batch": b, "base": f"n{b:04d}", "kind": "ladder", "trials": [], "fails": fails,   # noqa: E731
                                 "pending": "", "expose": "done", "knob": "", "why": why, "closed": 1.0, "tuned": ""}
    store.set_state(MT.K_TUNE, {"seq": 3, "current": {}, "tuned": [], "laddered": [], "batches": [
        lad(1, "knob exposure failed 2x (no candidate)", 2), lad(2, "knob exposure failed 2x (no candidate)", 2)]})
    for rid, st in (("expose-t1-1", "backend_unavailable"), ("expose-t1-2", "backend_unavailable"),
                    ("expose-t2-1", "backend_unavailable"), ("expose-t2-2", "failed"), ("req-7", "backend_unavailable")):
        store.log("research_started", request_id=rid)
        store.log("research_failed", request_id=rid, status=st)
    store.set_state("select.probes_used", ["rv-1:0", "rv-1:1"])
    store.log("meta_decision", component="review_probe", decision_id="rv-1:0", payload={"request": "req-7"})
    fake = _t.SimpleNamespace(store=store, selector=PortfolioSelector, clock=lambda: 5.0, _dirty=False)
    Driver._repair_outage(fake, acts := [])
    bk = MT.book(store)
    assert set(bk["current"]) == {"1"} and [b["batch"] for b in bk["batches"]] == [2], acts
    cur = bk["current"]["1"]
    assert cur["expose"] == "want" and cur["fails"] == 0 and "why" not in cur and "closed" not in cur
    assert store.get_state("select.probes_used") == ["rv-1:1"] and store.get_state(K_OUTAGE_REPAIR)["fixed"]
    Driver._repair_outage(fake, acts := [])
    assert acts == [] and len(store.events("outage_repair")) == 1


def test_research_starts_count_in_flight_supply_and_stagger_across_ticks():
    from loop.driver import research_starts
    # idle cluster (64 free -> target 32 nodes), 16 researchers allowed: 4 per tick, not 16 in one burst
    starts, flight = [], 0
    for _ in range(6):
        n = research_starts(0, flight, 32, 48, 16)
        starts.append(n)
        flight += n
    assert starts == [4, 4, 4, 4, 0, 0] and flight == 16
    # in-flight researchers are future supply: 3 in flight already cover a target of 2
    assert research_starts(0, 3, 2, 18, 16) == 0 and research_starts(0, 2, 2, 18, 16) == 1
    # a non-empty queue below target still gets researchers (the old low-water rule waited for it to drain)
    assert research_starts(6, 8, 16, 32, 16) == 4
    assert research_starts(10, 8, 16, 32, 16) == 0                   # 10 + 6 >= 16
    assert research_starts(0, 0, 2, 18, 1) == 1                       # max_concurrent 1: one at a time


def test_screen_futility_under_eb_is_relative_to_the_generations_prior():
    """Under the EB mixture a carrier child's PRIOR P(gain>0) is ~0.03 (pi0 0.90): the absolute 0.1 bar would end
    every screen at its first seed (live: confirmed n0655 / n0764 sat at 0.06 / 0.07 after 2 seeds). Futile is
    screen_futility_p x the prior: the seed made a real gain 10x less likely than before it ran."""
    own = [_run("n1", Stage.SCREEN, 11), _run("n1", Stage.SCREEN, 12, RunStatus.RUNNING)]
    post = lambda p, prior: dataclasses.replace(_post(inc=-0.004), gain_p=p, gain_prior=prior)   # noqa: E731
    assert _step(_node(NodeStatus.SCREENING), own, _roots([11, 12]), post(0.02, 0.03)).status is None
    d = _step(_node(NodeStatus.SCREENING), own, _roots([11, 12]), post(0.002, 0.03))
    assert d.status is NodeStatus.SCREENED_OUT and "< 0.003" in d.reason


def test_value_of_information_gate_promotes_a_valuable_screen_below_the_p_bar():
    """promotion.voi_gain: a screen whose P(gain>0) is under p_promote still promotes when its EB posterior's
    E[max(gain, 0)] >= voi_gain (and P(U>0) >= p_promote); off (0) it is only the P bar."""
    cfg = _cfg(seeds=SeedPlan(promote_stage=False, max_seeds_per_node=5),
               promotion=PromotionCfg(p_promote=0.3, confirm_max_seeds=3, voi_gain=0.0005))
    own = [_run("n1", Stage.SCREEN, 11), _run("n1", Stage.SCREEN, 12)]
    post = lambda plus, p=0.2: dataclasses.replace(_post(), gain_p=p, gain_prior=0.03, gain_plus=plus)   # noqa: E731
    step = lambda p, c=cfg: S.step(_node(NodeStatus.SCREENING), own_runs=own, control_runs=_roots([11, 12]),   # noqa: E731
                                   post=p, cfg=c, sigma_panel=0.004, infra_retry=2, raw=0.004)
    assert step(post(0.0006)).status is NodeStatus.CONFIRMING
    assert "E[gain+]=0.0006" in step(post(0.0006)).reason
    assert step(post(0.0004)).status is NodeStatus.SCREENED_OUT
    assert step(post(0.0006), _cfg(seeds=cfg.seeds, promotion=PromotionCfg(p_promote=0.3))).status is NodeStatus.SCREENED_OUT
    assert S.promote_ok(dataclasses.replace(post(0.0006), p_positive=0.1), cfg.promotion) is False


def test_early_confirmation_replaces_the_screen_extension():
    """EARLY CONFIRMATION: a done 2-seed screen below the gate with raw mean > 0, P(gain>0) >= futile and a Bayes
    factor for a gain > 1 goes to CONFIRMING on 3 fresh confirm seeds (not 2 more screen seeds and 1 confirm seed:
    n1030); a node whose selection seeds already leave < 3 confirm seeds under the 5-seed cap stays screened out."""
    cfg = _cfg(seeds=SeedPlan(promote=(13, 15, 16, 17), promote_stage=False, max_seeds_per_node=5),
               promotion=PromotionCfg(p_promote=0.3, confirm_max_seeds=3, early_confirm=True))
    own = [_run("n1", Stage.SCREEN, 11), _run("n1", Stage.SCREEN, 12)]
    post = lambda p, prior=0.03, **kw: dataclasses.replace(_post(**kw), gain_p=p, gain_prior=prior)   # noqa: E731
    step = lambda own_, p, raw=0.004, c=cfg: S.step(   # noqa: E731
        _node(NodeStatus.SCREENING), own_runs=own_, control_runs=_roots([11, 12, 13, 15]), post=p, cfg=c,
        sigma_panel=0.004, infra_retry=2, raw=raw)
    d = step(own, post(0.08))
    assert d.status is NodeStatus.CONFIRMING and "early confirmation" in d.reason and d.fields["confirm_alt"] > 0
    assert f"BF={S.bayes_factor(0.08, 0.03):.2f}" in d.reason and S.planned_confirm(cfg, own) == 3
    for p, raw, c in ((post(0.08), -0.001, cfg), (post(0.02), 0.004, cfg),          # raw <= 0; BF < 1
                      (post(0.08), 0.004, _cfg(promotion=PromotionCfg(p_promote=0.3))),   # early confirmation off
                      (post(0.08, prior=None), 0.004, cfg)):                       # no EB prior
        assert step(own, p, raw, c=c).status is NodeStatus.SCREENED_OUT
    assert step(own, post(0.5)).status is NodeStatus.CONFIRMING and "early" not in step(own, post(0.5)).reason
    four = own + [_run("n1", Stage.SCREEN, 13), _run("n1", Stage.SCREEN, 15)]    # a legacy extension: 1 seed left
    assert S.planned_confirm(cfg, four) == 1 and S.early_confirm_bf(post(0.08), cfg, 0.004, four) is None


def test_early_confirmation_needs_a_realistic_chance_to_pass_the_confirm_bar():
    """promotion.early_confirm_power: an early confirmation needs a predictive probability >= the threshold that its
    3 fresh confirm seeds reach 1/alpha_confirm (live 2026-10-07: n1764 P(gain>0)=0.563 E[gain+]=0.0006 BF=1.28 went
    to confirm at ~7% power); one whose positive part is near delta_1 still goes."""
    cfg = _cfg(seeds=SeedPlan(promote=(13, 15, 16, 17), promote_stage=False, max_seeds_per_node=5),
               promotion=PromotionCfg(p_promote=0.6, confirm_max_seeds=3, alpha_confirm=0.2, early_confirm=True,
                                      early_confirm_power=0.2))
    own = [_run("n1", Stage.SCREEN, 11), _run("n1", Stage.SCREEN, 12)]
    post = lambda p, plus: dataclasses.replace(_post(), gain_p=p, gain_prior=0.3, gain_plus=plus)   # noqa: E731
    step = lambda p, c=cfg: S.step(   # noqa: E731
        _node(NodeStatus.SCREENING), own_runs=own, control_runs=_roots([11, 12]), post=p, cfg=c,
        sigma_panel=0.0034, infra_retry=2, raw=0.002)
    weak, real = post(0.563, 0.0006), post(0.55, 0.002)
    assert S.confirm_power(weak, 0.0034, 0.2, 3) < 0.1 < 0.2 < S.confirm_power(real, 0.0034, 0.2, 3)
    # a null node: power is the test's size under the frozen alternative, far below the threshold
    assert S.confirm_power(post(0.0, 0.0), 0.0034, 0.2, 3) < 0.05
    # monotone in the predicted gain and in the confirm seeds
    assert S.confirm_power(post(0.55, 0.001), 0.0034, 0.2, 3) < S.confirm_power(real, 0.0034, 0.2, 3)
    assert S.confirm_power(real, 0.0034, 0.2, 1) < S.confirm_power(real, 0.0034, 0.2, 3)
    assert step(weak).status is NodeStatus.SCREENED_OUT
    d = step(real)
    assert d.status is NodeStatus.CONFIRMING and "early confirmation" in d.reason and "power=0.30" in d.reason
    off = _cfg(seeds=cfg.seeds, promotion=dataclasses.replace(cfg.promotion, early_confirm_power=0.0))
    assert step(weak, off).status is NodeStatus.CONFIRMING            # 0: the BF rule alone (as before)


def test_the_daily_digest_is_computed_written_and_logged_once_a_day(tmp_path):
    from loop import digest as DG
    ctx = _world(tmp_path, review_every=1000)
    driver, store, _ = _make(ctx, Scripts())
    now = [1.0e9]
    driver.clock = lambda: now[0]
    acts = _run_ticks(driver, 4)
    assert any(a.startswith("digest written") for a in acts) and len(store.events("digest")) == 1
    text = (Path(ctx.state_dir) / "digest.md").read_text()
    for part in ("## Milestones", "## LLM calls by role", "- researcher: ", "## GPU-h by lane", "idle ~",
                 "## Operator proposals open: 0"):
        assert part in text, (part, text)
    assert DG.tick(ctx, store, now[0] + 3600) == []                       # once a day
    now[0] += DG.EVERY_S
    driver.tick()
    assert len(store.events("digest")) == 2 and (Path(ctx.state_dir) / "digests").is_dir()


def test_redundant_llm_work_is_skipped_and_restated_output_merged(tmp_path):
    """loop.redundancy: a review / consolidation / scout run on unchanged evidence is skipped; a proposal restating
    an earlier one is dropped (counted on it); a meta restating a live one is merged (its probes re-pointed); a
    playbook restating the current rules is no new version."""
    from loop import redundancy as RD
    from loop.agents.analyst import FP_TRIGGER
    from loop.contracts import AxisPrediction, HypothesisRecord, LeverTags, ReviewReport
    from loop.driver import K_REVIEW
    ctx = _world(tmp_path, review_every=1000)
    driver, store, _ = _make(ctx, Scripts())
    driver.tick()
    driver.drain()
    acts: list[str] = []
    driver._harvest(acts)                                    # the epoch's first-principles review lands
    assert store.get_state(RD.K_INPUTS)["review"]["key"] == RD.evidence_key(store, driver.snapshot)
    acts = []
    driver.start_review("stall", acts)                       # nothing closed since: no LLM call
    assert driver._review is None and any(a.startswith("review:stall skipped") for a in acts), acts
    assert store.get_state(K_REVIEW)["skipped"] and store.events("llm_job_skipped")
    driver.start_review("manual", [])                        # an operator's review always runs
    assert driver._review is not None
    driver.drain()
    driver._harvest([])
    gate = "Gate screen promotion on the OWN paired increment vs the measure parent, not on total U vs root"
    rv = ReviewReport(id="rv-x", task_id=store.task_id, trigger=FP_TRIGGER, summary="s", findings=(),
                      meta_hypothesis_ids=(), probe_briefs=(), explore_bias={}, n_evals_at=0,
                      proposals=(gate + " -- evidence A", "Down-weight the survival axis -- noisy"))
    driver.persist_review(rv, [])
    claim = "the model is over-regularised: weaker augmentation and lower weight decay help the panel"
    live = HypothesisRecord(id="rv-x-m1", task_id=store.task_id, kind=HypothesisKind.META, claim=claim,
                            mechanism_tag="m", family="cross", predictions={PANEL: AxisPrediction(1, 0.003)},
                            falsification="f", levers=LeverTags({"regularization": -1}), source="review",
                            review_id="rv-x")
    store.put_hypothesis(live)
    again = dataclasses.replace(live, id="rv-y-m1", review_id="rv-y")
    rv2 = dataclasses.replace(rv, id="rv-y", proposals=(
        "Change the screen GATE decision to the OWN paired increment vs the measure parent, not U vs root -- B",
        "Add a smoke gate before design batches -- d3 crashed"), meta_hypothesis_ids=("rv-y-m1",),
        probe_briefs=(("rv-y-m1", "halve weight decay"),))
    driver.persist_review(rv2, [again])
    got = next(r for r in store.reviews() if r.id == "rv-y")
    assert got.proposals == ("Add a smoke gate before design batches -- d3 crashed",)
    assert store.get_state(RD.K_REPEATS) == {"rv-x#0": 1} and store.events("proposal_deduped")
    assert store.get_hypothesis("rv-y-m1") is None and got.meta_hypothesis_ids == ("rv-x-m1",)
    assert got.probe_briefs == (("rv-x-m1", "halve weight decay"),) and store.events("meta_merged")
    rules = [{"component": "researcher", "rule": "Predict smaller effect sizes; default to near-zero gains"}]
    assert RD.same_rules(rules, [{"component": "researcher", "rule": "Predict smaller effect sizes and default "
                                                                     "to near-zero gains"}])
    assert not RD.same_rules(rules, [{"component": "researcher", "rule": "Prefer data curation over objectives"}])
