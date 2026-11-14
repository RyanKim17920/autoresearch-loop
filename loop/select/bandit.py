"""Discounted Gaussian Thompson bandit over the strategy arms (D4.2), reused over brief kinds
(meta-learning which ideation works: computed credit per GPU-h, never an LLM judgement)."""

from __future__ import annotations

import math
import random
import re
import time
from types import SimpleNamespace
from typing import Any, Sequence

from loop.contracts import (BANDIT_ARMS, PANEL, ArmPull, Node, NodeStatus, Stage, Strategy, WorldSnapshot,
                            design_point)
from loop.spec import BanditCfg


K_KIND_PULLS = "select.kind_pulls"        # [{arm: brief kind, reward, ts, node_id}]


class Bandit:
    """State is a pure function of ``store.pulls()``; rebuild it before every choice. Arms may be
    any hashables (strategies, or brief kinds via :func:`kind_pulls`)."""

    def __init__(self, cfg: BanditCfg, rng: random.Random,
                 arms: Sequence[Strategy] = BANDIT_ARMS) -> None:
        self.cfg, self.rng, self.arms = cfg, rng, tuple(arms)
        self.rebuild(())

    def rebuild(self, pulls: Sequence[ArmPull]) -> "Bandit":
        g = self.cfg.discount
        self.S = {a: 0.0 for a in self.arms}
        self.N = {a: 0.0 for a in self.arms}
        rewards = []
        for p in sorted((p for p in pulls if p.arm in self.S), key=lambda x: x.ts):
            for a in self.arms:                        # discount once per close event
                self.S[a], self.N[a] = self.S[a] * g, self.N[a] * g
            self.S[p.arm] += p.reward
            self.N[p.arm] += 1.0
            rewards.append(p.reward)
        mu = sum(rewards) / max(len(rewards), 1)
        self.reward_var = max(1e-6, sum((r - mu) ** 2 for r in rewards) / (len(rewards) - 1)) \
            if len(rewards) >= 2 else 1.0
        return self

    def draw(self, arm: Strategy) -> float:
        n = self.N[arm]
        return self.rng.gauss(self.S[arm] / (n + 1.0), math.sqrt(self.reward_var / (n + 1.0)))

    def choose(self, eligible: Sequence[Strategy]) -> Strategy:
        """Floor mixing: with prob min(1, sum of floors) an arm drawn in proportion to its floor
        (so each eligible arm gets at least its floor), else argmax draw. The explore arm may
        carry its own higher floor (``explore_floor``), and so may the moonshot arm (``moonshot_floor``)."""
        el = [a for a in self.arms if a in set(eligible)]
        if not el:
            raise ValueError("no eligible arm")
        if len(el) == 1:
            return el[0]
        own = {Strategy.EXPLORE: self.cfg.explore_floor, Strategy.MOONSHOT: self.cfg.moonshot_floor}
        floors = [f if isinstance(a, Strategy) and (f := own.get(a)) is not None else self.cfg.floor
                  for a in el]
        if self.rng.random() < min(1.0, sum(floors)):
            if len(set(floors)) == 1:
                return self.rng.choice(el)
            return self.rng.choices(el, weights=floors)[0]
        return max(el, key=self.draw)


def _half_log_ratio(before: float, after: float) -> float:
    if before <= 0.0 or after <= 0.0:
        return 0.0
    return max(0.0, 0.5 * math.log(before / after))


def information_gain(world: Any, before: WorldSnapshot, after: WorldSnapshot,
                     node: Node) -> float:
    """1/2 log(v_prior/v_post) for eps_i, plus the same for the panel coefficient of each
    REGISTERED non-zero lever of the node. Clamped at 0."""
    post = after.nodes.get(node.id)
    prior_var = world.predictive(before, family=node.family, tag=node.mechanism_tag, levers=node.levers,
                                 **({"node": node} if getattr(before, "wm", None) is not None else {})).var
    ig = _half_log_ratio(prior_var, post.increment.var) if post is not None else 0.0
    nonzero = {l for l, d in node.levers.directions.items() if d}
    bse = {e.lever: e.se for e in before.levers if e.target == PANEL}
    for e in after.levers:
        if e.target == PANEL and e.lever in nonzero and e.lever in bse:
            ig += _half_log_ratio(bse[e.lever] ** 2, e.se ** 2)
    return ig


def compute_reward(store: Any, world: Any, cfg: BanditCfg, before: WorldSnapshot,
                   after: WorldSnapshot, node: Node, *, sigma_p: float,
                   now: float | None = None) -> ArmPull:
    """Bandit credit for a node that just closed. The gain compares the node's
    confirm-only estimate with the incumbent's confirm-only estimate (root: 0); a
    selection-based utility is never used (winner's curse)."""
    gain = 0.0
    post = after.nodes.get(node.id)
    if node.status == NodeStatus.CONFIRMED and post is not None and post.confirm_estimate is not None:
        inc = before.nodes.get(before.incumbent_id)
        inc_u = inc.confirm_estimate.mean if inc is not None and inc.confirm_estimate is not None else 0.0
        gain = max(0.0, post.confirm_estimate.mean - inc_u) / max(sigma_p, 1e-12)
    gpu_h = sum(r.gpu_h for r in store.runs(node_id=node.id))
    if m := re.search(r"\btune_of=(\S+)", node.notes or ""):      # a tuned moonshot pays for its tuning pass
        gpu_h += tune_gpu_h(store, m.group(1))
    r = (gain + cfg.info_weight * information_gain(world, before, after, node)) / max(
        gpu_h, cfg.gpu_h_min)
    return ArmPull(task_id=node.task_id, arm=node.strategy, node_id=node.id, reward=r,
                   gpu_h=gpu_h, ts=time.time() if now is None else now)


def tune_gpu_h(store: Any, base_id: str) -> float:
    """GPU-h of a moonshot's tuning pass (loop.tune): every trial's runs (config points with the moonshot as
    parent; a backrun's graft onto the incumbent, ``backrun_of=``, and its points) plus the moonshot's own PARENT
    runs at those trials' seeds (their controls)."""
    pts = [n for n in store.nodes() if design_point(n)]
    gr = {n.id for n in pts if f" backrun_of={base_id} " in f" {n.notes or ''} "}
    trials = [n for n in pts if n.parent_id == base_id or n.id in gr or n.parent_id in gr
              or f" campaign_of={base_id} " in f" {n.notes or ''} "]       # a campaign's points (loop.campaign)
    runs = [r for n in trials for r in store.runs(node_id=n.id)]
    seeds = {r.seed for r in runs if r.stage is Stage.SCREEN}
    return sum(r.gpu_h for r in runs) + sum(r.gpu_h for r in store.runs(node_id=base_id, stage=Stage.PARENT)
                                            if r.seed in seeds)


def brief_of(node: Node) -> str:
    """The brief kind of the request that produced ``node`` (its intake notes); '' if unknown."""
    m = re.search(r"\bbrief=(\S+)", node.notes or "")
    return m.group(1) if m else ""


def kind_credit(store: Any, after: WorldSnapshot, node: Node, cfg: BanditCfg, sigma_p: float) -> float:
    """A closed node's credit to its brief kind: P(U>0) after screening (0 if it never ran, e.g.
    rejected) plus, if confirmed, its shrunk gain over the parent in sigmas; per GPU-h."""
    post = after.nodes.get(node.id)
    q = post.p_positive if post is not None and post.utility.n > 0 else 0.0
    if node.status is NodeStatus.CONFIRMED and post is not None:
        q += max(0.0, post.increment.mean) / max(sigma_p, 1e-12)
    return q / max(sum(r.gpu_h for r in store.runs(node_id=node.id)), cfg.gpu_h_min)


def kind_pulls(store: Any) -> list[SimpleNamespace]:
    return [SimpleNamespace(**p) for p in store.get_state(K_KIND_PULLS, [])]
