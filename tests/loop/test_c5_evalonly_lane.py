"""v3-c5 I5: the v3 eval-only readout lane (loop.evalonly.Lane), the widened eligibility (readout-only config keys /
constants, the training-path guard), the eval-only pair noise in the posterior, the per-axis harm guard, and the
per-dataset feature rows in the levers section."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from loop import evalonly as EO
from loop.contracts import Node as CNode, NodeStatus, Run as CRun, RunStatus, Stage
from loop.core import data as CD
from loop.core import model as M
from loop.spec import EvalOnlyCfg
from tests.loop.test_adapters import git, make_np, np_ctx_with_locks
from tests.loop.test_evalonly import FAMS, MODEL, TRAIN, repo  # noqa: F401  (the fixture)

# ------------------------------------------------------------------------------------------------ eligibility

FWD = MODEL.replace("        return x * 2\n", "        x = x * 2\n        if not self.training and self.rn_mu is not None:\n"
                    "            x = x - 1\n        return x\n")


def adapter_for(tmp_path, repo, **eo):   # noqa: F811
    ctx = np_ctx_with_locks(tmp_path / "state", ("probe.py",), families=FAMS,
                            evalonly=EvalOnlyCfg(enabled=True, **eo))
    return make_np(ctx, repo)[0]


def commit_on(repo, base, name, **files):   # noqa: F811
    git(repo, "checkout", "-q", "-b", name, base)
    for rel, text in files.items():
        (repo / rel.replace("__", "/")).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", name)
    sha = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "main")
    return sha


def cfg_with(repo, **model):   # noqa: F811
    cfg = yaml.safe_load((repo / "configs" / "main.yaml").read_text())
    cfg["model"].update(model)
    return yaml.safe_dump(cfg)


READS_K = TRAIN.replace("fit(lr)", "fit(lr) * cfg['model']['readout_k']")


@pytest.mark.parametrize("keys,files,ok", [
    # a dose key read only inside the post-training fit span: eligible with readout_keys only
    (True, {"configs__main.yaml": "K", "train.py": READS_K}, True),
    (False, {"configs__main.yaml": "K", "train.py": READS_K}, False),
    # a key read by no code is no readout knob
    (True, {"configs__main.yaml": "K"}, False),
    # a key read outside the spans (training code) is a training change
    (True, {"configs__main.yaml": "K", "train.py": READS_K.replace("lr = 0.1", "lr = 0.1 * cfg['model']['readout_k']")},
     False),
    # a literal constant outside the spans, read only in the readout span
    (True, {"model.py": MODEL.replace("    def forward(self, x):\n        return x * 2",
                                      "    SHRINK = 0.5\n\n    def forward(self, x):\n        return x * self.SHRINK")},
     True),
    # ... read by trained code too
    (True, {"model.py": MODEL.replace("    def forward(self, x):\n        return x * 2",
                                      "    SHRINK = 0.5\n\n    def forward(self, x):\n        return x * self.SHRINK")
            .replace("return 1", "return self.SHRINK")}, False),
])
def test_readout_only_keys_and_constants(tmp_path, repo, keys, files, ok):   # noqa: F811
    ad = adapter_for(tmp_path, repo, readout_keys=keys)
    base = git(repo, "rev-parse", "main")
    files = {k: (cfg_with(repo, readout_k=4) if v == "K" else v) for k, v in files.items()}
    child = commit_on(repo, base, "c", **files)
    got = ad.eval_only(base, child)
    assert got[0] is ok, got


def test_probe_keys_are_never_readout_keys(tmp_path, repo):   # noqa: F811
    ad = adapter_for(tmp_path, repo, readout_keys=True)
    base = git(repo, "rev-parse", "main")
    cfg = yaml.safe_load((repo / "configs" / "main.yaml").read_text())
    cfg["probe"] = {"k": 1}
    child = commit_on(repo, base, "p", **{"configs__main.yaml": yaml.safe_dump(cfg)})
    ok, why, _ = ad.eval_only(base, child)
    assert not ok and "probe key" in why


def test_training_path_guard_on_train_spans(tmp_path, repo):   # noqa: F811
    base = git(repo, "rev-parse", "main")
    parent = commit_on(repo, base, "fwd", **{"model.py": FWD})
    inside = commit_on(repo, parent, "in", **{"model.py": FWD.replace("x = x - 1", "x = x - 2")})
    outside = commit_on(repo, parent, "out", **{"model.py": FWD.replace("x = x * 2", "x = x * 3")})
    guarded = adapter_for(tmp_path / "g", repo, train_spans=("model.py::ViT.forward",))
    ok, why, _ = guarded.eval_only(parent, inside)     # v3-c5b M3: the eval-mode EMA teacher runs that branch
    assert not ok and "training-path" in why
    ok, why, _ = guarded.eval_only(parent, outside)
    assert not ok and "training-path" in why
    assert adapter_for(tmp_path / "u", repo).eval_only(parent, outside)[0] is True    # default: no guard (old lane)


def test_clear_eval_drops_a_plan(tmp_path, repo):   # noqa: F811
    ad = adapter_for(tmp_path, repo)
    ad.plan_eval("n1", Stage.SCREEN, 7, {"commit": "c"})
    run = SimpleNamespace(node_id="n1", stage=Stage.SCREEN, seed=7)
    assert ad.eval_plan(run) == {"commit": "c"}
    ad.clear_eval("n1", Stage.SCREEN, 7)
    assert ad.eval_plan(run) is None


def test_task_spec_parses_the_lane_keys():
    from loop.spec import load_context
    eo = load_context("tasks/nanopath_v2.yaml", state_dir="/tmp/_eo_c5", require_measured=False).cfg.evalonly
    assert eo.readout_keys and eo.train_spans == ("model.py::ViT.forward", "model.py::ViT._forward_tokens")
    assert eo.harm_tol == {"slide": 0.005, "rob": 0.003} and eo.harm_prob == 0.8 and eo.noise_min_df == 4


# ------------------------------------------------------------------------------------------------ the lane

AX = dict.fromkeys(CD.AXES, 0.5)


class Store:
    def __init__(self, nodes):
        self.state, self.nodes, self.runs, self.events = {}, dict(nodes), {}, []

    def get_state(self, k, default=None):
        return self.state.get(k, default)

    def set_state(self, k, v):
        self.state[k] = v

    def get_node(self, nid):
        return self.nodes.get(nid)

    def get_run(self, rid):
        return self.runs.get(rid)

    def log(self, event, **f):
        self.events.append((event, f))

    def update_node(self, nid, **f):
        self.nodes[nid] = dataclasses.replace(self.nodes[nid], **f)
        return self.nodes[nid]


class Adapter:
    def __init__(self, elig=(True, "", False)):
        self.plans, self.elig = {}, elig

    def eval_only(self, parent, commit):
        return self.elig

    def plan_eval(self, node, stage, seed, plan):
        self.plans[(node, getattr(stage, "value", stage), int(seed))] = dict(plan)

    def eval_plan(self, run):
        return self.plans.get((run.node_id, getattr(run.stage, "value", run.stage), int(run.seed)))

    def clear_eval(self, node, stage, seed):
        self.plans.pop((node, getattr(stage, "value", stage), int(seed)), None)


def cnode(nid, parent=None, status=NodeStatus.SCREENING):
    return CNode(id=nid, task_id="t", commit=f"c-{nid}", parent_id=parent, strategy="explore", operator="improve",
                 family="eval_readout", mechanism_tag="m", diff_hash="h", status=status)


def lnode(nid, parent=None):
    return CD.Node(nid, parent, None, "explore", "eval_readout", "explore", "root", 0.0, "", "", "B")


def run(nid, stage, seed, status="done", out="", ts=0.0):
    return CD.Run(f"{nid}:{stage}:{seed}", nid, seed, stage, status, 1.0, ts, ts, out, 0, "B")


def obs(nid, stage, seed, panel, **axes):
    return CD.Obs(f"{nid}:{stage}:{seed}", nid, seed, stage, CD.SELECT, {**AX, **axes}, panel, 1.0, 0.0, 0, None, "B")


def ledger(runs, observations):
    nodes = {"root": lnode("root"), "p": lnode("p", "root"), "c": lnode("c", "p")}
    return CD.Ledger(nodes, tuple(runs), tuple(observations), frozenset(), ((0.0, "root"),),
                     bases=((0.0, "root", "B"),))


def lane(tmp_path, adapter=None, now=1000.0, **eo):
    cfg = EvalOnlyCfg(enabled=True, **eo)
    ctx = SimpleNamespace(cfg=SimpleNamespace(evalonly=cfg), spec=SimpleNamespace(caps=SimpleNamespace(infra_retry=2)),
                          noise=SimpleNamespace(replicate_sd=0.0005))
    store = Store({n: cnode(n, p) for n, p in (("root", None), ("p", "root"), ("c", "p"))})
    clock = [now]
    return EO.Lane(ctx, store, adapter or Adapter(), lambda: clock[0], exists=lambda p: "ckpt" in str(p)), store, clock


def spec(node, seed, stage=Stage.SCREEN):
    from loop.tick import RunSpec
    return RunSpec(node, stage, seed)


def test_route_null_control_first_then_parent_checkpoints(tmp_path):
    ln, store, clock = lane(tmp_path)
    data = ledger([run("p", "screen", 5, out="/ckpt/p5")], [obs("p", "screen", 5, 0.6)])
    acts = []
    out = ln.route(data, [spec("c", 5)], acts)
    # the candidate waits; the null control (the parent's own code on its own checkpoint) launches first
    assert [(s.node, s.stage, s.seed) for s in out] == [("p", Stage.CALIBRATE, 5)]
    null = ln.adapter.plans[("p", "calibrate", 5)]
    assert null["commit"] == "c-p" and null["resume"] == "/ckpt/p5/latest.pt" and null["source_run"] == "p:screen:5"
    # the null run reproduced the source: the candidate's run resumes the parent's checkpoint at the same seed
    store.runs["p:calibrate:5"] = CRun(id="p:calibrate:5", task_id="t", node_id="p", seed=5, stage=Stage.CALIBRATE,
                                       status=RunStatus.DONE)
    data = ledger([run("p", "screen", 5, out="/ckpt/p5"), run("p", "calibrate", 5)],
                  [obs("p", "screen", 5, 0.6), obs("p", "calibrate", 5, 0.6)])
    out = ln.route(data, [spec("c", 5)], acts)
    assert [(s.node, s.seed) for s in out] == [("c", 5)]
    plan = ln.adapter.plans[("c", "screen", 5)]
    assert plan == {"commit": "c-c", "seed": 5, "shim": False, "resume": "/ckpt/p5/latest.pt",
                    "source_run": "p:screen:5", "model": "p"}
    assert store.state[EO.K_V3]["null"]["B"]["status"] == "passed"


def test_failed_null_control_means_normal_training(tmp_path):
    ln, store, _ = lane(tmp_path)
    store.state[EO.K_V3] = {"null": {"B": {"status": "failed"}}}
    ln.adapter.plans[("c", "screen", 5)] = {"commit": "stale"}           # an earlier tick's plan, never launched
    data = ledger([run("p", "screen", 5, out="/ckpt/p5")], [obs("p", "screen", 5, 0.6)])
    out = ln.route(data, [spec("c", 5)], [])
    assert [(s.node, s.seed) for s in out] == [("c", 5)] and ("c", "screen", 5) not in ln.adapter.plans


def test_source_in_flight_waits_then_trains(tmp_path):
    ln, store, clock = lane(tmp_path, source_wait_s=100.0)
    store.state[EO.K_V3] = {"null": {"B": {"status": "passed"}}}
    data = ledger([run("p", "screen", 5, status="running", ts=990.0)], [])
    assert ln.route(data, [spec("c", 5)], []) == []                   # the parent's checkpoint is being trained
    clock[0] = 2000.0
    out = ln.route(data, [spec("c", 5)], [])
    assert [(s.node, s.seed) for s in out] == [("c", 5)] and not ln.adapter.plans    # waited too long: trains


def test_no_checkpoint_or_ineligible_trains_normally(tmp_path):
    ln, store, _ = lane(tmp_path)
    store.state[EO.K_V3] = {"null": {"B": {"status": "passed"}}}
    data = ledger([run("p", "screen", 5, out="/pruned/p5")], [obs("p", "screen", 5, 0.6)])
    assert [s.seed for s in ln.route(data, [spec("c", 5)], [])] == [5] and not ln.adapter.plans
    # (v3-c6a: the null control has its own start trigger -- off here, this test is about the candidate's routing)
    ln2, store2, _ = lane(tmp_path, adapter=Adapter((False, "outside", False)), null_at_start=False)
    data = ledger([run("p", "screen", 5, out="/ckpt/p5")], [obs("p", "screen", 5, 0.6)])
    assert [s.seed for s in ln2.route(data, [spec("c", 5)], [])] == [5] and not ln2.adapter.plans
    assert store2.state[EO.K_V3]["elig"]["c"][0] is False


def test_disabled_lane_is_a_no_op(tmp_path):
    ln, _, _ = lane(tmp_path)
    ln.ctx.cfg.evalonly = EvalOnlyCfg(enabled=False)
    specs = [spec("c", 5)]
    assert ln.route(ledger([], []), specs, []) == specs and ln.prepare(ledger([], []), []) == (frozenset(), {})


def eo_data(ln, slide_d, n=3, panel_d=0.001):
    """c ran eval-only on p's checkpoints at seeds 1..n."""
    rs, os_ = [], []
    for s in range(1, n + 1):
        rs += [run("p", "screen", s, out=f"/ckpt/p{s}"), run("c", "screen", s)]
        os_ += [obs("p", "screen", s, 0.6 + 0.003 * s, slide=0.7 + 0.01 * s),
                obs("c", "screen", s, 0.6 + 0.003 * s + panel_d, slide=0.7 + 0.01 * s + slide_d + 0.0002 * s)]
        ln.adapter.plans[("c", "screen", s)] = {"commit": "c-c"}
    return ledger(rs, os_)


def test_harm_guard_holds_a_slide_loser(tmp_path):
    ln, store, _ = lane(tmp_path, harm_tol={"slide": 0.005}, harm_prob=0.8)
    store.state[EO.K_V3] = {"elig": {"c": [True, "", False]}, "elig_v": EO.ELIG_V}
    acts = []
    pure, harm = ln.prepare(eo_data(ln, slide_d=-0.02), acts)
    assert pure == {"c"} and "slide" in harm["c"]
    assert store.nodes["c"].status is NodeStatus.VETOED and any(e == "evalonly_harm" for e, _ in store.events)
    ln2, store2, _ = lane(tmp_path, harm_tol={"final_probe_slide_mean_auc": 0.005})     # long axis names map too
    store2.state[EO.K_V3] = {"elig": {"c": [True, "", False]}, "elig_v": EO.ELIG_V}
    assert ln2.prepare(eo_data(ln2, slide_d=-0.001), [])[1] == {}                      # within tolerance
    ln3, store3, _ = lane(tmp_path, harm_tol={"slide": 0.005})
    store3.state[EO.K_V3] = {"elig": {"c": [True, "", False]}, "elig_v": EO.ELIG_V}
    assert ln3.prepare(eo_data(ln3, slide_d=-0.02, n=2), [])[1] == {}                  # < screen_ckpts pairs


def test_pure_needs_every_run_eval_only(tmp_path):
    ln, store, _ = lane(tmp_path)
    store.state[EO.K_V3] = {"elig": {"c": [True, "", False]}, "elig_v": EO.ELIG_V}
    data = eo_data(ln, 0.0)
    assert ln.pure(data) == {"c"}
    del ln.adapter.plans[("c", "screen", 2)]                          # one run trained
    assert ln.pure(data) == frozenset()


# ------------------------------------------------------------------------------------------------ noise


def test_posterior_noise_is_the_pooled_pair_spread_floored():
    nz = M.Noise({"panel": 1e-5}, {"panel": 0.4})
    ln, _, _ = lane(Path("."))
    data = eo_data(ln, 0.0, n=5)
    by = data.by_node()
    got = EO.posterior_noise(nz, by, {"c": "p"}, ["c"], None, 0.0005, 4)
    # c - p is 0.001 at every seed: zero spread -> floored at 0.0005 -> rho = 1 - 0.0005^2 / (2e-5) = .9875
    assert got.rho["panel"] == pytest.approx(1 - 0.0005 ** 2 / 2e-5)
    assert EO.posterior_noise(nz, by, {"c": "p"}, ["c"], None, 0.0005, 5) is None      # 4 dfs < 5
    assert EO.posterior_noise(M.Noise({"panel": 1e-5}, {"panel": 0.995}), by, {"c": "p"}, ["c"], None, 0.0005,
                              4).rho["panel"] == 0.995                                   # never below the class rho


def test_eval_only_pairs_tighten_the_posterior_increment():
    """In acquire.Posterior an eval-only node's CUPED increment uses its own pair noise: same delta, smaller V."""
    from loop.core.acquire import Posterior
    import random
    rnd = random.Random(3)
    nodes = {"root": lnode("root")}
    rs, os_ = [], []
    seed_fx = {s: rnd.gauss(0, 0.004) for s in range(1, 9)}
    for nid in ("root", "a", "b", "d", "e", "c"):
        if nid != "root":
            nodes[nid] = lnode(nid, "root")
        mu = {"c": 0.002}.get(nid, rnd.gauss(0, 0.002))
        for s in range(1, 9):
            y = 0.6 + mu + seed_fx[s] + (0.0 if nid == "c" else rnd.gauss(0, 0.002))
            if nid == "c":
                y = [o for o in os_ if o.node == "root" and o.seed == s][0].panel + 0.002 + rnd.gauss(0, 0.0003)
            rs.append(run(nid, "screen", s))
            os_.append(CD.Obs(f"{nid}:screen:{s}", nid, s, "screen", CD.SELECT, {a: y for a in CD.AXES}, y, 1.0, 0.0,
                              0, None, "B"))
    led = CD.Ledger(nodes, tuple(rs), tuple(os_), frozenset(), ((0.0, "root"),), bases=((0.0, "root", "B"),))
    plain = Posterior(led, [])
    eo = Posterior(led, [], evalonly=frozenset({"c"}), eo_floor=0.0005, eo_min_df=4)
    p, e = plain.est["c"][M.COMPOSITE], eo.est["c"][M.COMPOSITE]
    assert e.var < p.var / 4 and abs(e.delta - 0.002) < abs(p.delta - 0.002) + 0.001
    assert eo.est["a"][M.COMPOSITE].var == pytest.approx(plain.est["a"][M.COMPOSITE].var)   # others untouched


def test_module_core_passes_the_eval_only_set(monkeypatch):
    from loop.tick import ModuleCore
    core = ModuleCore()
    seen = {}

    class P:
        def __init__(self, led, blocks, *, near=frozenset(), evalonly=frozenset(), eo_floor=0.0, eo_min_df=4):
            seen.update(evalonly=evalonly, eo_floor=eo_floor, eo_min_df=eo_min_df)
    monkeypatch.setattr(core.A, "Posterior", P)
    core.evalonly, core.eo_floor, core.eo_min_df = frozenset({"c"}), 0.0005, 6
    core.posterior(SimpleNamespace(obs=(1,)), [])
    assert seen == {"evalonly": frozenset({"c"}), "eo_floor": 0.0005, "eo_min_df": 6}


# ------------------------------------------------------------------------------------------------ levers


def test_feature_rows_from_the_cached_diagnostic(monkeypatch):
    from loop.agents import diagnose as DX
    from loop.core import levers as LV
    from loop.core.ideation import ROW_ID

    def g(m):
        return {"n": 32, "mean": m, "p10": m / 2, "p50": m, "p90": m * 1.5}
    rep = {"wilds": {"tiles": 32, "norm_probe_features_head": g(4.0), "norm_cls": g(40.0), "cos_head_vs_cls": g(0.9),
                     "gates": [g(0.06)]},
           "bach": {"tiles": 32, "norm_probe_features_head": g(20.0), "norm_cls": g(40.0), "cos_head_vs_cls": g(0.95),
                    "gates": [g(0.30)]},
           "crc": {"tiles": 32, "norm_probe_features_head": g(24.0), "norm_cls": g(40.0), "cos_head_vs_cls": g(0.97),
                   "gates": [g(0.35)]}}
    seen = {}
    monkeypatch.setattr(DX, "cached_features", lambda ctx, store, inc=None: seen.setdefault("inc", inc) and rep)
    rows = LV.features(SimpleNamespace(incumbent="n9"), None, None)
    assert seen["inc"] == "n9" and [r["dataset"] for r in rows] == ["wilds", "bach", "crc"]
    assert rows[0]["gate_vs_median"] == pytest.approx(0.2) and rows[0]["ratio"] == pytest.approx(0.1)
    text = LV.features_text(rows, 16)
    assert "[G-feat-wilds] wilds (32 tiles): gate0 mean 0.060" in text and "0.20x the median dataset's 0.300" in text
    assert ROW_ID.findall(text) == ["G-feat-wilds", "G-feat-bach", "G-feat-crc"]
    monkeypatch.setattr(DX, "cached_features", lambda *a, **k: None)
    assert LV.features(SimpleNamespace(incumbent="n9"), None, None) == []
