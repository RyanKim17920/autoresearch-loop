"""loop.infra -- a smoke that died of the ENVIRONMENT (live 2026-10-21: the torch hub cache dir was deleted, four ideas
were REJECTED ``smoke_failed``, a program session produced nothing) is no verdict on the idea."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import loop.agents.researcher as RES
import loop.infra as INF
from loop import runner as R
from loop.agents.llm import FakeLLMClient
from loop.agents.researcher import LLMResearcher
from loop.contracts import ArmPull, CandidateRequest, Node, NodeStatus, Operator, ResearchResult, SmokeResult, Strategy
from loop.tick import Tick
from tests.loop.test_adapters import SlurmRunner, base_repo, make_np, np_ctx_with_locks, research_wt  # noqa: F401
from tests.loop.test_agents import FINISH, _ctx, _request, _snapshot, env  # noqa: F401  (fixtures)

HUB_TB = ("Traceback (most recent call last):\n  File \"train.py\", line 9, in <module>\n"
          "    torch.hub.load_state_dict_from_url(url)\n"
          "FileNotFoundError: [Errno 2] No such file or directory: '/home/user/.cache/torch/hub'\n")
CODE_TB = "Traceback (most recent call last):\n  File \"model.py\", line 77\nTypeError: bad shape 384 vs 768 at 0x7f3a\n"


# ------------------------------------------------------------------------------------------------ classification
@pytest.mark.parametrize("log,kind", [
    (HUB_TB, "cache path"),
    ("PermissionError: [Errno 13] Permission denied: '/home/u/.cache/huggingface/hub'", "cache path"),
    ("OSError: [Errno 28] No space left on device", "disk full"),
    ("RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory", "corrupt weights"),
    ("File \"torch/hub.py\", line 1, in load_state_dict_from_url\nurllib.error.URLError: <urlopen error "
     "[Errno -3] Temporary failure in name resolution>", "weights download"),
    ("ValueError: TORCH_HOME does not exist", "cache env"),
])
def test_environment_signatures_are_infra(log, kind):
    assert INF.classify(log).startswith(kind)


@pytest.mark.parametrize("log", [
    CODE_TB, "FileNotFoundError: [Errno 2] No such file or directory: '/data/user/dataset/tiles.parquet'",
    "urllib.error.URLError: <urlopen error timed out>",       # a network error outside any weights download
    "RuntimeError: CUDA out of memory", ""])
def test_an_ideas_own_failure_is_not_infra(log):
    assert INF.classify(log) == ""


def test_signature_collapses_numbers_ids_and_worktrees():
    a = "TypeError: bad shape 384 vs 768 in /w/loop-req-v3-00297/model.py at 0x7f3a"
    b = "TypeError: bad shape 192 vs 96 in /w/loop-req-v3-00301/model.py at 0x9bb1"
    assert INF.signature(a) == INF.signature(b) != ""
    assert INF.signature("no exception here") == ""


# ------------------------------------------------------------------------------------------------ the gate
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_gate_holds_then_lets_one_canary_through_and_clears(tmp_path):
    clk = Clock()
    g = INF.SmokeGate(tmp_path / "g.json", clk)
    assert g.blocked() == "" and not g.cleared()
    assert g.trip("cache path: x", "sig", "wt1") is True
    assert "held" in g.blocked() and g.wait_s() == INF.BACKOFF_S
    clk.t += INF.BACKOFF_S + 1
    assert g.blocked() == ""                                  # the canary
    assert "canary" in g.blocked()                            # everyone else still waits for it
    assert g.trip("cache path: x", "sig", "wt1") is False     # same signature: no second ALERT within the hour
    assert g.wait_s() == 2 * INF.BACKOFF_S                    # the backoff doubles
    clk.t += 2 * INF.BACKOFF_S + 1
    assert g.blocked() == "" and g.cleared() and g.blocked() == ""
    assert not g.status()["tripped"]


def test_gate_canary_that_timed_out_is_released_and_a_stale_one_replaced(tmp_path):
    clk = Clock()
    g = INF.SmokeGate(tmp_path / "g.json", clk)
    g.trip("r", "s")
    clk.t += INF.BACKOFF_S + 1
    assert g.blocked() == "" and g.blocked() != ""
    g.release_canary()
    assert g.blocked() == ""                                  # the next smoke is the canary
    clk.t += INF.CANARY_STALE_S + 1
    assert g.blocked() == ""                                  # its thread never reported: replaced


def test_three_sessions_with_one_tail_inside_an_hour_are_correlated(tmp_path):
    clk = Clock()
    g = INF.SmokeGate(tmp_path / "g.json", clk)
    sig = INF.signature(CODE_TB)
    assert g.correlated("loop-req-1", sig) == 1 and g.correlated("loop-req-1", sig) == 1   # one session, many smokes
    assert g.correlated("loop-req-2", sig) == 2
    clk.t += 3601
    assert g.correlated("loop-req-3", sig) == 1               # the first two aged out
    assert g.correlated("loop-req-4", sig) == 2 and g.correlated("loop-req-5", sig) == 3
    assert g.correlated("loop-req-6", "") == 0


# ------------------------------------------------------------------------------------------------ preflight
MODEL = ('VIT_VARIANTS = {"dinov2_vits14_reg": (384, 12, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/'
         'dinov2_vits14_reg4_pretrain.pth"),}\nVIT_VARIANTS["robust_norm_dinov2_vits14_reg"] = VIT_VARIANTS["dinov2_vits14_reg"]\n')


def test_weights_file_follows_a_variant_alias():
    assert INF.weights_files(MODEL, "robust_norm_dinov2_vits14_reg") == [
        ("https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
         "dinov2_vits14_reg4_pretrain.pth")]
    assert INF.weights_files(MODEL, "nope") == [] and INF.weights_files("", "x") == []


def test_preflight_names_a_dangling_hub_link(tmp_path):
    (tmp_path / "home").mkdir()
    os.symlink(tmp_path / "gone" / "torch", tmp_path / "home" / "torch")          # ~/.cache/torch -> deleted target
    env_ = {"TORCH_HOME": str(tmp_path / "home" / "torch")}
    msg = INF.preflight(MODEL, "robust_norm_dinov2_vits14_reg", env=env_, reachable=lambda u: True)
    assert "dangling symlink" in msg and str(tmp_path / "home" / "torch") in msg and str(tmp_path / "gone") in msg


def test_preflight_checks_the_weights_file_or_the_download(tmp_path):
    env_ = {"TORCH_HOME": str(tmp_path / "torch")}
    ck = tmp_path / "torch" / "hub" / "checkpoints"
    ck.mkdir(parents=True)
    down, up = (lambda u: False), (lambda u: True)
    msg = INF.preflight(MODEL, "robust_norm_dinov2_vits14_reg", env=env_, reachable=down)
    assert "dinov2_vits14_reg4_pretrain.pth" in msg and str(ck) in msg and "unreachable" in msg
    assert INF.preflight(MODEL, "robust_norm_dinov2_vits14_reg", env=env_, reachable=up) == ""     # downloadable
    (ck / "dinov2_vits14_reg4_pretrain.pth").write_bytes(b"w")
    assert INF.preflight(MODEL, "robust_norm_dinov2_vits14_reg", env=env_, reachable=down) == ""   # cached
    assert INF.preflight("", "", env={"TORCH_HOME": str(tmp_path / "fresh")}, reachable=down) == ""  # hub not made yet


# ------------------------------------------------------------------------------------------------ the adapter's smoke
@pytest.fixture
def smoker(tmp_path, base_repo):  # noqa: F811
    clk = Clock()
    runner = SlurmRunner({"sbatch": (0, "Submitted batch job 71\n", ""), "sacct": (0, "FAILED|1:0\n", "")})
    adapter, _ = make_np(np_ctx_with_locks(tmp_path / "sp"), base_repo, runner=runner, clock=clk)
    adapter.preflight = None
    return adapter, runner, base_repo, clk


def _log(wt, text):
    (Path(wt.path) / "slurm").mkdir(exist_ok=True)
    (Path(wt.path) / "slurm" / "71.out").write_text(text)


def _submits(runner):
    return [c for c in runner.calls if c["argv"][0] == "sbatch" or "launch" in c["argv"][0]]


def test_a_hub_cache_smoke_failure_is_infra_trips_the_gate_and_alerts_once(smoker, capsys):
    adapter, runner, base, clk = smoker
    wt = research_wt(adapter, base, "req-1")
    _log(wt, HUB_TB)
    res = adapter.smoke(wt)
    assert res.infra and res.unavailable and not res.ok and "cache path" in res.reason
    err = capsys.readouterr().err
    assert err.count("ALERT: smoke INFRA failure") == 1 and ".cache/torch/hub" in err
    n = len(_submits(runner))
    held = adapter.smoke(research_wt(adapter, base, "req-2"))             # held: nothing is submitted
    assert held.infra and "held" in held.reason and len(_submits(runner)) == n
    assert "ALERT" not in capsys.readouterr().err
    assert adapter.smoke_hold_wait() > 0
    clk.t += INF.BACKOFF_S + 1                                            # the canary goes through ...
    wt2 = research_wt(adapter, base, "req-3")
    _log(wt2, HUB_TB)
    assert adapter.smoke(wt2).infra and "ALERT" not in capsys.readouterr().err   # ... fails again: doubled, no 2nd ALERT
    clk.t += 2 * INF.BACKOFF_S + 1
    runner.replies["sacct"] = (0, "COMPLETED|0:0\n", "")
    wt3 = research_wt(adapter, base, "req-4")
    _log(wt3, "step 1 loss 6.9\ntraining complete\n")
    assert adapter.smoke(wt3).ok and not adapter.smoke_gate.status()["tripped"]   # the canary cleared the gate


def test_a_code_failure_is_a_verdict_and_clears_a_tripped_gate(smoker):
    adapter, runner, base, clk = smoker
    adapter.smoke_gate.trip("r", "s")
    clk.t += INF.BACKOFF_S + 1
    wt = research_wt(adapter, base, "req-1")
    _log(wt, CODE_TB)
    res = adapter.smoke(wt)
    assert not res.ok and not res.infra and not res.unavailable and "rc=1" in res.reason
    assert not adapter.smoke_gate.status()["tripped"]                    # it got past the environment


def test_three_different_sessions_with_one_traceback_tail_are_infra(smoker):
    adapter, runner, base, clk = smoker
    out = []
    for i in (1, 2, 3):
        wt = research_wt(adapter, base, f"req-{i}")
        _log(wt, CODE_TB.replace("384", str(100 + i)))
        out.append(adapter.smoke(wt))
    assert [r.infra for r in out] == [False, False, True] and "correlated" in out[2].reason
    assert adapter.smoke_gate.status()["tripped"]


def test_an_oom_is_never_correlated_into_infra(smoker):
    adapter, runner, base, clk = smoker
    for i in (1, 2, 3, 4):
        wt = research_wt(adapter, base, f"req-{i}")
        _log(wt, "Traceback\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 GiB\n")
        assert not adapter.smoke(wt).infra


def test_a_failed_preflight_alerts_with_the_path_and_submits_nothing(smoker, capsys):
    adapter, runner, base, clk = smoker
    adapter.preflight = lambda mpy, typ: "torch hub cache path /h/.cache/torch is under a dangling symlink (target missing)"
    res = adapter.smoke(research_wt(adapter, base, "req-1"))
    assert res.infra and "/h/.cache/torch" in res.reason and not _submits(runner)
    assert "ALERT" in capsys.readouterr().err and adapter.smoke_gate.status()["tripped"]


# ------------------------------------------------------------------------------------------------ the researcher / re-issue
def test_a_session_waits_out_an_infra_hold_and_keeps_its_work(env, monkeypatch):  # noqa: F811
    adapter, store = env
    monkeypatch.setattr(RES.time, "sleep", lambda s: None)
    script = lambda **_: {"tool_calls": [("edit_file", {"path": "fake_model.py", "old": "LR = 1e-3", "new": "LR = 2e-3"})],
                          "finish": FINISH}
    infra = SmokeResult(False, "smoke_unavailable: infra (cache path: x)", "", unavailable=True, infra=True)
    results = [infra, infra, SmokeResult(True)]
    adapter.smoke = lambda wt: (setattr(adapter, "smokes", adapter.smokes + 1), results.pop(0))[1]
    r = LLMResearcher(_ctx(), FakeLLMClient({"researcher": script}), adapter).research(_request(), store, _snapshot())
    assert r.status == "ok" and adapter.smokes == 3
    results[:] = [infra] * 12
    adapter.smokes = 0
    r = LLMResearcher(_ctx(), FakeLLMClient({"researcher": script}), adapter).research(_request(id="r2"), store, _snapshot())
    assert r.status == "backend_unavailable" and INF.is_infra_unavailable(r) and r.candidate.diff_text
    assert adapter.smokes <= 6                                           # bounded, not forever


def _research(ctx, store):
    return R.Research(ctx, store, None, None, None)


def _req(rid="req-v3-1"):
    return CandidateRequest(id=rid, task_id="t", strategy=Strategy.EXPLORE, operator=Operator.DRAFT, island="main",
                            parent_ids=("root",), brief="orig brief", brief_kind="bold")


def _res(rid="req-v3-1"):
    h = SimpleNamespace(family="optimization", mechanism_tag="m", claim="the claim")
    return ResearchResult("backend_unavailable", SimpleNamespace(hypothesis=h, diff_text="DIFF-BODY"),
                          "smoke_unavailable: smoke_unavailable: infra (cache path: x)")


def test_an_infra_session_is_reissued_with_its_diff_up_to_the_cap(ctx, store):
    rs = _research(ctx, store)
    rs.smoke_hold = lambda: 100.0
    line = rs.reissue_infra(_req(), _res())
    assert "re-issued (1/" in line and len(rs.requeue) == 1
    q = rs.requeue[0]
    assert INF.REISSUE_MARK in q.brief and "DIFF-BODY" in q.brief and q.brief.startswith("orig brief")
    assert any(e["status"] == "smoke_infra" for e in store.events("research_failed"))
    assert rs.start_requeued() == [] and len(rs.requeue) == 1            # held while the gate holds (no LLM session yet)
    for _ in range(INF.MAX_INFRA_REISSUES - 1):
        rs.reissue_infra(_req(), _res())
    assert "dropped" in rs.reissue_infra(_req(), _res())


def test_tick_intake_reissues_an_infra_session_instead_of_producing_nothing(ctx, store):
    rs = _research(ctx, store)
    fake = SimpleNamespace(research=rs, intake=lambda *a: pytest.fail("an infra session makes no node"))
    out = Tick._intake(fake, _req(), _res())
    assert "re-issued" in out and "produced nothing" not in out and len(rs.requeue) == 1
    other = SimpleNamespace(research=rs, intake=lambda r, s: "nothing")
    assert Tick._intake(other, _req(), ResearchResult("backend_unavailable", None, "backend_unavailable: URLError")) == "nothing"   # a smoke queue timeout is reissued (test_smoke_capacity)


def test_the_operators_drop_file_reissues_outage_rejected_nodes_from_their_stored_diffs(ctx, store, tmp_path):
    from loop.contracts import HandoffNote, HypothesisKind, HypothesisRecord
    st = Path(ctx.state_dir)
    st.mkdir(parents=True, exist_ok=True)
    h = HypothesisRecord(id="h-req-v3-7", task_id=ctx.task_id, kind=HypothesisKind.TRIAL, claim="sigmoid gate",
                         mechanism_tag="gate", family="architecture", predictions={}, falsification="f", created_ts=1.0)
    store.put_hypothesis(h)
    (st / "n9.diff").write_text("THE-DIFF")
    base = dict(task_id=ctx.task_id, parent_id="root", strategy=Strategy.EXPLORE, operator=Operator.DRAFT,
                family="architecture", mechanism_tag="gate", diff_hash="d", island="main", created_ts=1.0, updated_ts=1.0)
    store.ensure_root("base")
    store.put_node(Node(id="n9", commit="", status=NodeStatus.REJECTED, reject_reason="smoke_failed",
                        diff_path=str(st / "n9.diff"), hypothesis_id=h.id,
                        notes="request=req-v3-7 brief=bold lane=explore persona=pathology_domain lens_cat=architecture", **base))
    store.put_node(Node(id="n10", commit="", status=NodeStatus.REJECTED, reject_reason="intake_dup", notes="request=req-v3-8", **base))
    store.record_pull(ArmPull(task_id=ctx.task_id, arm=Strategy.EXPLORE, node_id="n9", reward=0.0, gpu_h=0.0, ts=1.0))
    rs = _research(ctx, store)
    assert INF.reissue_from_file(ctx, store, rs, 5.0) == []              # no file: nothing
    (st / INF.REISSUE_FILE).write_text('{"nodes": ["n9", "n10", "n404"]}')
    out = INF.reissue_from_file(ctx, store, rs, 5.0)
    assert store.get_node("n9").reject_reason == INF.REISSUED_REASON and store.get_node("n10").reject_reason == "intake_dup"
    assert [p.node_id for p in store.pulls()] == []                       # its bandit credit is dropped
    (q,) = rs.requeue
    assert q.id == "req-v3-7" and q.persona == "pathology_domain" and "THE-DIFF" in q.brief and "sigmoid gate" in q.brief
    assert sum("skipped" in o for o in out) == 2 and not (st / INF.REISSUE_FILE).exists()
    assert INF.reissue_from_file(ctx, store, rs, 6.0) == []               # once only
    assert dataclasses.is_dataclass(q)


# ------------------------------------------------------------------------------------------------ review 10-21 fixes
def test_a_canary_in_flight_reads_as_a_poll_interval_not_zero(tmp_path):
    clk = Clock()
    g = INF.SmokeGate(tmp_path / "g.json", clk)
    g.trip("r", "s")
    clk.t += INF.BACKOFF_S + 1
    assert g.wait_s() == 0.0                                  # nobody is the canary yet: the next smoke goes
    assert g.blocked() == "" and g.wait_s() == INF.CANARY_POLL_S
    assert g.blocked() != ""                                  # a waiter behind it is held AND told to poll, not to retry at once
    clk.t += INF.CANARY_STALE_S + 1
    assert g.wait_s() == 0.0


def test_waiters_behind_a_canary_do_not_burn_their_waits(env, monkeypatch):  # noqa: F811
    adapter, store = env
    naps = []
    monkeypatch.setattr(RES.time, "sleep", lambda s: naps.append(s))
    script = lambda **_: {"tool_calls": [("edit_file", {"path": "fake_model.py", "old": "LR = 1e-3", "new": "LR = 2e-3"})],
                          "finish": FINISH}
    infra = SmokeResult(False, "smoke_unavailable: infra hold: a canary smoke is in flight", "", unavailable=True, infra=True)
    results = [infra] * 6 + [SmokeResult(True)]
    adapter.smoke_hold_wait = lambda: INF.CANARY_POLL_S
    adapter.smoke = lambda wt: (setattr(adapter, "smokes", adapter.smokes + 1), results.pop(0))[1]
    r = LLMResearcher(_ctx(), FakeLLMClient({"researcher": script}), adapter).research(_request(), store, _snapshot())
    assert r.status == "ok" and len(naps) == 6 > RES.INFRA_SMOKE_WAITS      # six polls, more than the 3 tries


def test_the_same_session_failing_with_a_shared_tail_again_is_its_own_bug(smoker):
    adapter, runner, base, clk = smoker
    out = []
    for i in (1, 2, 3):
        wt = research_wt(adapter, base, f"req-{i}")
        _log(wt, CODE_TB.replace("384", str(100 + i)))
        out.append(adapter.smoke(wt))
    assert out[2].infra
    clk.t += INF.BACKOFF_S + 1                                            # the hold ends; its canary is req-3 again
    again = adapter.smoke(research_wt(adapter, base, "req-3"))
    assert not again.infra and "rc=1" in again.reason                     # a verdict, not an endless re-issue
    assert not adapter.smoke_gate.status()["tripped"]


def test_the_operators_reissue_frees_the_old_nodes_diff_hash_so_the_reapplied_diff_is_no_exact_dup(ctx, store):
    from loop.contracts import HypothesisKind, HypothesisRecord
    st = Path(ctx.state_dir)
    st.mkdir(parents=True, exist_ok=True)
    h = HypothesisRecord(id="h-req-v3-7", task_id=ctx.task_id, kind=HypothesisKind.TRIAL, claim="c", mechanism_tag="m",
                         family="architecture", predictions={}, falsification="f", created_ts=1.0)
    store.put_hypothesis(h)
    store.ensure_root("base")
    store.put_node(Node(id="n9", task_id=ctx.task_id, commit="", parent_id="root", strategy=Strategy.EXPLORE,
                        operator=Operator.DRAFT, family="architecture", mechanism_tag="m", diff_hash="DH", island="main",
                        status=NodeStatus.REJECTED, reject_reason="smoke_failed", hypothesis_id=h.id,
                        created_ts=1.0, updated_ts=1.0, notes="request=req-v3-7"))
    (st / INF.REISSUE_FILE).write_text('{"nodes": ["n9"]}')
    INF.reissue_from_file(ctx, store, _research(ctx, store), 5.0)
    assert store.get_node("n9").diff_hash == "" and store.find_by_diff_hash("DH") is None
