"""loop.core.meta: the computed report, proposal vetting (locked paths / keys refused), the held-out config backtest,
config / prompt versions with auto-revert and ``loop revert``, the lock hashes and the tick's breaker, the prompt A/B
arm, the wall-time clock, the CLI."""

from __future__ import annotations

import dataclasses
import random
import subprocess
from pathlib import Path

import pytest
import yaml

from loop import spec as SP
from loop import tick as TK
from loop.agents import briefs
from loop.agents.llm import FakeLLMClient
from loop.core import backtest as B
from loop.core import data as D
from loop.core import meta as MT
from loop.store import SqliteStore

from ..test_tick import _ledger
from .conftest import TASK


# ---------------------------------------------------------------------------------------------- synthetic points
def _point(i: int, m_c: float, fresh: float, *, rob=(0.0, 1e-6), fresh_rob=0.0, ts=None) -> MT.EvalPoint:
    p = B.Point(f"n{i}", 1e9 + (ts if ts is not None else i) * 3600, "explore", "f", "root", 3, 4, "confirm",
                m_c, 1e-5, m_c, m_c, 1e-6, m_c, 1e-6, 0.0, 1, fresh, 2e-6, fresh, 2e-6)
    return MT.EvalPoint(p, 4, rob, fresh_rob)


def _harm_points(fresh_rob_good: float = -0.001) -> list[MT.EvalPoint]:
    """12 good nodes (fresh +.004) whose robustness belief the harm limit blocks at q .2 but not at .35, 12 bad nodes."""
    rng = random.Random(3)
    good = [_point(i, 0.003 + rng.gauss(0, 3e-4), 0.004 + rng.gauss(0, 5e-4), rob=(-0.006, 1.6e-5),
                   fresh_rob=fresh_rob_good) for i in range(12)]
    bad = [_point(12 + i, 0.001 + rng.gauss(0, 3e-4), -0.002 + rng.gauss(0, 5e-4)) for i in range(12)]
    return good + bad


def test_backtest_accepts_a_measured_gain_and_refuses_no_change_and_a_worse_guard():
    pts, cur = _harm_points(), SP.tunable_values()
    bt = MT.backtest_config(pts, cur, {**cur, "acquire.harm_q": 0.35}, split_h=1e6)
    assert bt["accept"] and bt["new"]["value"] > bt["cur"]["value"] + 2 * bt["se"]["value"]
    same = MT.backtest_config(pts, cur, dict(cur), split_h=1e6)
    assert not same["accept"]
    harmful = MT.backtest_config(_harm_points(fresh_rob_good=-0.01), cur, {**cur, "acquire.harm_q": 0.35}, split_h=1e6)
    assert not harmful["accept"] and "harm" in harmful["worse"]


def test_backtest_split_is_the_last_hours_or_the_last_min_eval_points():
    pts = [_point(i, 0.001, 0.001, ts=i * 10) for i in range(30)]          # one point per 10 h
    assert len(MT.split(pts, 72.0)) == MT.MIN_EVAL
    assert len(MT.split(pts, 1e6)) == 30


# ---------------------------------------------------------------------------------------------- vetting
@pytest.mark.parametrize("p, why", [
    (MT.Proposal("config", "loop/core/test.py", 0.5), "LOCKED path"),
    (MT.Proposal("prompt", "loop/core/test.py", patch={"find": "ALPHA = 0.1", "replace": "ALPHA = 0.5"}), "LOCKED path"),
    (MT.Proposal("prompt", "loop/store.py", patch={"find": "a", "replace": "b"}), "LOCKED path"),
    (MT.Proposal("prompt", "prompts/../core/model.py", patch={"find": "a", "replace": "b"}), "LOCKED path"),
    (MT.Proposal("config", "test.ALPHA", 0.2), "LOCKED"),
    (MT.Proposal("config", "seeds.fresh_base", 5), "LOCKED"),
    (MT.Proposal("config", "costs.max_seeds", 8), "LOCKED"),
    (MT.Proposal("config", "acquire.harm_q", 0.9), "outside the schema bounds"),
    (MT.Proposal("config", "acquire.harm_q", "0.3"), "not a number"),
    (MT.Proposal("config", "acquire.foo", 1.0), "not a TUNABLE key"),
    (MT.Proposal("config", "acquire.max_of_k_beta", 2.5), "outside the schema bounds"),
    (MT.Proposal("prompt", "prompts/review.md", patch={"find": "a", "replace": "b"}), "not a patchable prompt"),
    (MT.Proposal("prompt", "prompts/ideation.md", patch={"find": "", "replace": "b"}), "find"),
    (MT.Proposal("code", "loop/core/acquire.py", "x"), "unknown proposal kind"),
    (MT.Proposal("prompt", "probe.py", patch={"find": "a", "replace": "b"}), "LOCKED path"),
])
def test_vet_refuses_locked_paths_keys_and_bad_values(p, why):
    assert why in MT.vet(p, surface_locked=("probe.py", "benchmarking/*"))


def test_vet_admits_a_tunable_in_bounds_and_a_patch_to_a_patchable_prompt():
    assert MT.vet(MT.Proposal("config", "acquire.harm_q", 0.3)) == ""
    assert MT.vet(MT.Proposal("config", "state/loop_config/acquire.harm_h_rob", 0.005)) == ""
    assert MT.vet(MT.Proposal("prompt", "prompts/ideation.md", patch={"find": "x", "replace": "y"})) == ""


# ---------------------------------------------------------------------------------------------- versions, apply
@pytest.fixture
def store(tmp_path):
    s = SqliteStore(tmp_path / "loop.db", TASK)
    yield s
    s.close()


def test_versions_accept_write_the_yaml_and_revert(store, tmp_path):
    ver = MT.Versions(store, tmp_path)
    assert ver.active("config")["v"] == 0 and ver.values() == SP.tunable_values()
    v = ver.accept_config({"acquire.harm_q": 0.3}, "why", {"accept": True})
    assert v == 1 and ver.values()["acquire.harm_q"] == 0.3
    doc = yaml.safe_load((tmp_path / "loop_config" / "v1.yaml").read_text())
    assert doc["parent"] == 0 and doc["values"]["acquire.harm_q"] == 0.3
    assert [r["state"] for r in store.config_versions("config")] == ["superseded", "active"]
    assert "v1 -> v0" in ver.revert("config", 0, "test")
    assert ver.active("config")["v"] == 0 and store.config_versions("config")[1]["state"] == "reverted"
    with pytest.raises(KeyError):
        ver.revert("config", 7, "x")
    assert store.events("meta_revert")[-1]["to"] == 0


def test_apply_installs_rank_keywords_and_the_max_of_k_beta():
    from loop.core import acquire as A
    core = TK.ModuleCore()
    vals = {**SP.tunable_values(), "acquire.harm_q": 0.3, "acquire.harm_h_rob": 0.005, "acquire.max_of_k_beta": 0.5}
    try:
        assert MT.apply(vals, core) == []
        assert core.harm == {"harm_q": 0.3, "harm_h": {"rob": 0.005}} and A.MAX_OF_K_BETA == 0.5
        assert MT.vet(MT.Proposal("config", "acquire.max_of_k_beta", 0.5)) == ""
    finally:
        MT.apply(SP.tunable_values(), core)
    assert A.MAX_OF_K_BETA == SP.TUNABLE["acquire.max_of_k_beta"].default == 0.0


def test_prompt_version_patch_and_the_ab_arm_serves_it(store, tmp_path, ctx):
    meta = MT.Meta(dataclasses.replace(ctx, state_dir=str(tmp_path)), store, locks=None)
    first = (MT.PROMPTS / "ideation.md").read_text().splitlines()[1]
    out = meta.accept({"diagnosis": "d", "proposals": [
        {"kind": "prompt", "path": "prompts/ideation.md", "patch": {"find": first, "replace": first + " ZZZ"},
         "rationale": "r", "predicted_effect": "e"},
        {"kind": "prompt", "path": "prompts/researcher.md", "patch": {"find": "never-present-text", "replace": "q"},
         "rationale": "r", "predicted_effect": "e"}]}, None)
    assert any("A/B arm on 30%" in x for x in out) and any("already running" in x for x in out)
    arm = tmp_path / "prompts" / "v1"
    assert "ZZZ" in (arm / "ideation.md").read_text() and (arm / "researcher.md").exists()
    sides = {r: meta.arm_for(r) for r in (f"req-{i}" for i in range(200))}
    share = sum(d == arm for d in sides.values()) / len(sides)
    assert 0.2 < share < 0.4 and meta.arm_for("req-3") == sides["req-3"]                  # deterministic
    assert briefs.in_prompt_dir(arm, briefs.render, "ideation").count("ZZZ") == 1
    assert "ZZZ" not in briefs.render("ideation")


def test_prompt_arm_is_rederived_on_top_of_the_current_prompts(store, tmp_path, ctx, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    for n in ("ideation.md", "researcher.md"):
        (base / n).write_text(f"{n} line one\nline two\n")
    monkeypatch.setattr(MT, "PROMPTS", base)
    meta = MT.Meta(dataclasses.replace(ctx, state_dir=str(tmp_path / "st")), store, locks=None)
    meta.accept({"diagnosis": "d", "proposals": [{"kind": "prompt", "path": "prompts/ideation.md",
                 "patch": {"find": "line two", "replace": "line two ZZZ"}, "rationale": "r", "predicted_effect": "e"}]}, None)
    arm = tmp_path / "st" / "prompts" / "v1"
    (base / "ideation.md").write_text("ideation.md NEW BUDGET\nline two\n")             # the base moved on
    (base / "researcher.md").write_text("researcher.md NEW\n")
    assert meta.versions.prompt_dir(1) == arm
    assert (arm / "ideation.md").read_text() == "ideation.md NEW BUDGET\nline two ZZZ\n"
    assert (arm / "researcher.md").read_text() == "researcher.md NEW\n"


# ---------------------------------------------------------------------------------------------- the review + acceptance
def test_review_accepts_one_backtested_config_and_refuses_the_locked_ones(store, tmp_path, ctx):
    meta = MT.Meta(dataclasses.replace(ctx, state_dir=str(tmp_path)), store, locks=None)
    meta.points = _harm_points()
    meta.cfg = dataclasses.replace(meta.cfg, split_h=1e6)
    out = meta.accept({"diagnosis": "the harm limit blocks good tests", "proposals": [
        {"kind": "config", "path": "acquire.harm_q", "value": 0.35, "rationale": "r", "predicted_effect": "e"},
        {"kind": "config", "path": "test.ALPHA", "value": 0.2, "rationale": "r", "predicted_effect": "e"},
        {"kind": "prompt", "path": "loop/core/test.py", "patch": {"find": "A", "replace": "B"},
         "rationale": "r", "predicted_effect": "e"}]}, None)
    assert out[0].startswith("review: the harm limit")
    assert any(x.startswith("ACCEPTED config v1: acquire.harm_q 0.2 -> 0.35") for x in out)
    assert sum(x.startswith("refused") for x in out) == 2
    verdicts = [(e["path"], e["verdict"]) for e in store.events("meta_proposal")]
    assert ("test.ALPHA", "refused") in verdicts and ("loop/core/test.py", "refused") in verdicts
    assert meta.versions.values()["acquire.harm_q"] == 0.35
    assert store.events("meta_cycle")[-1]["summary"].count("refused") == 2


def test_the_next_report_auto_reverts_a_version_whose_guard_left_its_bounds(store, tmp_path, ctx):
    meta = MT.Meta(dataclasses.replace(ctx, state_dir=str(tmp_path)), store, locks=None)
    v = meta.versions.accept_config({"acquire.harm_q": 0.35}, "r", {})
    store.set_state(MT.K_CHECK, {"v": v, "guards": {"coverage80": 0.8}})
    bad = MT.MetaReport(ts=1.0, since=0.0, calibration={"coverage80": 0.4, "slope": 1.0, "rho": 0.4})
    assert "AUTO-REVERT config v1 -> v0" in meta.check_revert(bad)[0]
    assert meta.versions.active("config")["v"] == 0 and store.get_state(MT.K_CHECK) == {}
    v = meta.versions.accept_config({"acquire.harm_q": 0.3}, "r", {})
    store.set_state(MT.K_CHECK, {"v": v, "guards": {"coverage80": 0.8}})
    ok = MT.MetaReport(ts=2.0, since=1.0, calibration={"coverage80": 0.8, "slope": 1.0, "rho": 0.4})
    assert meta.check_revert(ok) == [f"config v{v} kept: guards in bounds"] and meta.versions.active("config")["v"] == v


# ---------------------------------------------------------------------------------------------- locks + breaker
def _git_repo(path: Path) -> Path:
    (path / "loop" / "core").mkdir(parents=True)
    (path / "loop" / "core" / "test.py").write_text("ALPHA = 0.1\n")
    for args in (["init", "-q"], ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)
    return path


def test_locks_compare_git_blob_hashes_with_the_deployed_head(tmp_path, store, ctx):
    repo = _git_repo(tmp_path / "repo")
    locks = MT.Locks(repo, ("loop/core/test.py",))
    assert locks.blobs["loop/core/test.py"] == MT.blob_hash(b"ALPHA = 0.1\n") and locks.verify() == []
    meta = MT.Meta(dataclasses.replace(ctx, state_dir=str(tmp_path)), store, locks=locks)
    assert meta.guard() is None
    (repo / "loop" / "core" / "test.py").write_text("ALPHA = 0.5\n")
    assert "BREAKER" in meta.guard() and store.get_state(MT.K_BREAKER)["active"]
    assert store.events("lock_breach")[-1]["files"] == ["loop/core/test.py"]
    (repo / "loop" / "core" / "test.py").write_text("ALPHA = 0.1\n")
    assert meta.guard() is None and not store.get_state(MT.K_BREAKER)["active"]
    assert MT.Locks(tmp_path / "not-a-repo").verify() == []                     # no git: unavailable, never trips


def test_a_tripped_breaker_stops_every_launch_and_session(ctx, tmp_path):
    db = _ledger(tmp_path / "live.db")
    c = dataclasses.replace(ctx, task_id=TASK, cfg=dataclasses.replace(ctx.cfg, slots=64))
    store = SqliteStore(db, TASK, weights=c.cfg.weights)
    repo = _git_repo(tmp_path / "repo")
    meta = MT.Meta(c, store, locks=MT.Locks(repo, ("loop/core/test.py",)))
    t = [2e9]
    tick = TK.Tick(c, store, TK.DryRunAdapter(store, lambda: t[0]), TK.ModuleCore(), db_path=db, clock=lambda: t[0],
                   dry=True, meta=meta)
    (repo / "loop" / "core" / "test.py").write_text("ALPHA = 0.5\n")
    rep = tick.step()
    assert rep.launched == [] and any(x.startswith("BREAKER") for x in rep.lines)
    assert not any(x.startswith("would brief") for x in rep.lines)
    (repo / "loop" / "core" / "test.py").write_text("ALPHA = 0.1\n")
    rep = tick.step()
    assert rep.launched and not any(x.startswith("BREAKER") for x in rep.lines)
    tick.close()
    store.close()


# ---------------------------------------------------------------------------------------------- report + cycle
def test_wall_clock_counts_only_loop_time(store, tmp_path, ctx):
    meta = MT.Meta(dataclasses.replace(ctx, state_dir=str(tmp_path)), store, locks=None)
    meta.cfg = dataclasses.replace(meta.cfg, every_h=1.0)
    t = 1e9
    assert not meta.due(t)
    for _ in range(5):                    # 5 ticks 10 min apart: 50 min
        t += 600
        assert not meta.due(t)
    t += 86400                            # the loop was down a day: counts MAX_GAP_S only
    assert meta.due(t) and not meta.due(t + 60)


def test_report_on_the_ledger_fixture_and_one_cycle_with_a_fake_review(ctx, tmp_path):
    db = _ledger(tmp_path / "ledger.db")
    c = dataclasses.replace(ctx, task_id=TASK, state_dir=str(tmp_path))
    store = SqliteStore(db, TASK)
    led = D.load(db, TASK)
    rep, pts = MT.report(led, store, now=2e9)
    assert rep.incumbent and rep.sources and rep.tests["closed"] > 0 and set(rep.gpu_h) == set(MT.PURPOSES)
    assert "panel" in rep.noise and rep.plateau and "posterior" not in rep.errors
    md = rep.markdown()
    assert "## Yield per source" in md and "## GPU-h by purpose" in md and "## Noise" in md
    seen = {}
    llm = FakeLLMClient({"review": lambda **call: seen.update(call) or {"diagnosis": "nothing to change", "proposals": []}})
    meta = MT.Meta(c, store, llm=llm, clock=lambda: 2e9, locks=None)
    meta.points = pts
    lines = meta.cycle(led, llm=llm, wait=True)
    assert lines[0].startswith("meta: meta report ") and "meta: review: nothing to change" in lines
    assert "## TUNABLE config" in seen["prompt"] and "acquire.harm_q = 0.2" in seen["prompt"]
    assert "loop/core/test.py" in seen["prompt"] and "{report}" not in seen["prompt"]
    assert list((tmp_path / "meta").glob("*.md")) and store.get_state(MT.K_LAST)["ts"] == 2e9
    assert store.events("meta_cycle")
    store.close()


def test_report_on_a_history_with_no_pass_yet_completes(ctx, tmp_path, monkeypatch):
    """Live 11-08 (nanobrain, thin ledger): no switched PASS yet -> plateau median = inf -> round(inf) raised
    OverflowError out of report() every meta cycle. The median is None then; a finite one is still rounded."""
    db = _ledger(tmp_path / "ledger.db")
    store = SqliteStore(db, TASK)
    led = D.load(db, TASK)
    rep, _ = MT.report(led, store, now=2e9)
    assert isinstance(rep.plateau["median"], int) and rep.plateau["passes"] > 0      # nanopath: unchanged
    real = MT.ID.plateau
    assert real(led, []).median == float("inf")
    monkeypatch.setattr(MT.ID, "plateau", lambda led, judged: real(led, []))
    rep, _ = MT.report(led, store, now=2e9)
    assert rep.plateau["median"] is None and rep.plateau["passes"] == 0 and "plateau" not in rep.errors
    assert "(no pass yet)" in rep.markdown()
    store.close()


def test_cli_meta_and_revert(fake_loop_spec, tmp_path, capsys):
    from loop.__main__ import main
    db = _ledger(tmp_path / "ledger.db")
    assert main(["meta", "--task", str(fake_loop_spec), "--db", str(db), "--task-id", TASK]) == 0
    out = capsys.readouterr().out
    assert "# Meta report" in out and "meta report" in out
    store = SqliteStore(db, TASK)
    MT.Versions(store, tmp_path).accept_config({"acquire.harm_q": 0.3}, "r", {})
    store.close()
    assert main(["revert", "v0", "--task", str(fake_loop_spec), "--db", str(db), "--task-id", TASK]) == 0
    assert "config v1 -> v0" in capsys.readouterr().out
    assert main(["revert", "v9", "--task", str(fake_loop_spec), "--db", str(db), "--task-id", TASK]) == 2


def test_llm_sessions_wall_pairs_a_reissued_request_with_its_own_end(store):
    """Live 10-21: a request re-issued after a restart has two starts and two ends; the report said '-8.18 h wall'
    because the latest start was paired with the FIRST end."""
    def ev(ts, name, **f):
        import json
        store._db.execute("INSERT INTO log (task_id, ts, event, fields) VALUES (?,?,?,?)", (TASK, ts, name, json.dumps(f)))
    ev(100.0, "research_started", request_id="r1")
    ev(160.0, "research_failed", request_id="r1")           # killed by the restart
    ev(1000.0, "research_started", request_id="r1")          # re-issued
    ev(1600.0, "intake", request_id="r1")
    got = MT.llm_sessions(store, 0.0)
    assert got["started"] == 1 and got["finished"] == 1 and got["failed"] == 0
    assert got["wall_h"] > 0 and abs(got["wall_h"] - 600 / 3600) < 0.01


def test_tests_section_reports_local_tier_passes_apart_from_gs2():
    """Review 11-10 S9: type1_sim prices gs2 only, so the report splits the local re-root tier's passes out and says so."""
    t = {"closed": 4, "window": 4, "passed": 3, "pass_rate": 0.75, "mean_n": 5.0, "harm_rate": 0.0,
         "local_passes": 1, "gs2_passed": 2, "type1_sim": 0.05, "type1_sim_exact": 0.05}
    md = MT.render_markdown(MT.MetaReport(ts=2e9, since=0.0, tests=t))
    assert "1 came from the LOCAL re-root tier" in md and "2 from gs2" in md and "prices gs2 only" in md
    md0 = MT.render_markdown(MT.MetaReport(ts=2e9, since=0.0, tests={**t, "local_passes": 0, "gs2_passed": 3}))
    assert "LOCAL re-root tier" not in md0
