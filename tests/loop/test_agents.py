"""loop.agents — researcher, analyst, briefs, LLM clients. Fakes only:
no network, no SLURM, no git (an in-file adapter double works on temp dirs)."""

from __future__ import annotations

import dataclasses
import difflib
import json
import shutil
import sys
import types
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from kernel.model import TaskSpec
from kernel.stats import NoiseModel
from loop.agents import FakeLLMClient, LLMAnalyst, LLMResearcher, brief
from loop.agents import llm as llm_mod
from loop.agents.llm import AgentRun, BackendUnavailable, SchemaError, SdkLLMClient, validate
from loop.agents import briefs
from loop.agents.analyst import probe_violation
from loop.agents.researcher import WorktreeTools, diff_hash
from loop.contracts import (PANEL, ROOT_NODE_ID, AxisPrediction, CandidateRequest, HandoffNote,
                            HypothesisEvidence, HypothesisKind, HypothesisRecord,
                            HypothesisStatus, InitCompat, LeverEffect, LeverTags, Node,
                            NodePosterior, NodeStatus, Operator, Posterior, Postmortem, Run,
                            RunStatus, SmokeResult, Stage, Strategy, WorldSnapshot, Worktree)
from loop.spec import FamilyRule, LoopConfig, LoopContext, NoveltyCfg, ResearcherCfg
from loop.select.intake import path_matches

REPO = Path(__file__).resolve().parents[2]
TID = "fake-offline"


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class MemStore:
    """Minimal in-memory Store (read side + table_dump)."""

    task_id = TID

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._runs: list[Run] = []
        self._hyps: dict[str, HypothesisRecord] = {}
        self._pms: list[Postmortem] = []
        self._handoffs: list[HandoffNote] = []
        self._reviews: list = []
        self.logged: list[tuple[str, dict]] = []

    def put_node(self, n: Node) -> None: self._nodes[n.id] = n
    def get_node(self, i: str) -> Node | None: return self._nodes.get(i)

    def nodes(self, *, status=None, island=None) -> list[Node]:
        st = set(status) if status is not None else None
        return [n for n in self._nodes.values() if (st is None or n.status in st)
                and (island is None or n.island == island)]

    def runs(self, *, node_id=None, status=None, stage=None) -> list[Run]:
        st = set(status) if status is not None else None
        return [r for r in self._runs if (node_id is None or r.node_id == node_id)
                and (st is None or r.status in st) and (stage is None or r.stage == stage)]

    def get_hypothesis(self, i: str): return self._hyps.get(i)

    def hypotheses(self, *, kind=None, status=None):
        st = set(status) if status is not None else None
        return [h for h in self._hyps.values() if (kind is None or h.kind == kind)
                and (st is None or h.status in st)]

    def postmortems(self, *, node_id=None):
        return [p for p in self._pms if node_id is None or p.node_id == node_id]

    def handoffs(self, *, node_id=None):
        return [h for h in self._handoffs if node_id is None or h.node_id == node_id]

    def reviews(self): return list(self._reviews)
    def find_by_diff_hash(self, h: str): return next((n for n in self._nodes.values() if n.diff_hash == h), None)
    def get_state(self, k: str, default=None): return default
    def log(self, event: str, **fields) -> None: self.logged.append((event, fields))

    def table_dump(self, *, max_rows: int = 500) -> str:
        return "LEDGER\n" + "\n".join(f"{n.id} {n.family} {n.status.value}"
                                      for n in list(self._nodes.values())[:max_rows])


CONFIG_TEXT = ("# fake config\ntrain:\n  lr: 0.001   # peak LR\n  steps: 10\n"
               "aug:\n  p: 0.5\n  scale: [0.4, 1.0]\neval:\n  dataset: x\n")


class DirAdapter:
    """ExecutionAdapter double over plain directories (no git); ``locks`` are
    ``file::Qual.name`` AST locks judged like the nanopath adapter's."""

    def __init__(self, template: Path, scratch: Path) -> None:
        self.template, self.scratch = template, scratch
        self.locks: tuple[str, ...] = ()
        self.smoke_ok, self.init_ok = True, True
        self.merge_result: tuple[bool, str] = (True, "")
        self.reaped: list[str] = []
        self.merged: list[str] = []
        self.commits = 0
        self.smokes = 0

    def root_commit(self) -> str: return "c0"

    def prepare_worktree(self, *, parent_commit: str, name: str) -> Worktree:
        dst = self.scratch / name
        shutil.copytree(self.template, dst)
        return Worktree(path=str(dst), branch=name, base_commit=parent_commit)

    def reap_worktree(self, wt: Worktree) -> None:
        self.reaped.append(wt.path)
        shutil.rmtree(wt.path, ignore_errors=True)

    def diff(self, wt: Worktree):
        out, paths = [], []
        for f in sorted(Path(wt.path).rglob("*")):
            if not f.is_file():
                continue
            rel = str(f.relative_to(wt.path))
            old = self.template / rel
            a = old.read_text().splitlines(True) if old.exists() else []
            b = f.read_text().splitlines(True)
            if a != b:
                paths.append(rel)
                out += difflib.unified_diff(a, b, f"a/{rel}", f"b/{rel}")
        from loop.adapters import flatten
        from loop.adapters.nanopath_v2 import locked_source
        paths += [l for l in self.locks if l.split("::")[0] in paths and locked_source(
            (self.template / l.split("::")[0]).read_text(), l) != locked_source(
            (Path(wt.path) / l.split("::")[0]).read_text(), l)]
        old = flatten(yaml.safe_load((self.template / "fake_config.yaml").read_text()))
        new = flatten(yaml.safe_load((Path(wt.path) / "fake_config.yaml").read_text()))
        return "".join(out), tuple(paths), {k: new.get(k) for k in set(old) | set(new) if old.get(k) != new.get(k)}

    def merge(self, wt: Worktree, *, other_commit: str):
        self.merged.append(other_commit)
        return self.merge_result

    def smoke(self, wt: Worktree) -> SmokeResult:
        self.smokes += 1
        return SmokeResult(ok=self.smoke_ok, reason="ok" if self.smoke_ok else "nan loss",
                           log_tail="step 1 loss 2.3")

    def init_compat(self, wt: Worktree, *, parent_commit: str) -> InitCompat:
        return InitCompat(ok=self.init_ok, reason="" if self.init_ok else "shape mismatch")

    def commit(self, wt: Worktree, *, message: str) -> str:
        self.commits += 1
        return f"c{self.commits}"


def _ctx(**researcher) -> LoopContext:
    spec = TaskSpec.load(REPO / "tasks" / "fake.yaml")
    cfg = LoopConfig(execution_adapter="tests:DirAdapter", axes=("offline", "robust"),
                     weights={"offline": 0.5, "robust": 0.5}, panel_sd=0.004,
                     axis_sd={"offline": 0.004, "robust": 0.006},
                     researcher=ResearcherCfg(**researcher) if researcher else ResearcherCfg(),
                     novelty=NoveltyCfg(sweep_redirect=False),   # knob-move scripts: tests/loop/test_novelty.py
                     families={"optim": FamilyRule(keys=("train.*",), paths=("fake_model.py",)),
                               "augment": FamilyRule(keys=("aug.*",), paths=("fake_aug.py",))})
    return LoopContext(spec=spec, cfg=cfg, noise=NoiseModel.from_spec(spec), task_id=spec.name,
                       spec_path=str(REPO / "tasks" / "fake.yaml"), state_dir="")


def _snapshot(nodes: dict[str, NodePosterior] | None = None, **kw) -> WorldSnapshot:
    return WorldSnapshot(
        task_id=kw.get("task_id", TID), n_scored=kw.get("n_scored", 12),
        sigma={PANEL: 0.004, "offline": 0.004, "robust": 0.006}, variance={},
        nodes=nodes or {}, incumbent_id=ROOT_NODE_ID, family_means={},
        levers=(LeverEffect("effective_lr", PANEL, -0.004, 0.0015, 0.01, 7),
                LeverEffect("regularization", PANEL, -0.003, 0.001, 0.02, 9)),
        hypotheses=kw.get("hypotheses", {}))


def _node(i: str, *, status=NodeStatus.SCREENED_OUT, family="optim", task_id=TID, **kw) -> Node:
    return Node(id=i, task_id=task_id, commit=kw.pop("commit", f"sha-{i}"), parent_id=ROOT_NODE_ID,
                strategy=Strategy.EXPLORE, operator=Operator.DRAFT,
                family=family, mechanism_tag=kw.pop("tag", "lr_decay"), diff_hash="h",
                status=status, **kw)


def _post(i: str, mean: float) -> NodePosterior:
    p = Posterior(mean, 0.003, 2)
    return NodePosterior(node_id=i, increment=p, utility=p, axes={"offline": p, "robust": p},
                         p_positive=0.2, p_beats_incumbent=0.1, vetoed=False)


def _request(op=Operator.DRAFT, parents=(ROOT_NODE_ID,), **kw) -> CandidateRequest:
    return CandidateRequest(id=kw.pop("id", "req1"), task_id=kw.pop("task_id", TID),
                            strategy=Strategy.EXPLORE, operator=op, island="main",
                            parent_ids=tuple(parents), brief=kw.pop("brief", "try a cosine LR"),
                            **kw)


FINISH = {
    "hypothesis": {"claim": "lower LR avoids over-training", "mechanism_tag": "lr_down",
                   "family": "optim", "falsification": "panel delta <= 0 on 2 seeds",
                   "predictions": {"panel": {"sign": 1, "size": 0.003},
                                   "offline": {"sign": 1, "size": 0.002}}},
    "levers": {"effective_lr": -1, "regularization": 0},
    "handoff": {"actions": "lr 1e-3 -> 5e-4", "learnings": "warmup is 10%", "concerns": "none"},
}


@pytest.fixture
def env(tmp_path):
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "fake_model.py").write_text("LR = 1e-3\nWD = 0.05\n\n\ndef score_fn(x):\n    return x\n")
    (tpl / "fake_config.yaml").write_text(CONFIG_TEXT)
    (tpl / "fake_probe.py").write_text("SCORE = 1\n")
    (tmp_path / "wt").mkdir()
    return DirAdapter(tpl, tmp_path / "wt"), MemStore()


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------

def test_researcher_end_to_end(env):
    adapter, store = env
    llm = FakeLLMClient({"researcher": [{
        "tool_calls": [("read_file", {"path": "fake_model.py"}),
                       ("edit_file", {"path": "fake_model.py", "old": "LR = 1e-3", "new": "LR = 5e-4"}),
                       ("smoke", {})],
        "finish": FINISH}]})
    res = LLMResearcher(_ctx(), llm, adapter).research(_request(), store, _snapshot())
    assert res.status == "ok", res.error
    c = res.candidate
    assert c.commit == "c1" and c.parent_commit == "c0" and c.changed_paths == ("fake_model.py",)
    assert "+LR = 5e-4" in c.diff_text and c.diff_hash == diff_hash(c.diff_text)
    h = c.hypothesis
    assert h.kind is HypothesisKind.TRIAL and h.task_id == TID and h.source == "researcher"
    assert h.predictions[PANEL] == AxisPrediction(1, 0.003)
    assert dict(h.levers.directions) == {"effective_lr": -1}
    assert c.handoff.learnings == "warmup is 10%" and c.init_compat.ok
    assert adapter.smokes == 1                       # agent's smoke reused (not dirty)
    assert adapter.reaped                             # worktree always reaped
    call = llm.calls[0]
    assert "try a cosine LR" in call["prompt"] and "Ledger (newest first" in call["prompt"]
    assert "effective_lr" in call["prompt"] and call["max_turns"] == 48
    assert call["timeout_s"] == 20 * 60
    out = dict((n, o) for n, _, o in call["tool_log"])
    assert "LR = 1e-3" in out["read_file"] and out["edit_file"].startswith("edited")


def test_banned_commands_refused(env):
    adapter, _ = env
    wt = adapter.prepare_worktree(parent_commit="c0", name="b")
    tools = WorktreeTools(adapter, wt, writable=("fake_model.py",), locked=("fake_probe.py",))
    for cmd in ("git reset --hard HEAD", "git checkout main", "git stash", "git clean -fd",
                "git rebase x", "git commit -am x", "sbatch run.sh", "srun -N1 true",
                "scancel 123", "cd .. && git   reset --hard"):
        assert tools("bash", {"command": cmd}).startswith("REFUSED"), cmd
    assert tools("bash", {"command": "echo hi && git status"}).startswith("exit=")  # allowed


def test_locked_and_unwritable_edits_refused(env):
    adapter, _ = env
    wt = adapter.prepare_worktree(parent_commit="c0", name="l")
    tools = WorktreeTools(adapter, wt, writable=("fake_model.py", "fake_config.yaml"),
                          locked=("fake_probe.py", "benchmarking/"))
    assert "LOCKED" in tools("edit_file", {"path": "fake_probe.py", "old": "SCORE = 1", "new": "SCORE = 2"})
    assert "LOCKED" in tools("write_file", {"path": "benchmarking/x.py", "content": "x"})
    assert "not in surface.writable" in tools("write_file", {"path": "new.py", "content": "x"})
    with pytest.raises(PermissionError):
        tools("read_file", {"path": "../../x"})
    assert (Path(wt.path) / "fake_probe.py").read_text() == "SCORE = 1\n"
    assert path_matches("benchmarking/a/b.py", "benchmarking/")
    assert not path_matches("benchmarkingx.py", "benchmarking/")


def test_locked_path_changed_via_bash_is_a_surface_violation(env):
    adapter, store = env
    llm = FakeLLMClient({"researcher": [{
        "tool_calls": [("bash", {"command": "echo 'SCORE = 9' > fake_probe.py"}),
                       ("finish", FINISH), ("finish", FINISH)],        # bounced twice, then let through
        "finish": FINISH}]})
    res = LLMResearcher(_ctx(), llm, adapter).research(_request(), store, _snapshot())
    # c4j (review M1): bash refuses a write onto a locked path; the file is untouched, so the finish has no diff
    assert llm.calls[0]["tool_log"][0][2].startswith("REFUSED (LOCKED)")
    assert res.status != "ok" and "empty_diff" in res.error and adapter.commits == 0


def test_debug_cap_per_lineage(env):
    adapter, store = env
    store.put_node(_node("n1", status=NodeStatus.CRASHED, debug_depth=3))
    llm = FakeLLMClient({"researcher": []})
    res = LLMResearcher(_ctx(), llm, adapter).research(
        _request(Operator.DEBUG, parents=("n1",)), store, _snapshot())
    assert res.status == "failed" and res.error.startswith("debug_cap")
    assert llm.calls == [] and adapter.reaped == []   # refused before any worktree/LLM

    store.put_node(_node("n2", status=NodeStatus.CRASHED, debug_depth=2))
    llm2 = FakeLLMClient({"researcher": [{"tool_calls": [
        ("edit_file", {"path": "fake_model.py", "old": "WD = 0.05", "new": "WD = 0.01"})],
        "finish": FINISH}]})
    res2 = LLMResearcher(_ctx(), llm2, adapter).research(
        _request(Operator.DEBUG, parents=("n2",)), store, _snapshot())
    assert res2.status == "ok" and "Debug depth 3/3" in llm2.calls[0]["prompt"]
    assert adapter.smokes == 1                         # dirty after edit -> harness smoke


def test_smoke_and_init_compat_failures_are_typed(env):
    adapter, store = env
    script = lambda **_: {"tool_calls": [("edit_file", {"path": "fake_model.py",
                                                       "old": "LR = 1e-3", "new": "LR = 2e-3"})],
                          "finish": FINISH}
    adapter.smoke_ok = False
    r = LLMResearcher(_ctx(), FakeLLMClient({"researcher": script}), adapter).research(
        _request(), store, _snapshot())
    assert r.status == "smoke_failed" and r.candidate is not None and adapter.commits == 0
    adapter.smoke_ok, adapter.init_ok = True, False
    r = LLMResearcher(_ctx(), FakeLLMClient({"researcher": script}), adapter).research(
        _request(id="req2"), store, _snapshot())
    assert r.status == "init_incompatible" and r.candidate.init_compat.reason == "shape mismatch"


def test_no_finish_empty_diff_timeout_and_bad_levers(env):
    adapter, store = env
    # no finish
    r = LLMResearcher(_ctx(), FakeLLMClient({"researcher": [{"tool_calls": []}]}), adapter
                      ).research(_request(), store, _snapshot())
    assert r.status == "failed" and r.error.startswith("no_finish")
    # empty diff: finish bounces twice (check says empty_diff), the third is let through
    llm = FakeLLMClient({"researcher": [{"tool_calls": [("finish", FINISH)] * 2, "finish": FINISH}]})
    r = LLMResearcher(_ctx(), llm, adapter).research(_request(), store, _snapshot())
    assert r.status == "failed" and r.error.startswith("empty_diff")
    assert "REJECT empty_diff" in llm.calls[0]["tool_log"][0][2]
    # timeout: zero-minute budget -> tools refused, no finish
    llm = FakeLLMClient({"researcher": [{"tool_calls": [("list_dir", {})]}]})
    r = LLMResearcher(_ctx(timeout_min=0.0), llm, adapter).research(_request(), store, _snapshot())
    assert r.status == "timeout"
    assert "TIME LIMIT" in llm.calls[0]["tool_log"][0][2]
    # unknown lever -> finish rejected by schema, never accepted
    bad = json.loads(json.dumps(FINISH))
    bad["levers"] = {"vibes": 1}
    llm = FakeLLMClient({"researcher": [{"finish": bad}]})
    r = LLMResearcher(_ctx(), llm, adapter).research(_request(), store, _snapshot())
    assert r.status == "failed" and "REJECTED" in llm.calls[0]["tool_log"][0][2]


def test_crossover_merges_second_parent_and_shows_conflicts(env):
    adapter, store = env
    store.put_node(_node("a", status=NodeStatus.CONFIRMED, family="optim"))
    store.put_node(_node("b", status=NodeStatus.CONFIRMED, family="augment"))
    adapter.merge_result = (False, "<<<<<<< fake_model.py")
    llm = FakeLLMClient({"researcher": [{"tool_calls": [
        ("edit_file", {"path": "fake_model.py", "old": "WD = 0.05", "new": "WD = 0.02"})],
        "finish": FINISH}]})
    r = LLMResearcher(_ctx(), llm, adapter).research(
        _request(Operator.CROSSOVER, parents=("a", "b")), store, _snapshot())
    assert r.status == "ok" and adapter.merged == ["sha-b"]
    assert "<<<<<<< fake_model.py" in llm.calls[0]["prompt"]
    # both are root children: the diff (and the paired control) is taken against root, not a
    assert r.candidate.parent_commit == adapter.root_commit()
    store.put_node(_node("c", status=NodeStatus.NOT_CONFIRMED, family="optim"))       # a stack merges every leg
    adapter.merged, adapter.merge_result = [], (True, "")
    llm = FakeLLMClient({"researcher": [{"tool_calls": [
        ("edit_file", {"path": "fake_model.py", "old": "WD = 0.05", "new": "WD = 0.03"})], "finish": FINISH}]})
    r = LLMResearcher(_ctx(), llm, adapter).research(
        _request(Operator.CROSSOVER, parents=("a", "b", "c")), store, _snapshot())
    assert r.status == "ok", r.error
    assert adapter.merged == ["sha-b", "sha-c"] and "(b, c) has been merged" in llm.calls[0]["prompt"]


# ---------------------------------------------------------------------------
# Briefs
# ---------------------------------------------------------------------------

def test_briefs_render_and_guard(env):
    _, store = env
    snap = _snapshot()
    lit = brief("literature", store, snap, family="optim")
    assert "optim" in lit and "LEDGER" not in lit        # the ledger lives in the context pack
    assert "w [optim" not in brief("frontier", store, snap)   # no winner yet: literature
    store.put_node(_node("w", status=NodeStatus.CONFIRMED))
    assert "w [optim/lr_decay]" in brief("frontier", store, snap)
    store._pms.append(Postmortem(id="pm1", task_id=TID, node_id="x", verdict="loss",
                                 explanation="lesson one", mechanism_reasoning="", levers=LeverTags()))
    assert "lesson one" in brief("failure", store, snap)
    with pytest.raises(ValueError):
        brief("migration", store, snap)


def test_probe_briefs(env):
    _, store = env
    store._hyps["m1"] = HypothesisRecord(
        id="m1", task_id=TID, kind=HypothesisKind.META, claim="over-regularised",
        mechanism_tag="train_weaker", family="cross",
        predictions={PANEL: AxisPrediction(1, 0.004)}, falsification="no gain",
        levers=LeverTags({"effective_lr": -1, "regularization": -1}), source="review")
    txt = brief("probe", store, _snapshot(), hypothesis_id="m1", probe="halve WD")
    assert "effective_lr DOWN" in txt and "regularization DOWN" in txt and "halve WD" in txt
    assert "shorter warmup" in brief("probe", store, _snapshot(), hypothesis_id=None,
                                     probe="shorter warmup")       # a global probe


# ---------------------------------------------------------------------------
# Analyst
# ---------------------------------------------------------------------------

def _scored_store(store: MemStore) -> None:
    store.put_node(_node("bad", status=NodeStatus.SCREENED_OUT, hypothesis_id="h1",
                         levers=LeverTags({"regularization": 1})))
    store._hyps["h1"] = HypothesisRecord(
        id="h1", task_id=TID, kind=HypothesisKind.TRIAL, claim="more WD helps",
        mechanism_tag="wd_up", family="optim", predictions={PANEL: AxisPrediction(1, 0.003)},
        falsification="panel <= 0")
    for seed, (b, n) in {11: (0.60, 0.594), 12: (0.61, 0.603)}.items():
        store._runs.append(Run(id=f"root:baseline:{seed}", task_id=TID, node_id=ROOT_NODE_ID,
                               seed=seed, stage=Stage.BASELINE, status=RunStatus.DONE,
                               scores={"offline": b, "robust": b}, aggregate=b))
        store._runs.append(Run(id=f"bad:screen:{seed}", task_id=TID, node_id="bad", seed=seed,
                               stage=Stage.SCREEN, status=RunStatus.DONE,
                               scores={"offline": n, "robust": b}, aggregate=n))



def test_context_pack_lists_work_in_progress(env):
    from loop.agents import briefs as B
    from loop.contracts import K_INFLIGHT
    _, store = env
    state = {K_INFLIGHT: [{"id": "req1", "strategy": "explore", "family": "optim"},
                          {"id": "req2", "strategy": "explore", "family": "aug"}],
             "config_diff:q": {"data.tissue_thresh": 0.25}}
    store.get_state = lambda k, default=None: state.get(k, default)
    store.put_node(_node("q", status=NodeStatus.SCREENING, family="data", tag="tissue_filter"))
    store.put_node(_node("old", status=NodeStatus.SCREENED_OUT, tag="closed_tag"))
    pack = B.context_pack(_ctx(), store, _snapshot(), request_id="req1")
    sec = pack.split("### Already in progress")[1].split("###")[0]
    assert "family aug" in sec and "family optim" not in sec             # not its own request
    assert "q screening: data/tissue_filter (data.tissue_thresh=0.25)" in sec and "closed_tag" not in sec
    assert "DATA BANK" not in pack
    c = _ctx()
    c = dataclasses.replace(c, cfg=dataclasses.replace(c.cfg, data_bank=("/data/x/arrow: subset X",)))
    assert "DATA BANK" in (p2 := B.context_pack(c, store, _snapshot(), request_id="req1")) and "/data/x/arrow: subset X" in p2


def test_big_review_meta_hypotheses_and_probes(env):
    _, store = env
    _scored_store(store)
    store._pms.append(Postmortem(id="pm1", task_id=TID, node_id="bad", verdict="loss",
                                 explanation="over-regularised", mechanism_reasoning="",
                                 levers=LeverTags({"regularization": 1})))
    store.put_node(_node("w", status=NodeStatus.CONFIRMED, island="i1"))
    ev = {"h1": HypothesisEvidence("h1", 0.4, 6.0, 2, HypothesisStatus.TESTING)}
    llm = FakeLLMClient({"review": [{
        "summary": "regularisers hurt; weaker training helps",
        "findings": ["3/3 WD increases lost (bad)"],
        "meta_hypotheses": [
            {"claim": "model is over-regularised -> train weaker", "mechanism_tag": "train_weaker",
             "levers": {"effective_lr": -1, "regularization": -1, "unknown": 1},
             "predicted_sign": 1, "predicted_size": 0.004, "falsification": "WD-down loses",
             "probes": ["halve WD", "lr x0.7"]},
            {"claim": "vague", "mechanism_tag": "x", "levers": {}, "predicted_sign": 1,
             "falsification": "n/a"}],
        "probe_briefs": ["shorter warmup"],
        "explore_bias": {"optim": 0.5, "augment": -0.9, "data": 0.001}}]})
    rep, metas = LLMAnalyst(_ctx(), llm).big_review(store, _snapshot(hypotheses=ev),
                                                   trigger="stall")
    assert rep.explore_bias == {"optim": 0.5, "augment": -0.9, "data": 0.001}   # capped where used
    assert len(metas) == 1                              # lever-less meta dropped
    m = metas[0]
    assert m.kind is HypothesisKind.META and m.source == "review" and m.review_id == rep.id
    assert dict(m.levers.directions) == {"effective_lr": -1, "regularization": -1}
    assert m.predictions[PANEL] == AxisPrediction(1, 0.004) and m.task_id == TID
    assert rep.meta_hypothesis_ids == (m.id,) and rep.trigger == "stall" and rep.n_evals_at == 12
    assert (None, "shorter warmup") in rep.probe_briefs
    assert [q for q in rep.probe_briefs if q[0] == m.id] == [(m.id, "halve WD"), (m.id, "lr x0.7")]
    p = llm.calls[0]["prompt"]
    assert "TRAIN WEAKER" in p and "lower LR" in p and "positively dependent" in p
    assert "effective_lr | -0.00400" in p and "island i1" in p and "over-regularised" in p
    assert "6.00" in p                                  # e-values reach the prompt


def test_big_review_prompt_uses_corrected_estimates_and_constraints(env):
    _, store = env
    _scored_store(store)                                # raw panel delta for "bad" is -0.0065
    post = NodePosterior(node_id="bad", increment=Posterior(0.0011, 0.0024, 2),
                         utility=Posterior(0.0011, 0.0024, 2),
                         axes={"offline": Posterior(-0.0021, 0.003, 2), "robust": Posterior(0.0, 0.003, 2)},
                         p_positive=0.41, p_beats_incumbent=0.1, vetoed=False)
    llm = FakeLLMClient({"review": [{"summary": "s", "findings": [], "meta_hypotheses": []}]})
    LLMAnalyst(_ctx(), llm).big_review(store, _snapshot({"bad": post}), trigger="manual")
    p = llm.calls[0]["prompt"]
    row = next(l for l in p.splitlines() if l.startswith("bad | "))
    assert "+0.0011±0.0024 | 0.41" in row                               # corrected U and P(U>0) first
    assert row.index("+0.0011") < row.index("-0.0065 (n=2)")            # raw last
    assert "raw, uncorrected, single-seed — do not cite as effects" in p
    assert "LEDGER" not in p                                            # the raw ledger is gone
    sec = p[p.index("## Hard constraints"):p.index("## Output")]
    assert "eval.dataset" in sec and "eval.split_seed" in sec and "fake_probe.py" in sec
    assert "TRAINING BUDGET is fixed" in sec and "init-compatible" in sec
    assert "fake_model.py" in sec and "Seeds:" in sec


def test_big_review_drops_unimplementable_probes(env):
    _, store = env
    _scored_store(store)
    llm = FakeLLMClient({"review": [{
        "summary": "s", "findings": [],
        "meta_hypotheses": [{"claim": "capacity helps", "mechanism_tag": "cap", "levers": {"capacity": 1},
                             "predicted_sign": 1, "falsification": "f",
                             "probes": ["+33% training steps with the same LR", "widen the MLP ratio to 5"]}],
        "probe_briefs": ["set eval.split_seed to 3", "train for longer at a lower LR",
                         "double the number of epochs", "halve weight decay", "more warmup steps"]}]})
    rep, metas = LLMAnalyst(_ctx(), llm).big_review(store, _snapshot(), trigger="manual")
    kept = [b for _, b in rep.probe_briefs]
    assert kept == ["halve weight decay", "more warmup steps", "widen the MLP ratio to 5"]
    dropped = [f for e, f in store.logged if e == "probe_dropped"]
    assert {d["brief"] for d in dropped} == {"+33% training steps with the same LR", "set eval.split_seed to 3",
                                             "train for longer at a lower LR", "double the number of epochs"}
    assert all(d["review_id"] == rep.id and d["reason"] for d in dropped)
    assert next(d for d in dropped if d["brief"].startswith("+33%"))["meta_id"] == metas[0].id


@pytest.mark.parametrize("text,bad", [
    ("+33% training steps", True), ("Train for 33% more steps", True), ("raise max_train_samples", True),
    ("extend the training schedule by 20%", True), ("set probe.datasets to a subset", True),
    ("train.max_train_flops x1.5", True), ("add 2 more local crops per sample", False),
    ("more Sinkhorn iterations", False), ("reduce steps by 20%", False), ("CLS probe-friendly pooling", False),
    ("increase drop-path to 0.2", False)])
def test_probe_violation(text, bad):
    keys = ("train.max_train_samples", "train.max_train_flops", "probe", "data.split_seed")
    assert bool(probe_violation(text, keys)) is bad


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def test_fake_client_validates_and_records():
    llm = FakeLLMClient({"judge": [{"winner": "Z", "rationale": ""}, RuntimeError("down")]})
    schema = {"type": "object", "required": ["winner"],
              "properties": {"winner": {"type": "string", "enum": ["A", "B"]}}}
    with pytest.raises(SchemaError):
        llm.complete_json("judge", system="s", prompt="p", schema=schema)
    with pytest.raises(RuntimeError):
        llm.complete_json("judge", system="s", prompt="p", schema=schema)
    with pytest.raises(KeyError):
        llm.complete_json("nobody", system="s", prompt="p", schema=schema)
    assert [c["role"] for c in llm.calls] == ["judge", "judge", "nobody"]
    # max_turns bounds scripted tool calls
    calls = []
    llm = FakeLLMClient({"r": [{"tool_calls": [("t", {})] * 5, "finish": {}}]})
    run = llm.run_agent("r", system="", prompt="", cwd=".", tools=[{"name": "t"}],
                        executor=lambda n, a: calls.append(n) or "ok",
                        finish_schema={"type": "object"}, max_turns=3, timeout_s=60)
    assert not run.ok and run.error.startswith("max_turns") and len(calls) == 3
    passthrough = AgentRun(ok=False, finish=None, turns=0, wall_s=0.0, error="x")
    assert FakeLLMClient({"r": [passthrough]}).run_agent(
        "r", system="", prompt="", cwd=".", tools=[], executor=None, finish_schema={},
        max_turns=1, timeout_s=1) is passthrough


def test_validate_subset():
    s = {"type": "object", "required": ["a"], "additionalProperties": False,
         "properties": {"a": {"type": "array", "items": {"type": "integer", "minimum": 0},
                              "minItems": 1}, "b": {"type": "number", "exclusiveMinimum": 0}}}
    assert validate({"a": [1, 2]}, s) == []
    assert validate({"a": [True]}, s)                  # bool is not an integer
    assert validate({"a": [], "b": 0, "c": 1}, s) and len(validate({"a": [], "b": 0, "c": 1}, s)) == 3


def test_sdk_client_routing_and_reask(monkeypatch):
    replies = iter(["not json", json.dumps({"x": "bad"}), json.dumps({"x": 3})])
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(llm_mod, "qwen_chat",
                        lambda system, prompt, temp, **kw: seen.append(("qwen", prompt)) or next(replies))

    @dataclass
    class Role:
        tier: str
        backend: str = ""

    @dataclass
    class Spec:
        llm_roles: dict = field(default_factory=lambda: {"judge": Role("cheap", "qwen"),
                                                         "review": Role("strong", "opus")})

    c = SdkLLMClient(Spec())
    assert c.tier("judge") == "qwen" and c.tier("review") == "opus"
    assert c.tier("researcher") == "opus" and c.tier("postmortem") == "qwen"   # defaults
    out = c.complete_json("judge", system="s", prompt="P",
                          schema={"type": "object", "properties": {"x": {"type": "integer"}}})
    assert out == {"x": 3} and len(seen) == 3 and all(t == "qwen" for t, _ in seen)
    assert "not parseable" in seen[1][1] and "violated the schema" in seen[2][1]


def test_default_backend_opus_routes_undeclared_roles_to_opus():
    import types
    roles = {"judge": types.SimpleNamespace(tier="cheap", backend="qwen")}
    q = SdkLLMClient(types.SimpleNamespace(llm_roles=roles))                        # setting absent = today
    assert q.tier("postmortem") == "qwen" and q.tier("researcher") == "opus"
    q2 = SdkLLMClient(types.SimpleNamespace(llm_roles=roles, llm_default_backend="qwen"))
    assert q2.tier("bold_knob_judge") == "qwen"
    o = SdkLLMClient(types.SimpleNamespace(llm_roles=roles, llm_default_backend="opus"))
    assert all(o.tier(r) == "opus" for r in ("postmortem", "bold_knob_judge", "mech_judge", "prescreen",
                                              "slate_judge", "review", "scout"))
    assert o.tier("judge") == "qwen"                                                # explicit declaration still wins


def test_task_yamls_default_backend():
    from kernel.model import TaskSpec
    nb, npv = (TaskSpec.load(Path(REPO) / "tasks" / f"{n}.yaml") for n in ("nanobrain", "nanopath_v2"))
    assert nb.llm_default_backend == "opus" and npv.llm_default_backend == "qwen"
    assert SdkLLMClient(nb).tier("bold_knob_judge") == "opus"
    assert SdkLLMClient(npv).tier("bold_knob_judge") == SdkLLMClient(types.SimpleNamespace(
        llm_roles=npv.llm_roles)).tier("bold_knob_judge")
    assert "qwen-serve-low-pri" not in nb.loop["slurm"]["protect_jobs"]
    assert "qwen-serve-low-pri" in npv.loop["slurm"]["protect_jobs"]


def test_no_config_import_from_agents():
    import subprocess
    code = ("import sys, loop.agents, loop.agents.llm, loop.agents.researcher, "
            "loop.agents.analyst, loop.agents.briefs; "
            "assert 'config' not in sys.modules and 'llm' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


# ---------------------------------------------------------------------------
# Integration: real SimAdapter + SqliteStore (conftest fixtures)
# ---------------------------------------------------------------------------

def test_researcher_against_sim_adapter_and_sqlite_store(ctx, store):
    from loop.adapters.sim import SimAdapter
    adapter = SimAdapter(ctx)
    snap = WorldSnapshot(task_id=ctx.task_id, n_scored=0, sigma={PANEL: ctx.cfg.panel_sd},
                         variance={}, nodes={}, incumbent_id=ROOT_NODE_ID, family_means={},
                         levers=(), hypotheses={})

    def script(cwd: str, **_):
        text = (Path(cwd) / "config.yaml").read_text()
        line = next(l for l in text.splitlines() if l.strip().startswith("lr:"))
        return {"tool_calls": [("edit_file", {"path": "config.yaml", "old": line,
                                              "new": line.split(":")[0] + ": 0.0005"}),
                               ("edit_file", {"path": "eval.py", "old": "0.0", "new": "1.0"}),
                               ("smoke", {})],
                "finish": {**FINISH, "hypothesis": {**FINISH["hypothesis"], "family": "optimization",
                                                    "predictions": {"panel": {"sign": 1, "size": 0.004}}}}}

    llm = FakeLLMClient({"researcher": script})
    req = CandidateRequest(id="r-sim", task_id=ctx.task_id, strategy=Strategy.EXPLORE,
                           operator=Operator.DRAFT, island="main", parent_ids=(ROOT_NODE_ID,),
                           brief=brief("literature", store, snap))
    res = LLMResearcher(ctx, llm, adapter).research(req, store, snap)
    assert res.status == "ok", res.error
    c = res.candidate
    assert c.changed_paths == ("config.yaml",) and c.config_diff.get("train.lr") == 0.0005
    assert c.commit and c.commit != adapter.root_commit() and c.init_compat.ok
    assert "LOCKED" in llm.calls[0]["tool_log"][1][2]          # eval.py refused
    assert adapter.true_panel(c.commit) > 0                    # planted "train weaker" truth


def test_big_review_against_sqlite_store(ctx, store):
    snap = WorldSnapshot(task_id=ctx.task_id, n_scored=0, sigma={PANEL: ctx.cfg.panel_sd},
                         variance={}, nodes={}, incumbent_id=ROOT_NODE_ID, family_means={},
                         levers=(), hypotheses={})
    llm = FakeLLMClient({"review": [{"summary": "cold start", "findings": [],
                                     "meta_hypotheses": [{"claim": "train weaker",
                                                          "mechanism_tag": "tw",
                                                          "levers": {"effective_lr": -1},
                                                          "predicted_sign": 1,
                                                          "falsification": "lr-down loses"}]}]})
    rep, metas = LLMAnalyst(ctx, llm).big_review(store, snap, trigger="manual")
    assert rep.task_id == ctx.task_id and metas[0].predictions[PANEL].size == ctx.cfg.panel_sd
    store.put_review(rep)
    for h in metas:
        store.put_hypothesis(h)
    assert store.hypotheses(kind=HypothesisKind.META)[0].source == "review"


# ---------------------------------------------------------------------------
# Fixes 13 / 5 / 7 and backend outages
# ---------------------------------------------------------------------------

def test_bash_is_confined_to_the_worktree(env, tmp_path):
    adapter, _ = env
    wt = adapter.prepare_worktree(parent_commit="c0", name="c")
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "src.py").write_text("X = 1\n")
    tools = WorktreeTools(adapter, wt, writable=("fake_model.py",), locked=(),
                          readonly_roots=(str(clone),))
    for cmd in ("cat /etc/passwd", "cat ../../x", "cd .. && ls", "cat ~/.bashrc", "ls -la /",
                "cat $HOME/x", "cat `echo /etc/passwd`", "cat /admin/data/x/ledger.jsonl",
                "cat kgraph.sqlite", "ls /data/run1", "cat nanopath-old/train.py",
                "rg foo overfitX/", "cat injected_ideas.json", "cat IDEAS.md",
                "python -c \"open('/etc/passwd')\"", "cp fake_model.py /tmp/x", "echo x >/tmp/y",
                "cat 'unbalanced"):
        assert tools("bash", {"command": cmd}).startswith("REFUSED"), cmd
    (Path(wt.path) / "data").mkdir()
    (Path(wt.path) / "escape").symlink_to("/etc")
    assert tools("bash", {"command": "cat escape/passwd"}).startswith("REFUSED")
    assert tools("bash", {"command": f"cat {clone}/src.py"}).endswith("X = 1\n")   # read-only clone
    for cmd in ("cat fake_model.py", "ls ./data", "pwd", "echo hi > /dev/null", "cat ./fake_model.py"):
        assert tools("bash", {"command": cmd}).startswith("exit="), cmd
    assert tools("bash", {"command": "pwd"}).strip().endswith(str(Path(wt.path).resolve()))
    for path in ("kgraph.sqlite", "a/ledger.jsonl", "IDEAS.md"):
        with pytest.raises(PermissionError):
            tools("read_file", {"path": path})


def test_family_must_come_from_cfg_families(env):
    from loop.spec import FamilyRule
    adapter, store = env
    ctx = _ctx()
    ctx = LoopContext(**{**ctx.__dict__, "cfg": LoopConfig(
        **{**ctx.cfg.__dict__, "families": {"optim": FamilyRule(paths=("fake_model.py",)),
                                             "data": FamilyRule()}})})
    edit = ("edit_file", {"path": "fake_model.py", "old": "LR = 1e-3", "new": "LR = 5e-4"})
    bad = {**FINISH, "hypothesis": {**FINISH["hypothesis"], "family": "made_up"}}
    llm = FakeLLMClient({"researcher": [{"tool_calls": [edit], "finish": bad}]})
    res = LLMResearcher(ctx, llm, adapter).research(_request(), store, _snapshot())
    assert res.status == "failed" and res.error.startswith("no_finish")     # finish REJECTED
    assert "not in ['data', 'optim']" in llm.calls[0]["tool_log"][-1][2]
    assert "data, optim" in llm.calls[0]["prompt"] or "['data', 'optim']" in llm.calls[0]["prompt"]
    adapter.commits = 0
    llm = FakeLLMClient({"researcher": [{"tool_calls": [edit], "finish": FINISH}]})
    assert LLMResearcher(ctx, llm, adapter).research(_request(id="r2"), store, _snapshot()).status == "ok"


def test_backend_outage_is_typed_not_failed(env, monkeypatch):
    adapter, store = env
    down = AgentRun(ok=False, finish=None, turns=0, wall_s=0.0,
                    error="backend_unavailable: URLError")
    res = LLMResearcher(_ctx(), FakeLLMClient({"researcher": [down]}), adapter).research(
        _request(), store, _snapshot())
    assert res.status == "backend_unavailable" and adapter.reaped
    # qwen down -> BackendUnavailable (never SchemaError / a void)
    def boom(*a, **k): raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setenv("OVFX_LOOP_QWEN_URL", "http://127.0.0.1:9")
    def opus_never(*a, **k): raise AssertionError("a qwen role never falls back to Opus")
    monkeypatch.setattr(llm_mod, "opus_chat", opus_never)
    c = SdkLLMClient(types.SimpleNamespace(llm_roles={}))
    with pytest.raises(BackendUnavailable, match="qwen: URLError"):      # the job is skipped, asked again later
        c.complete_json("cheap", system="s", prompt="p", schema={"type": "object"})
    assert llm_mod.qwen_url() == "http://127.0.0.1:9/v1"
    monkeypatch.setenv("OVFX_LOOP_QWEN_URL", "http://h:1/v1/")
    assert llm_mod.qwen_url() == "http://h:1/v1"


def test_qwen_chat_parses_think_and_json(monkeypatch):
    monkeypatch.setenv("OVFX_LOOP_QWEN_URL", "http://h:1")
    body = {"choices": [{"message": {"content": '<think>hm {x}</think>```json\n{"a": 1}\n```'}}]}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(body).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: Resp())
    assert llm_mod.parse_json(llm_mod.qwen_chat("s", "p", 0.1)) == {"a": 1}


def test_qwen_down_skips_the_job_never_opus_then_switches_back(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(llm_mod.time, "monotonic", lambda: clock[0])
    qwen_up, calls = [False], []

    def qwen(system, prompt, temp, **kw):
        calls.append("qwen")
        if not qwen_up[0]:
            raise BackendUnavailable("qwen: URLError: name not resolved")
        return json.dumps({"x": 1})
    monkeypatch.setattr(llm_mod, "qwen_chat", qwen)
    monkeypatch.setattr(llm_mod, "opus_chat", lambda *a, **k: calls.append("opus") or json.dumps({"x": 2}))
    c = SdkLLMClient(types.SimpleNamespace(llm_roles={}), qwen_cooldown_s=600)
    ask = lambda: c.complete_json("cheap", system="s", prompt="p", schema={"type": "object"})   # noqa: E731
    with pytest.raises(BackendUnavailable, match="name not resolved"):
        ask()
    with pytest.raises(BackendUnavailable, match="cooldown"):
        ask()                                                              # cooldown: qwen not even asked
    assert calls == ["qwen"]
    qwen_up[0], clock[0] = True, clock[0] + 601
    assert ask() == {"x": 1} and calls == ["qwen", "qwen"] and "opus" not in calls


def test_circuit_breaker_turns_sdk_failures_into_an_outage(monkeypatch):
    """Live 2026-10-03: 'Claude Code returned an error result: success' failed ~3200 calls and paused nothing. That
    text is an outage at once; any other failure signature repeated BREAKER_N times in the window is one too; a
    success restarts the count."""
    llm_mod.take_outage()
    llm_mod.note_success(now=1.0)
    assert llm_mod.note_failure("Exception: Claude Code returned an error result: success", now=10.0)
    assert llm_mod.take_outage() == {"kind": "sdk", "resets_at": None}
    for i in range(llm_mod.BREAKER_N - 1):
        assert not llm_mod.note_failure(f"RuntimeError: stream closed after {i} bytes", now=20.0 + i)
    llm_mod.note_success(now=30.0)                                       # a success in between: count restarts
    assert not llm_mod.note_failure("RuntimeError: stream closed after 7 bytes", now=31.0)   # 1 of N again
    assert not llm_mod.note_failure("ValueError: a different failure", now=32.0)          # another signature
    assert llm_mod.take_outage() is None
    hits = [llm_mod.note_failure("RuntimeError: stream closed after 9 bytes", now=40.0 + i)
            for i in range(llm_mod.BREAKER_N)]
    assert hits[-1] and not any(hits[:-1]) and llm_mod.take_outage()["kind"] == "sdk"
    assert not llm_mod.note_failure("other", now=41.0 + llm_mod.BREAKER_WINDOW_S)
    # an opus call that raises feeds it, one that works resets it
    monkeypatch.setattr(llm_mod, "opus_chat", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("Claude Code returned an error result: success")))
    c = SdkLLMClient(types.SimpleNamespace(llm_roles={}))
    with pytest.raises(BackendUnavailable):
        c.complete_json("review", system="s", prompt="p", schema={"type": "object"})
    assert llm_mod.take_outage()["kind"] == "sdk"
    mark = llm_mod.last_success()
    monkeypatch.setattr(llm_mod, "opus_chat", lambda *a, **k: "{}")
    assert c.complete_json("review", system="s", prompt="p", schema={"type": "object"}) == {}
    assert llm_mod.last_success() > mark or llm_mod.last_success() >= 30.0


def test_unknown_llm_role_is_a_load_error(tmp_path):
    from loop.spec import load_context
    src = (REPO / "tasks" / "nanopath_v2.yaml").read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(src.replace("  review: {tier: strong, backend: opus}\n",
                               "  review: {tier: strong, backend: opus}\n  ideate: {tier: cheap}\n"))
    assert "ideate: {tier: cheap}" in bad.read_text()
    with pytest.raises(ValueError, match="llm_roles.ideate: unknown role"):
        load_context(bad, state_dir=tmp_path, require_measured=False)
    load_context(REPO / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path, require_measured=False)


def test_agent_session_wall_covers_the_gate_deadline_and_a_smoke(monkeypatch):
    seen = {}
    fake_llm = types.SimpleNamespace(OPUS_TIMEOUT=600.0)

    def session(prompt, **kw):
        seen["wall"] = fake_llm.OPUS_TIMEOUT
        return {"ok": True, "turns": 1}
    fake_llm.sdk_session = session
    fake_agent_llm = types.SimpleNamespace(sdk_tool_server=lambda defs, gate, name: ({}, [], []))
    monkeypatch.setitem(sys.modules, "llm", fake_llm)
    monkeypatch.setitem(sys.modules, "agent", types.SimpleNamespace(llm=fake_agent_llm))
    monkeypatch.setitem(sys.modules, "agent.llm", fake_agent_llm)
    run = SdkLLMClient(types.SimpleNamespace(llm_roles={})).run_agent(
        "researcher", system="s", prompt="p", cwd=".", tools=[], executor=lambda n, a: "",
        finish_schema={"type": "object"}, max_turns=5, timeout_s=1200.0)
    assert seen["wall"] == 1200.0 + llm_mod.AGENT_TOOL_GRACE_S and run.error.startswith("no_finish")


def _fake_urlopen(monkeypatch, content, seen):
    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout: seen.append(json.loads(req.data)) or Resp())


def test_qwen_chat_disables_thinking_and_strips_think(monkeypatch):
    monkeypatch.setenv("OVFX_LOOP_QWEN_URL", "http://h:1")
    seen: list = []
    _fake_urlopen(monkeypatch, '<think>x</think>{"a": 1}', seen)
    assert llm_mod.qwen_chat("s", "p", 0.1) == '{"a": 1}'
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_qwen_empty_content_is_backend_unavailable(monkeypatch):
    monkeypatch.setenv("OVFX_LOOP_QWEN_URL", "http://h:1")
    _fake_urlopen(monkeypatch, "", [])
    with pytest.raises(BackendUnavailable):
        llm_mod.qwen_chat("s", "p", 0.1)


# ---------------------------------------------------------------------------
# check / set_config / finish bounce / context pack
# ---------------------------------------------------------------------------

def _locked_ctx() -> LoopContext:
    ctx = _ctx()
    surface = ctx.spec.surface.model_copy(update={"locked": ("fake_probe.py", "fake_model.py::score_fn")})
    return dataclasses.replace(ctx, spec=ctx.spec.model_copy(update={"surface": surface}))


def _tools(adapter, store, ctx, name="k"):
    r = LLMResearcher(ctx, FakeLLMClient({}), adapter)
    wt = adapter.prepare_worktree(parent_commit="c0", name=name)
    req = _request()
    tools = WorktreeTools(adapter, wt, writable=tuple(ctx.spec.surface.writable),
                          locked=tuple(ctx.spec.surface.locked), config_rel=r.config_rel(),
                          protected=tuple(ctx.spec.surface.protected_keys),
                          checker=lambda **kw: r.check(req, store, wt, **kw))
    return tools, wt


def test_check_rejects_locked_function_and_two_families_ok_for_config(env):
    adapter, store = env
    ctx = _locked_ctx()
    adapter.locks = ("fake_model.py::score_fn",)
    tools, _ = _tools(adapter, store, ctx, "a")
    assert tools("check", {}).startswith("OK (no diff yet)")   # c4i: a mechanism is checkable before any edit
    assert tools("edit_file", {"path": "fake_model.py", "old": "return x", "new": "return 2 * x"}) \
        .startswith("edited")                              # the file is writable; the function is not
    out = tools("check", {"family": "optim"})
    assert out.startswith("REJECT surface_locked") and "fake_model.py::score_fn" in out and "\n" not in out

    tools, _ = _tools(adapter, store, ctx, "b")
    assert tools("set_config", {"key": "train.lr", "value": "5e-4"}).startswith("set train.lr")
    assert tools("set_config", {"key": "aug.p", "value": 0.7}).startswith("set aug.p")
    out = tools("check", {"family": "optim"})
    assert out.startswith("REJECT multi_family") and "augment" in out and "optim" in out

    tools, _ = _tools(adapter, store, ctx, "c")
    tools("set_config", {"key": "train.lr", "value": 0.0005})
    assert tools("check", {}).startswith("OK") and "family optim" in tools("check", {"family": "optim"})
    assert tools("check", {"family": "augment"}).startswith("REJECT family_mismatch")


def test_finish_bounces_on_a_failing_check_then_passes_after_the_fix(env):
    adapter, store = env
    adapter.locks = ("fake_model.py::score_fn",)
    edit_locked = ("edit_file", {"path": "fake_model.py", "old": "return x", "new": "return 2 * x"})
    undo = ("edit_file", {"path": "fake_model.py", "old": "return 2 * x", "new": "return x"})
    lr = ("edit_file", {"path": "fake_model.py", "old": "LR = 1e-3", "new": "LR = 5e-4"})
    llm = FakeLLMClient({"researcher": [{"tool_calls": [edit_locked, ("finish", FINISH), undo, lr],
                                         "finish": FINISH}]})
    res = LLMResearcher(_locked_ctx(), llm, adapter).research(_request(), store, _snapshot())
    log = llm.calls[0]["tool_log"]
    assert log[1][0] == "finish" and log[1][2].startswith("finish BOUNCED: REJECT surface_locked")
    assert "bounce 1/2" in log[1][2]
    assert res.status == "ok", res.error                   # fixed, and the second finish was accepted


def test_set_config_round_trips_and_refuses_protected_keys(env):
    adapter, store = env
    tools, wt = _tools(adapter, store, _ctx(), "s")
    cfg = Path(wt.path) / "fake_config.yaml"
    assert tools("set_config", {"key": "train.lr", "value": "1e-4"}) == \
        "set train.lr: 0.001 -> 0.0001. Next: check, then smoke."
    assert cfg.read_text() == CONFIG_TEXT.replace("lr: 0.001   # peak", "lr: 0.0001   # peak")
    tools("set_config", {"key": "aug.scale", "value": "[0.3, 1.0]"})
    assert yaml.safe_load(cfg.read_text())["aug"] == {"p": 0.5, "scale": [0.3, 1.0]}
    assert tools("set_config", {"key": "eval.dataset", "value": "y"}).startswith("REFUSED: eval.dataset is protected")
    out = tools("set_config", {"key": "train.lrr", "value": 1})
    assert out.startswith("ERROR: unknown key 'train.lrr'") and "train.lr" in out
    assert tools("set_config", {"key": "train", "value": 1}).startswith("ERROR: unknown key 'train'")
    from loop.agents.researcher import set_yaml_value
    with pytest.raises(ValueError, match="is a block"):
        set_yaml_value("a:\n  xs:\n    - 1\n    - 2\n  y: 3\n", "a.xs", [1])
    assert set_yaml_value("a:\n  xs:\n    - 1\n  y: 3  # c\nb:\n  y: 4\n", "b.y", 5).endswith("b:\n  y: 5\n")
    assert tools("set_config", {"key": "train.steps", "value": "many"}).startswith("ERROR: train.steps is a number")
    assert tools("nope", {}).startswith("ERROR: unknown tool") and \
        tools("read_file", {"pth": "x"}).startswith("ERROR: bad arguments for read_file")
    assert "0.0001" in cfg.read_text() and "steps: 10" in cfg.read_text()


def test_context_pack_stays_under_budget_and_carries_the_lessons(env, monkeypatch):
    from loop.agents import briefs as B
    _, store = env
    for i in range(600):
        store.put_node(_node(f"n{i:04d}", tag=f"mechanism_number_{i}_with_a_long_name",
                             levers=LeverTags({"effective_lr": -1}), created_ts=float(i)))
    store._hyps.update({f"m{i}": HypothesisRecord(
        id=f"m{i}", task_id=TID, kind=HypothesisKind.META, claim="x" * 400, mechanism_tag="t",
        family="cross", predictions={PANEL: AxisPrediction(1, 0.004)}, falsification="f") for i in range(40)})
    config = yaml.safe_load(CONFIG_TEXT)
    monkeypatch.setitem(sys.modules, "loop.quality", types.SimpleNamespace(
        ideation_lessons=lambda st, sn: "optim works; your predictions were 2.3x too optimistic\n" + "y" * 20000))
    from loop.adapters import flatten
    pack = B.context_pack(_ctx(), store, _snapshot(), config=flatten(config))
    assert B.estimate_tokens(pack) <= B.PACK_TOKENS
    for s in ("Editable files", "LOCKED files", "Protected config keys", "### Families", "train.lr=0.001",
              "### Levers", "Carried lesson", "What we've learned so far", "2.3x too optimistic",
              "Open meta-hypotheses", "n0599 | optim/", "older trials omitted", '"falsification"'):
        assert s in pack, s
    assert "n0000 |" not in pack                                   # newest first, oldest trimmed
    monkeypatch.delitem(sys.modules, "loop.quality")
    monkeypatch.setattr(B, "optional_block", lambda *a: "")
    assert "(nothing yet)" in B.context_pack(_ctx(), store, _snapshot())


def test_big_review_includes_quality_table_and_lessons_when_present(env, monkeypatch):
    _, store = env
    _scored_store(store)
    monkeypatch.setitem(sys.modules, "loop.quality", types.SimpleNamespace(
        quality_table=lambda st, sn: "QTABLE brief kinds x hit rate",
        ideation_lessons=lambda st, sn: "LESSONS regularisers lose"))
    out = {"summary": "s", "findings": [], "meta_hypotheses": []}
    llm = FakeLLMClient({"review": [out, out]})
    LLMAnalyst(_ctx(), llm).big_review(store, _snapshot(), trigger="manual")
    assert "QTABLE" in llm.calls[0]["prompt"] and "LESSONS" in llm.calls[0]["prompt"]
    monkeypatch.setitem(sys.modules, "loop.quality", types.SimpleNamespace())   # broken module: no crash
    LLMAnalyst(_ctx(), llm).big_review(store, _snapshot(), trigger="manual")
    assert "(not available)" in llm.calls[1]["prompt"]


def test_agent_session_loads_tools_up_front(monkeypatch):
    monkeypatch.delenv("ENABLE_TOOL_SEARCH", raising=False)
    seen = {}
    fake_llm = types.SimpleNamespace(OPUS_TIMEOUT=0, sdk_session=lambda prompt, **kw: seen.update(
        env=__import__("os").environ.get("ENABLE_TOOL_SEARCH")) or {"ok": True, "turns": 1})
    fake_agent_llm = types.SimpleNamespace(sdk_tool_server=lambda defs, gate, name: ({}, [], []))
    monkeypatch.setitem(sys.modules, "llm", fake_llm)
    monkeypatch.setitem(sys.modules, "agent", types.SimpleNamespace(llm=fake_agent_llm))
    monkeypatch.setitem(sys.modules, "agent.llm", fake_agent_llm)
    SdkLLMClient(types.SimpleNamespace(llm_roles={})).run_agent(
        "researcher", system="s", prompt="p", cwd=".", tools=[], executor=lambda n, a: "",
        finish_schema={"type": "object"}, max_turns=5, timeout_s=60.0)
    assert seen["env"] == "false"


# ---------------------------------------------------------------------------
# Moonshot: its own caps, web tools, sizing and a brief that states the constraints
# ---------------------------------------------------------------------------

def test_moonshot_caps_web_and_sizing_override(env):
    adapter, store = env
    script = {"tool_calls": [("edit_file", {"path": "fake_model.py", "old": "LR = 1e-3", "new": "LR = 5e-4"}),
                             ("smoke", {})], "finish": FINISH}
    llm = FakeLLMClient({"researcher": [script, script]})
    r = LLMResearcher(_ctx(), llm, adapter)
    exploit = dataclasses.replace(_request(brief_kind="frontier"), strategy=Strategy.EXPLOIT)
    assert r.research(exploit, store, _snapshot()).status == "ok"
    r.research(dataclasses.replace(_request(id="req2"), strategy=Strategy.MOONSHOT), store, _snapshot())
    normal, moon = llm.calls
    assert (normal["max_turns"], normal["timeout_s"], normal["web"]) == (48, 20 * 60, False)
    assert (moon["max_turns"], moon["timeout_s"], moon["web"]) == (60, 45 * 60, True)
    assert "60 tool calls, 45.0 minutes" in moon["prompt"]
    assert "IGNORE its advice to shrink" in moon["prompt"] and "IGNORE" not in normal["prompt"]
    assert "scale\nyour predicted sizes" in normal["prompt"]


def test_web_research_flag_prompt_and_counts(env):
    """Literature / explore / moonshot sessions get WebSearch+WebFetch and the 'research first, cite' text;
    an exploit (frontier) session and a DEBUG fix do not; the web calls are counted onto the result."""
    from loop.agents.researcher import WEB_TEXT, wants_web
    adapter, store = env
    edit = [("edit_file", {"path": "fake_model.py", "old": "LR = 1e-3", "new": "LR = 5e-4"}), ("smoke", {})]
    web = {"tool_calls": [("WebSearch", {"query": "cosine LR SSL"}), ("WebSearch", {"query": "warmup"}),
                          ("WebFetch", {"url": "https://arxiv.org/abs/2304.07193"}), *edit], "finish": FINISH}
    llm = FakeLLMClient({"researcher": [web, {"tool_calls": edit, "finish": FINISH}]})
    r = LLMResearcher(_ctx(), llm, adapter)
    res = r.research(_request(brief_kind="literature"), store, _snapshot())
    assert res.status == "ok" and (res.web, res.web_searches, res.web_fetches) == (True, 2, 1)
    assert llm.calls[0]["web"] is True and "Stage 1 -- RESEARCH" in llm.calls[0]["prompt"]     # literature: research
    assert {"notes", "note"} <= set(llm.calls[0]["tools"])
    assert "2-5 searches" in WEB_TEXT and "Cite your sources" in WEB_TEXT
    exploit = dataclasses.replace(_request(id="req2", brief_kind="frontier"), strategy=Strategy.EXPLOIT)
    res = r.research(exploit, store, _snapshot())
    assert (res.web, res.web_searches) == (False, 0) and llm.calls[1]["web"] is False
    assert "Background research (web)" not in llm.calls[1]["prompt"]
    assert wants_web(dataclasses.replace(_request(), strategy=Strategy.MOONSHOT, brief_kind="moonshot"))
    assert not wants_web(_request(op=Operator.DEBUG, brief_kind="failure"))


def test_moonshot_brief_states_the_hard_constraints(env):
    _, store = env
    ctx = _ctx()
    txt = brief("moonshot", store, _snapshot(), ctx=ctx, probe="[moonshot] PCGrad across the losses")
    assert "Protected config keys" in txt and "LOCKED files" in txt and "PCGrad across the losses" in txt
    for must in ("SAME samples and FLOPs", "zero", "init-compat", "5 h", "Init swap", "init_illegal",
                 "over_flop_cap", "BOLD STRUCTURAL brief", "pretext", "bold_knob_only"):
        assert must in txt, must
    assert "effective_lr" not in txt and "Lever trends" not in txt     # v3-c8b: retired into the bold brief, no levers


def test_init_swap_reference_code_is_readable_by_researchers(tmp_path):
    """init_compat.references resolve against the task spec (a researcher's read-only roots, named in the brief)."""
    from loop.spec import load_context
    src = Path(__file__).resolve().parents[2] / "tasks" / "nanopath_v2.yaml"
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "dinov3_port.patch").write_text("# reference port\n")
    spec = tmp_path / "nanopath_v2.yaml"
    spec.write_text(src.read_text().replace("    references: []", "    references: [references/dinov3_port.patch]"))
    c = load_context(spec, state_dir=tmp_path / "s", require_measured=False)
    refs = briefs.references(c)
    assert refs and all(Path(r).is_file() for r in refs) and any("dinov3" in r for r in refs)
    assert refs[0] in briefs.hard_constraints(c)


def test_sdk_run_agent_allows_web_tools_only_when_asked(monkeypatch):
    seen: list = []
    fake_llm = types.SimpleNamespace(OPUS_TIMEOUT=0, sdk_session=lambda prompt, **kw: seen.append(
        kw["allowed_tools"]) or {"ok": True, "turns": 1})
    fake_agent_llm = types.SimpleNamespace(sdk_tool_server=lambda defs, gate, name: ({}, ["mcp__t"], []))
    monkeypatch.setitem(sys.modules, "llm", fake_llm)
    monkeypatch.setitem(sys.modules, "agent", types.SimpleNamespace(llm=fake_agent_llm))
    monkeypatch.setitem(sys.modules, "agent.llm", fake_agent_llm)
    for web in (False, True):
        SdkLLMClient(types.SimpleNamespace(llm_roles={})).run_agent(
            "researcher", system="s", prompt="p", cwd=".", tools=[], executor=lambda n, a: "",
            finish_schema={"type": "object"}, max_turns=5, timeout_s=60.0, web=web)
    assert seen == [["mcp__t"], ["mcp__t", "WebSearch", "WebFetch"]]


# ---------------------------------------------------------------------------
# Operator focus + idea-bank briefs
# ---------------------------------------------------------------------------

FOCUS = "Mess around with JEPA: predictor width/depth, masking curricula."


def _focus_ctx() -> LoopContext:
    ctx = _ctx()
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, operator_focus=FOCUS))


def test_operator_focus_leads_ideation_briefs_but_not_the_exploit_brief(env):
    from loop.agents.briefs import FOCUS_HEADING
    _, store = env
    ctx, snap = _focus_ctx(), _snapshot()
    idea = {"id": "j1", "title": "multi-block predictor", "family": "objective", "moonshot": False,
            "jepa": True, "idea": "predict 4 target blocks", "why_it_might_work": "harder latent task",
            "v1_fate": "", "why_v1_verdict_may_not_transfer": "v1 scored a different suite", "source": "v1"}
    for kind, kw in (("literature", {}), ("failure", {}), ("probe", {"probe": "p"}), ("moonshot", {}),
                     ("bank", {"bank": idea}), ("bank", {"bank": idea, "moonshot": True})):
        txt = brief(kind, store, snap, ctx=ctx, **kw)
        assert txt.startswith(f"{FOCUS_HEADING}\n{FOCUS}\n"), kind
    store.put_node(_node("w", status=NodeStatus.CONFIRMED))
    assert FOCUS not in brief("frontier", store, snap, ctx=ctx)
    assert FOCUS_HEADING not in brief("literature", store, snap, ctx=_ctx())      # unset: nothing
    txt = brief("bank", store, snap, ctx=ctx, bank=idea)
    for must in ("IDEA BANK idea `j1`", "predict 4 target blocks", "harder latent task",
                 "v1 scored a different suite", "clean slate", "EXPLORE brief"):
        assert must in txt, must
    assert "What happened before" not in txt                                      # an empty fate is omitted
    moon = brief("bank", store, snap, ctx=ctx, bank=idea, moonshot=True)
    assert "BOLD STRUCTURAL brief" in moon and "predict 4 target blocks" in moon and "Protected config keys" in moon


def test_big_and_first_principles_reviews_carry_the_operator_focus(env):
    from loop.agents.analyst import FP_TRIGGER
    from loop.agents.briefs import FOCUS_HEADING
    from tests.loop.test_selftune import FP_OUT
    _, store = env
    llm = FakeLLMClient({"review": [{"summary": "s", "findings": [], "meta_hypotheses": []}, FP_OUT]})
    LLMAnalyst(_focus_ctx(), llm).big_review(store, _snapshot(), trigger="manual")
    LLMAnalyst(_focus_ctx(), llm).big_review(store, _snapshot(), trigger=FP_TRIGGER)
    for call in llm.calls:
        assert f"{FOCUS_HEADING}\n{FOCUS}" in call["prompt"] and "Bias your probe briefs" in call["prompt"]
    llm = FakeLLMClient({"review": [{"summary": "s", "findings": [], "meta_hypotheses": []}]})
    LLMAnalyst(_ctx(), llm).big_review(store, _snapshot(), trigger="manual")
    assert FOCUS_HEADING not in llm.calls[0]["prompt"]


def test_a_smoke_that_never_got_a_slot_is_backend_unavailable_and_requeued_once_in_thread(env):
    """Live n0521-n0526: rc 124 (queued past SMOKE_WAIT_S) was rejected smoke_failed (+0 reward). It is no verdict:
    the researcher queues the smoke again once, then returns backend_unavailable with the candidate kept."""
    adapter, store = env
    script = lambda **_: {"tool_calls": [("edit_file", {"path": "fake_model.py",
                                                       "old": "LR = 1e-3", "new": "LR = 2e-3"})],
                          "finish": FINISH}
    results = [SmokeResult(False, "smoke_unavailable: queue timeout", "", unavailable=True)] * 2
    adapter.smoke = lambda wt: (setattr(adapter, "smokes", adapter.smokes + 1), results.pop(0))[1]
    r = LLMResearcher(_ctx(), FakeLLMClient({"researcher": script}), adapter).research(
        _request(), store, _snapshot())
    assert r.status == "backend_unavailable" and r.error.startswith("smoke_unavailable")
    assert r.candidate is not None and r.candidate.diff_text and adapter.commits == 0 and adapter.smokes == 2
    results[:] = [SmokeResult(False, "smoke_unavailable: queue timeout", "", unavailable=True), SmokeResult(True)]
    r = LLMResearcher(_ctx(), FakeLLMClient({"researcher": script}), adapter).research(
        _request(id="req2"), store, _snapshot())
    assert r.status == "ok" and adapter.commits == 1        # the second queueing got a slot


def test_a_sessions_own_unavailable_smoke_counts_as_the_first_queueing(env):
    """Live 2026-10-02: the session's smoke, then 2 more after finish, held each researcher 3 h (3 x SMOKE_WAIT_S)
    behind a full association. The session's own unavailable smoke is the first queueing: one more, then out."""
    adapter, store = env
    script = lambda **_: {"tool_calls": [("edit_file", {"path": "fake_model.py", "old": "LR = 1e-3", "new": "LR = 2e-3"}),
                                         ("smoke", {})],
                          "finish": FINISH}
    results = [SmokeResult(False, "smoke_unavailable: queue timeout", "", unavailable=True)] * 3
    adapter.smoke = lambda wt: (setattr(adapter, "smokes", adapter.smokes + 1), results.pop(0))[1]
    r = LLMResearcher(_ctx(), FakeLLMClient({"researcher": script}), adapter).research(
        _request(), store, _snapshot())
    assert r.status == "backend_unavailable" and adapter.smokes == 2


def test_a_usage_limit_is_classified_quota_with_its_reset_and_never_retried(monkeypatch):
    """Live 2026-10-03 22:00Z: the account hit its weekly limit; the SDK streamed an AssistantMessage error=rate_limit
    ("You've hit your weekly limit · resets 12am (UTC)") then raised only "error result: success", which was retried
    3x per call as kind=other. It is kind='quota' with resets_at, returned at once (no backoff sleep)."""
    import claude_agent_sdk as cas
    import llm as root_llm
    calls, slept = [], []

    def query(prompt, options):
        calls.append(1)

        async def gen():
            yield cas.RateLimitEvent(cas.RateLimitInfo(status="rejected", resets_at=1791072000), "u", "s")
            yield cas.AssistantMessage([cas.TextBlock("You've hit your weekly limit · resets 12am (UTC)")], "m",
                                       error="rate_limit")
            raise Exception("Claude Code returned an error result: success")
        return gen()
    monkeypatch.setattr(cas, "query", query)
    monkeypatch.setattr(root_llm.time, "sleep", slept.append)
    res = root_llm.sdk_session("hi", max_turns=2)
    assert res["error_kind"] == "quota" and res["resets_at"] == 1791072000 and calls == [1] and not slept

    def query_auth(prompt, options):
        async def gen():
            yield cas.AssistantMessage([cas.TextBlock("Not logged in · Please run /login")], "m",
                                       error="authentication_failed")
            raise Exception("Claude Code returned an error result: success")
        return gen()
    monkeypatch.setattr(cas, "query", query_auth)
    assert root_llm.sdk_session("hi", max_turns=2)["error_kind"] == "auth"
    now = 1791066600.0                                              # 2026-10-03 22:30Z
    assert root_llm._resets_from_text("resets 12am (UTC)", now) == 1791072000.0
    assert root_llm._resets_from_text("resets 11:30pm (UTC)", now) == 1791070200.0
    assert root_llm._resets_from_text("You've hit your weekly limit · resets Oct 5, 7am (UTC)", now) == 1791183600.0


def test_an_account_refusal_in_a_session_or_a_chat_is_noted_for_the_driver_pause(monkeypatch):
    LA = llm_mod
    LA.take_outage()
    fake_llm = types.SimpleNamespace(OPUS_TIMEOUT=0, sdk_session=lambda prompt, **kw: {
        "ok": False, "error_kind": "quota", "resets_at": 1791072000, "error": "Exception: success"})
    fake_agent_llm = types.SimpleNamespace(sdk_tool_server=lambda defs, gate, name: ({}, [], []))
    monkeypatch.setitem(sys.modules, "llm", fake_llm)
    monkeypatch.setitem(sys.modules, "agent", types.SimpleNamespace(llm=fake_agent_llm))
    monkeypatch.setitem(sys.modules, "agent.llm", fake_agent_llm)
    run = SdkLLMClient(types.SimpleNamespace(llm_roles={})).run_agent(
        "researcher", system="s", prompt="p", cwd=".", tools=[], executor=lambda n, a: "",
        finish_schema={"type": "object"}, max_turns=5, timeout_s=60.0)
    assert run.error.startswith("backend_unavailable") and LA.take_outage() == {"kind": "quota", "resets_at": 1791072000}
    assert LA.take_outage() is None

    class Refused(RuntimeError):
        kind, resets_at = "auth", None
    monkeypatch.setattr(LA, "opus_chat", lambda *a: (_ for _ in ()).throw(Refused("Not logged in")))
    with pytest.raises(BackendUnavailable):
        SdkLLMClient(types.SimpleNamespace(llm_roles={}))._ask("review", "s", "p", 0.2)
    assert LA.take_outage() == {"kind": "auth", "resets_at": None}


def test_check_probes_init_compat_early_once_per_diff(ctx, store):
    """2026-10-06: 75 init_incompatible rejections (49 moonshots, up to 35 a day), each a whole session lost at intake.
    The researcher's check runs the intake's init-compat check on the current worktree when the change needs it:
    REJECT with the reason (once per diff), OK once it loads."""
    from types import SimpleNamespace
    from loop.agents.researcher import LLMResearcher
    from loop.contracts import CandidateRequest, InitCompat, Operator, ROOT_NODE_ID, Strategy
    store.ensure_root("c0")
    diff = {"text": "diff --git a/model.py b/model.py\n--- a/model.py\n+++ b/model.py\n@@ -1 +1,2 @@\n x = 1\n+def head(): pass\n"}
    calls, ok = [], {"v": False}
    adapter = SimpleNamespace(diff=lambda wt: (diff["text"], ["model.py"], {}),
                              init_compat=lambda wt, parent_commit: calls.append(parent_commit) or InitCompat(
                                  ok["v"], "" if ok["v"] else "checkpoint_keys {'missing': ['head.weight']}"))
    res = LLMResearcher(ctx, None, adapter)
    req = CandidateRequest(id="r", task_id=store.task_id, strategy=Strategy.MOONSHOT, operator=Operator.DRAFT,
                           island="i", parent_ids=(ROOT_NODE_ID,), brief="")
    wt = SimpleNamespace(path="/tmp", branch="b", base_commit="c0")
    got = res.check(req, store, wt, family="architecture", claim="a new head", mechanism_tag="new_head")
    assert got.startswith("REJECT init_incompatible") and "head.weight" in got and calls == ["c0"]
    res.check(req, store, wt, family="architecture", claim="a new head", mechanism_tag="new_head")
    assert calls == ["c0"]                                                   # same diff: probed once
    diff["text"] += "+y = 2\n"
    ok["v"] = True
    assert "init-compat probe OK" in res.check(req, store, wt, family="architecture", claim="a new head",
                                               mechanism_tag="new_head") and len(calls) == 2
    assert store.events("init_compat_probe")[0]["reason"].startswith("checkpoint_keys")


def test_the_research_stage_gates_finish_on_papers_actually_read(ctx, tmp_path):
    """Moonshot / literature / scout ideation (loop.agents.notes): finish bounces ONCE unless the session searched
    >= min_searches times, READ (WebFetch) and noted >= min_new_papers papers, built on >= min_papers distinct papers
    (new + cited earlier notes) and cites a note id; a note needs its URL fetched in this session; notes persist."""
    from loop.agents import notes as NT
    from loop.agents.researcher import WorktreeTools
    r = dataclasses.replace(ctx.cfg.researcher, notes_path=str(tmp_path / "rn.jsonl"), min_papers=3,
                            min_new_papers=2, min_searches=2)
    c = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, researcher=r))
    st = NT.ResearchStage(c, "req-1")
    tools = WorktreeTools(None, types.SimpleNamespace(path=str(tmp_path)), writable=(), locked=())
    tools.research = st
    fields = {f: f"a long enough {f} for the note, with numbers +1.2%" for f in NT.FIELDS}
    assert tools("note", {"url": "https://arxiv.org/abs/2401.00001", "title": "t", **fields}).startswith("ERROR")
    tools.on_web("WebSearch", {"query": "q1"})
    tools.on_web("WebFetch", {"url": "https://arxiv.org/html/2401.00001v2"})       # same paper, other URL form
    n1 = tools("note", {"url": "https://arxiv.org/abs/2401.00001", "title": "Paper one", **fields}).split()[1]
    fin = {"hypothesis": {"claim": "x", "builds_on": f"{n1}: its loss"}}
    gap = tools.check_finish(fin)
    assert "RESEARCH STAGE incomplete" in gap and "1/2 WebSearch" in gap and "1/2 new papers" in gap
    assert tools.check_finish(fin) == ""                                              # bounces once only
    st2 = NT.ResearchStage(c, "req-2")                                                # a later session reuses it
    assert n1 in st2.t_notes("paper one loss")
    for i, u in enumerate(("https://example.org/physics-paper/", "https://arxiv.org/abs/2502.01234")):
        st2.on_web("WebSearch", {"query": f"q{i}"})
        st2.on_web("WebFetch", {"url": u})
        st2.t_note(url=u, title=f"p{i}", **fields)
    assert len(st2.new) == 2 and st2.missing({"hypothesis": {"builds_on": "none"}}).startswith("RESEARCH")
    st3 = NT.ResearchStage(c, "req-3")
    st3.searches, st3.new = ["a", "b"], list(st2.new)
    assert st3.missing({"hypothesis": {"builds_on": f"{n1} and {st2.new[0]}"}}) == ""   # 3 distinct papers, cited
    assert NT.key("https://www.Example.org/a/?x=1") == "example.org/a" and NT.distant("r", 0.0) is False


def test_v3_session_gets_its_brief_and_must_declare_a_measurable_mechanism(env):
    """A loop.core.ideation request (id prefix req-v3-): no world-model snapshot, the brief is the context, the
    first-principles procedure lists the diagnostics quantities, and the finish's mechanism_prediction + knobs ride
    on the result's plan."""
    adapter, store = env
    edit = [("edit_file", {"path": "fake_model.py", "old": "LR = 1e-3", "new": "LR = 5e-4"}), ("smoke", {})]
    mp = {"diagnostic": "lr_end_frac", "direction": -1, "axes": {"offline": 1},
          "budget": [{"axis": "cls", "row": "D-cls", "gain": 0.01}]}     # c4i: the size is a per-axis budget
    fin = {**FINISH, "hypothesis": {**FINISH["hypothesis"], "mechanistic_model": "m", "bottleneck": "b",
                                    "prior_attempts": "none", "mechanism_prediction": mp},
           "knobs": [{"key": "train.lr", "lo": 2.5e-4, "hi": 1e-3, "scale": "log"}]}
    llm = FakeLLMClient({"researcher": [{"tool_calls": edit, "finish": fin}]})
    req = dataclasses.replace(_request(id="req-v3-00001", brief="REPAIR classification: HEADROOM TABLE"),
                              strategy=Strategy.EXPLOIT, brief_kind="repair")
    ctx = _ctx()
    ctx.cfg = dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(ctx.cfg.ideation, portfolio_repair=False))   # c9 C3: a repair proposes a portfolio by default; this test is the slate-free path
    res = LLMResearcher(ctx, llm, adapter).research(req, store, None)
    p = llm.calls[0]["prompt"]
    assert res.status == "ok" and res.plan["knobs"][0]["scale"] == "log"
    assert res.plan["mechanism_prediction"] == {**mp, "size": pytest.approx(0.003)}   # 0.30 x 0.01
    assert "HEADROOM TABLE" in p and "Think from first principles" in p and "lr_end_frac" in p
    assert "Ledger (newest first" not in p and "## Surface" in p and "propose_ideas" not in llm.calls[0]["tools"]
