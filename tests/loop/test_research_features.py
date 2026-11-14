"""11-08 research features: the board ingest (loop.board), the root assumption ledger (loop.root_audit), the evidence
gate (ideation.require_evidence), the v3 idea-bank serving (ideation.bank_v3), training_only, and that every one of them
is OFF (no text, no schema change, no component) for nanopath."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop import board as BD
from loop import components as CG
from loop import root_audit as RA
from loop.agents import briefs as BR
from loop.agents.llm import FakeLLMClient
from loop.select import bank as BK
from loop.spec import load_context
from loop.store import SqliteStore

REPO = Path(__file__).resolve().parents[2]
FIX = Path(__file__).parent / "fixtures" / "board"
LOG, PROJ = json.loads((FIX / "log.json").read_text()), json.loads((FIX / "proj.json").read_text())
SRC = json.loads((FIX / "sources.json").read_text())


def fake_fetch(calls: list[str] | None = None):
    def fetch(url: str, timeout: float):
        if calls is not None:
            calls.append(url)
        if url == "P":
            return PROJ
        if url == "L":
            return LOG
        rid = url.rsplit("/", 1)[-1]
        if rid not in SRC:
            raise OSError("404")
        return SRC[rid]
    return fetch


ROOT_CFG = BD.flatten(BD._yaml(next(f["content"] for f in SRC[next(r["run_id"] for r in LOG["runs"]
                                                                    if r["title"] == "root-recipe")]["files"])))


@pytest.fixture
def nb(tmp_path, monkeypatch):
    """The nanobrain context with the board pointed at the fixtures (no network, no git)."""
    ctx = load_context(REPO / "tasks" / "nanobrain.yaml", state_dir=tmp_path / "state", require_measured=False)
    # the synthetic fixture board (tests/loop/fixtures/board): its root group is the root-recipe pair
    bc = dataclasses.replace(ctx.cfg.board, project_url="P", log_url="L", source_url="S/{run_id}",
                             root_runs=("root-recipe", "root-recipe-s2"))
    ctx.cfg = dataclasses.replace(ctx.cfg, board=bc)
    monkeypatch.setattr(BD, "root_config", lambda _ctx: {**ROOT_CFG, "time_schedule": True})   # our fork's change
    monkeypatch.setattr(BD, "_git_show", lambda repo, rev, path: None)
    store = SqliteStore(tmp_path / "nb.db", ctx.task_id)
    return ctx, store


@pytest.fixture
def nanopath(tmp_path):
    ctx = load_context(REPO / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path / "np", require_measured=False)
    return ctx, SqliteStore(tmp_path / "np.db", ctx.task_id)


# ------------------------------------------------------------------ board
def test_board_refresh_stores_runs_and_computes_the_digest(nb):
    ctx, store = nb
    res = BD.refresh(ctx, store, 1000.0, fake_fetch())
    assert res["ok"] and res["n_runs"] >= 13 and res["sources"] == len(SRC) and res["changed"]
    runs = store.board_runs()                                     # the loop db's board_runs table
    assert len(runs) == len(LOG["runs"]) and all(r["_source"] and r["_source"]["config"] for r in runs)
    d = BD.digest(store)
    assert d["root"]["titles"][:2] == ["root-recipe", "root-recipe-s2"] or set(d["root"]["titles"]) >= {
        "root-recipe", "root-recipe-s2"}
    assert d["root"]["n"] == 2 and abs(d["root"]["mean"] - 0.5327) < 5e-4
    assert d["noise"]["sd"] and 0.002 < d["noise"]["sd"] < 0.01 and d["noise"]["df"] >= 5
    labels = [r["label"] for r in d["refs"]]
    assert labels[0] == "Walnut" and "DINOv3" in labels
    text = BD.block(ctx, store)
    assert text.startswith("## Competition board digest")
    for s in ("SEED NOISE", "Walnut", "ONE-KEY CONTRASTS", "warmup_steps", "OUR FORK'S OWN CHANGES", "time_schedule",
              "Frozen reference models"):
        assert s in text, s


def test_board_material_changes_only_on_new_evidence(nb):
    ctx, store = nb
    first = BD.refresh(ctx, store, 1000.0, fake_fetch())
    again = BD.refresh(ctx, store, 2000.0, fake_fetch())
    assert first["changed"] and not again["changed"] and again["sources"] == 0      # sources fetched once


def test_board_step_cadence_and_failure_retry(nb):
    ctx, store = nb
    calls: list[str] = []
    clock = [10_000.0]
    b = BD.Board(ctx, store, None, lambda: clock[0], fake_fetch(calls))
    acts: list[str] = []
    b.step(acts)
    assert any("board:" in a for a in acts) and "L" in calls
    n = len(calls)
    gap = 3600.0 * float(BD.cfg_of(ctx).every_hours)               # the task's cadence (nanobrain 11-10: hourly)
    clock[0] += 0.5 * gap
    b.step(acts)
    assert len(calls) == n                                         # half the cadence: not due
    clock[0] += gap
    b.step(acts)
    assert len(calls) > n
    bad = BD.Board(ctx, store, None, lambda: clock[0] + 99 * 3600, lambda u, t: (_ for _ in ()).throw(OSError("down")))
    out: list[str] = []
    bad.step(out)
    assert "failed" in out[-1] and store.get_state(BD.K_BOOK)["last_ok"] is False


def test_source_is_read_as_text_and_diffed_against_the_root(nb):
    ctx, _ = nb
    src = {"files": [{"path": "src/m.py", "present": True, "content": "A = 1\nB = 3\n"},
                     {"path": "../evil.py", "present": True, "content": "x"},
                     {"path": "config.yaml", "present": True, "content": "base_lr: 0.002\nmodel_kwargs: {depth: 8}\n"}]}
    got = BD.source_digest(ctx, src, lambda p: "A = 1\nB = 2\n" if p == "src/m.py" else None)
    assert got["config"] == {"base_lr": 0.002, "model_kwargs.depth": 8}
    assert got["diff_stat"] == {"src/m.py": [1, 1]} and "+B = 3" in got["diff"]
    assert "../evil.py" not in got["files"]


def test_board_off_is_silent(nanopath):
    ctx, store = nanopath
    assert not BD.enabled(ctx) and BD.block(ctx, store) == ""
    acts: list[str] = []
    BD.Board(ctx, store, None, lambda: 1e9, lambda u, t: pytest.fail("fetched")).step(acts)
    assert acts == []


# ------------------------------------------------------------------ root audit
AUDIT = {"summary": "the root bakes in Muon wd and the level head",
         "root_assumptions": [
             {"choice": "weight_decay", "where": "config", "value": "0.05", "verdict": "untested", "evidence": "none",
              "tested_by": "", "arm": {"kind": "sweep_up", "config_diff": {"weight_decay": 0.1}, "mechanism": "Muon wd",
                                      "axis": "cls", "priority": 0.7}},
             {"choice": "MNI alignment", "where": "model.py:641", "value": "none", "verdict": "untested",
              "evidence": "Walnut / Neuro-JEPA register to MNI", "arm": {"kind": "swap", "change": "apply _mni.txt",
                                                                         "priority": 0.9, "axis": "reg"}},
             {"choice": "level_weight", "where": "config", "value": "0.5", "verdict": "real_gain",
              "evidence": "r11_lvl1", "arm": {"kind": "none"}}]}


def test_root_audit_banks_ranked_arms_once_and_renders_the_ledger(nb):
    ctx, store = nb
    BD.refresh(ctx, store, 1000.0, fake_fetch())
    assert RA.due(ctx, store)
    llm = FakeLLMClient({"review": [dict(AUDIT)]})
    ra = RA.RootAudit(ctx, store, lambda: llm, None, lambda: 2000.0)
    acts: list[str] = []
    ra.step(acts)
    assert "2 audit arm(s)" in acts[-1] and not RA.due(ctx, store)
    call = llm.calls[0]
    assert "Competition board digest" in call["prompt"]
    assert "Hard-coded constants" in call["prompt"]
    lines = [json.loads(l) for l in Path(RA.bank_path(ctx)).read_text().splitlines()]
    assert [l["id"] for l in lines] == ["root_audit-mni_alignment-swap", "root_audit-weight_decay-sweep_up"]
    assert all(l["priority"] for l in lines) and lines[1]["knobs"] == {"weight_decay": 0.1}
    # banked ideas load through the normal bank (extra fields kept)
    got = {e["id"]: e for e in BK.load([RA.bank_path(ctx)])}
    assert got["root_audit-weight_decay-sweep_up"]["axis"] == "cls"
    text = RA.block(ctx, store)
    assert text.startswith("## Root assumption ledger") and text.index("weight_decay") < text.index("level_weight")
    # same key: no re-audit; a materially new digest re-arms it
    store.set_state(BD.K_DIGEST, {**BD.digest(store), "material": "new"})
    assert RA.due(ctx, store)
    RA.accept(ctx, store, AUDIT, RA.audit_key(ctx, store), 3000.0)
    assert len(Path(RA.bank_path(ctx)).read_text().splitlines()) == 2            # ids already banked: not re-appended


def test_root_audit_waits_for_the_first_digest_and_retries_failures(nb):
    ctx, store = nb
    assert not RA.due(ctx, store)                                  # board on, no digest yet
    BD.refresh(ctx, store, 1000.0, fake_fetch())
    llm = FakeLLMClient({"review": [RuntimeError("boom")]})
    ra = RA.RootAudit(ctx, store, lambda: llm, None, lambda: 2000.0)
    acts: list[str] = []
    ra.step(acts)
    assert "failed" in acts[-1]
    ra.step(acts)
    assert len(llm.calls) == 1                                     # retried after an hour, not every tick


def test_constants_lists_module_constants_and_numeric_defaults():
    text = "LR_FLOOR = 1e-6\nclass X:\n    def f(self, depth: int = 12, tau=0.04, name='a'):\n        y = 3\n"
    got = RA.constants(text, "m.py", 10)
    assert "m.py:1 LR_FLOOR = 1e-6" in got and any("tau=0.04" in g for g in got) and any("depth=12" in g for g in got)


# ------------------------------------------------------------------ evidence gate
def test_evidence_problem(nb):
    from loop.agents.researcher import evidence_problem
    ctx, store = nb
    BD.refresh(ctx, store, 1000.0, fake_fetch())
    path = "it changes the frozen pooled token features the age regression probe reads through anatomy alignment"
    assert evidence_problem({"evidence": ["https://arxiv.org/abs/2301.08243 I-JEPA"], "readout_path": path}) == ""
    assert evidence_problem({"evidence": ["Walnut: registered, best age"], "readout_path": path}, ctx, store) == ""
    rid = LOG["runs"][0]["run_id"]
    assert evidence_problem({"evidence": [f"{rid} baseline"], "readout_path": path}, ctx, store) == ""
    assert "does not have" in evidence_problem({"evidence": ["run_sub_deadbeef00"], "readout_path": path}, ctx, store)
    assert "hypothesis.evidence" in evidence_problem({"evidence": ["my intuition"], "readout_path": path})
    assert "readout_path" in evidence_problem({"evidence": ["arXiv:2301.08243"], "readout_path": "it helps"})


def test_evidence_schema_and_bounce(nb, nanopath):
    from loop.agents import researcher as RS
    ctx, _ = nb
    hy = RS.finish_schema(ctx, v3=True)["properties"]["hypothesis"]["properties"]
    assert "evidence" in hy and "readout_path" in hy
    np_ctx, _ = nanopath
    hy = RS.finish_schema(np_ctx, v3=True)["properties"]["hypothesis"]["properties"]
    assert "evidence" not in hy and "readout_path" not in hy
    # the bounce: own counter, then the gap is recorded (intake parks it)
    t = SimpleNamespace(v3=True, need_evidence=True, evidence_cap=1, evidence_bounces=0, evidence_missing="",
                        evidence_check=RS.evidence_problem)
    why = t.evidence_check({"evidence": [], "readout_path": ""})
    assert why and "hypothesis.evidence" in why


def test_parked_claims_reach_the_next_literature_brief(nb):
    ctx, store = nb
    store.set_state(BR.K_EVIDENCE, [{"id": "req-v3-00007", "claim": "cross-sequence latent prediction", "kind": "bold"}])
    lit = "\n".join(BR.research_blocks(ctx, store, kind="literature"))
    assert BR.EVIDENCE_HEADING in lit and "cross-sequence latent prediction" in lit
    bold = "\n".join(BR.research_blocks(ctx, store, kind="bold"))
    assert "Evidence required" in bold and BR.EVIDENCE_HEADING not in bold


# ------------------------------------------------------------------ banks, training-only, components
def test_bank_accepts_research_agent_fields_and_nanobrain_families(tmp_path, nb):
    ctx, _ = nb
    p = tmp_path / "latent.jsonl"
    row = {"id": "nb-latent-ema-target", "title": "EMA latent target for masked patches", "family": "objective",
           "moonshot": True, "jepa": True, "priority": True,
           "idea": "model.py: an EMA teacher encodes the full volume; the student predicts its block-12 tokens",
           "why_it_might_work": "latent targets front-load semantic features (data2vec 2.0)",
           "v1_fate": "untested on nanobrain", "source": ["https://arxiv.org/abs/2212.07525"],
           "evidence": "data2vec 2.0: MAE accuracy at 16.4x less time; Walnut gap .023", "axis": "cls",
           "knobs": {"model_kwargs.latent_weight": [0.5, 1.0]}}
    p.write_text(json.dumps(row) + "\n")
    (e,) = BK.load([str(p)])
    assert e["family"] in ctx.cfg.families and e["priority"] and e["evidence"].startswith("data2vec")
    assert e["knobs"] == {"model_kwargs.latent_weight": [0.5, 1.0]} and e["axis"] == "cls"
    from loop.core.ideation import bank_text
    t = bank_text(e)
    assert "Build THIS idea" in t and "Evidence: data2vec" in t and "latent_weight" in t


def test_nanobrain_yaml_banks_exist_and_load():
    ctx = load_context(REPO / "tasks" / "nanobrain.yaml", require_measured=False, state_dir="/tmp/nb-bank-test")
    assert all(Path(p).exists() for p in ctx.cfg.idea_banks)     # the public spec ships no banks (optional)
    got = BK.load(ctx.cfg.idea_banks)                 # loadable (a family outside the task's set only orders the
    assert all(e["id"] and e["idea"] for e in got)    # interleave; the session declares the real one)


def test_ideation_serves_one_bank_idea_and_tags_its_node(nb, tmp_path):
    from loop.core.ideation import Ideation
    ctx, store = nb
    p = tmp_path / "b.jsonl"
    p.write_text("".join(json.dumps({"id": f"i{k}", "title": f"idea {k}", "idea": f"do thing {k}", "family": "objective",
                                     "priority": k == 2}) + "\n" for k in range(3)))
    ctx.cfg = dataclasses.replace(ctx.cfg, idea_banks=(str(p),))
    ide = Ideation(ctx, store)
    assert ide._bank_pick()["id"] == "i2"                          # priority first
    ide._claim_pair({"kind": "explore", "bank": "i2"}, "req-v3-00001")
    assert ide._bank_pick()["id"] == "i0"
    assert ide._bank_tag("req-v3-00001") == " bank=i2"


def test_training_only_drops_the_readout_kind(nb):
    from loop.core.ideation import Ideation, View
    ctx, store = nb
    v = View.__new__(View)
    assert Ideation(ctx, store)._readout_ok(v) is False
    assert "TRAINING-SIDE changes only" in "\n".join(BR.research_blocks(ctx, store))


def test_board_and_root_audit_are_audited_components(nb, nanopath):
    ctx, _ = nb
    assert CG.v3_components(ctx) == ("web", "postmortem", "board", "root_audit", "local_runs")
    np_ctx, np_store = nanopath
    assert CG.v3_components(np_ctx) == CG.V3_COMPONENTS
    with pytest.raises(ValueError):
        CG.draw(np_ctx, np_store, "board", "explore", "k")
    brief = "## A\nx\n## Competition board digest (c)\nrows\n## B\ny\n"
    assert BR.strip_section(brief, "## Competition board digest") == "## A\nx\n## B\ny\n"


def test_nanopath_research_blocks_are_empty(nanopath):
    ctx, store = nanopath
    store.set_state(BR.K_EVIDENCE, [{"id": "x", "claim": "y"}])
    assert BR.research_blocks(ctx, store, kind="literature") == []
    assert not ctx.cfg.scout.v3 and not ctx.cfg.ideation.bank_v3 and not RA.enabled(ctx)


def test_nanobrain_yaml_turns_the_research_features_on():
    ctx = load_context(REPO / "tasks" / "nanobrain.yaml", require_measured=False, state_dir="/tmp/nb-yaml-test")
    c = ctx.cfg
    assert c.board.enabled and c.root_audit.enabled and c.scout.enabled and c.scout.v3
    assert c.ideation.require_evidence and c.ideation.bank_v3 and c.ideation.training_only
    assert "nano-projects/nanobrain-smri" in c.board.project_url and "experiment-log" in c.board.log_url
    facts = " ".join(c.board.reference_facts)
    for s in ("Walnut .556", "nnFoundation ViT .492", "Neuro-JEPA .478", "DINOv3 ViT-B/16 .429", "objective or input space"):
        assert s in facts, s
    f = c.operator_focus
    assert "CONSTRAINT, not a lever" in f and "NATINIT" in f and "FROM SCRATCH" not in f and "buys more images" not in f


def test_board_and_ledger_lead_every_v3_brief(nb):
    from loop.core.ideation import Ideation
    ctx, store = nb
    BD.refresh(ctx, store, 1000.0, fake_fetch())
    RA.accept(ctx, store, AUDIT, RA.audit_key(ctx, store), 2000.0)
    ide = Ideation(ctx, store)
    ide.diag = lambda v: ("", [])
    secs = ide.head_sections({"kind": "explore"}, None)
    joined = "\n".join(secs)
    assert secs[0].startswith(BR.FOCUS_HEADING)
    for h in ("## Scope: TRAINING-SIDE", "## Competition board digest", "## Root assumption ledger"):
        assert h in joined, h
    prompt = RA.prompt(ctx, store)
    assert "## Competition board digest" in prompt and "Hard-coded constants" in prompt


def test_noop_knob_is_bounced_before_any_gpu(nb, nanopath, tmp_path):
    from loop.select import intake as IN
    ctx, store = nb
    rel = ctx.cfg.project.config_rel
    assert "model_kwargs.hard_gamma" in ctx.cfg.intake.noop_keys and "epochs" in ctx.cfg.intake.noop_keys
    fams = {k for f in ctx.cfg.families.values() for k in f.keys}
    for k in ("model_kwargs.hard_gamma", "model_kwargs.self_recon_weight", "epochs", "decay_epochs"):
        assert k not in fams, k                                      # out of every family (and lever group)
    assert not {"epochs", "decay_epochs"} & set(ctx.cfg.intake.lever_groups["optimizer"])
    cand = SimpleNamespace(diff_text="+  hard_gamma: 2.0\n", changed_paths=(rel,),
                           config_diff={"model_kwargs.hard_gamma": 2.0}, parent_commit="x",
                           hypothesis=SimpleNamespace(claim="x", family="objective", mechanism_tag="t"))
    rule, why = IN.diagnose(ctx, store, cand, operator=None, embedder=None)
    assert rule == "noop_knob" and "model.py:382" in why
    np_ctx, np_store = nanopath
    assert dict(np_ctx.cfg.intake.noop_keys) == {}


def _post(ctl_on: float, kids: dict[str, float]):
    """A posterior double: control 'p' at 0.50 on seeds 1..4 and ``ctl_on`` on the screen seeds 11, 12; children on 11, 12."""
    class O:
        def __init__(self, seed, y):
            self.seed, self.y, self.resumed = seed, y, 0

        def value(self, target, weights):
            return self.y
    by = {"p": [O(s, 0.50) for s in (1, 2, 3, 4)] + [O(s, ctl_on) for s in (11, 12)]}
    nodes = {"p": SimpleNamespace(parent="", base="b")}
    for k in kids:
        by[k] = [O(11, 0.5), O(12, 0.5)]
        nodes[k] = SimpleNamespace(parent="p", base="b")
    noise = SimpleNamespace(sigma2={"composite": 0.005 ** 2})
    return SimpleNamespace(led=SimpleNamespace(nodes=nodes), by=by, est={k: 1 for k in kids},
                           belief=lambda n: (kids[n], 1e-6), fit=lambda base, far=False: (noise, {}))


def test_control_bar_raises_the_entry_bar_on_a_low_shared_control():
    from loop.core import acquire as A
    from loop.core.model import COMPOSITE
    assert COMPOSITE == "composite"
    kids = {"a": 0.004, "b": 0.006, "c": 0.003, "d": -0.001}
    low = _post(0.492, kids)                            # control ran -.008 (z -1.9) on the screen seeds
    ok, why = A.control_bar("a", low, {}, 1.0, 0.7, 3)
    assert not ok and why.startswith("control_offset") and "3/4" in why
    assert A.control_bar("b", _post(0.492, {**kids, "b": 0.012}), {}, 1.0, 0.7, 3)[0]   # clears the shortfall
    assert A.control_bar("a", _post(0.499, kids), {}, 1.0, 0.7, 3)[0]                  # control normal there
    assert A.control_bar("a", _post(0.492, {"a": 0.004, "b": -0.002, "c": -0.003}), {}, 1.0, 0.7, 3)[0]  # few positive


def test_control_bar_is_configured_only_for_nanobrain(nb, nanopath):
    from loop.core import acquire as A, status as ST
    saved = [(m, {k: v for k, v in vars(m).items() if k.isupper()}) for m in (A, ST)]
    try:
        A.configure(nanopath[0].cfg)
        assert A.CONTROL_BAR is None
        A.configure(nb[0].cfg)
        assert A.CONTROL_BAR == (1.0, 0.7, 3)
    finally:
        for m, g in saved:
            for k, v in g.items():
                setattr(m, k, v)
