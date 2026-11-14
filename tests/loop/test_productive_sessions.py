"""11-08: make the research sessions productive (1,207 failed nanobrain sessions, 30,580 slate proposals).

1. the mechanism memory: the imported manual runs, measured nodes and bounced candidates are visible AND binding --
   the slate gate / check_idea bounce a repeat (fuzzy, across reworded tags) with the prior result, unless the
   candidate says what is different;
2. in-flight work: another live session's chosen mechanism is listed at session start and a near-duplicate bounced;
3. the sandbox: the project python and `cd` inside the worktree run; writes / network / env vars outside stay refused;
4. the knowledge file + the bar (2 paired sd per axis) reach every brief.
(5, loop.selfcheck: tests/loop/test_selfcheck.py)
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop import knowledge as KN
from loop.agents import briefs as BR
from loop.agents import history as HX
from loop.agents import researcher as RS
from loop.contracts import K_INFLIGHT, Worktree
from loop.core import bold as BOLD
from loop.core import tested as TS
from loop.spec import load_context
from loop.store import SqliteStore

REPO = Path(__file__).resolve().parents[2]
AXES = ("nb_cls_mean", "nb_reg_mean", "nb_seg_mean")
KNOWLEDGE = """### Where the score is
- MNI rigid registration gives seg +.018; FINO fb gives age +.08 and cnp, loses seg -.009.
- The Walnut gap is adhd / soop_mrs; site is 98% linearly decodable from fb's embedding.

## Tested mechanisms
- local:mni-affine-alignment: MNI affine alignment / MNI affine resampling: volumes resampled into MNI152 space with the shipped affine
- local:register-tokens: register tokens / registers: 4 extra learnable register tokens in the encoder
- local:drop-path: drop path / stochastic depth: drop_path_rate 0 -> 0.1
- local:i-jepa: I-JEPA / JEPA objective / EMA latent prediction: predict EMA-teacher latents of masked blocks
"""


def _row(name, seed, d_panel, cfg, branch, files=()):
    s = {"panel": 0.53 + d_panel, "nb_cls_mean": 0.31, "nb_reg_mean": 0.86, "nb_seg_mean": 0.75 + d_panel}
    return {"name": name, "kind": "full", "seed": seed, "lineage": "root", "comparable": True, "scores": s,
            "config_delta": cfg, "code_numstat": {f: [10, 0] for f in files}, "lineage_numstat": {}, "branch": branch,
            "sha": "abc", "code_hash": name, "integrity": [], "requeued": False, "steps": 1}


@pytest.fixture
def nb(tmp_path):
    ctx = load_context(REPO / "tasks" / "nanobrain.yaml", state_dir=tmp_path / "state", require_measured=False)
    (k := tmp_path / "knowledge.md").write_text(KNOWLEDGE)
    lc = dataclasses.replace(ctx.cfg.local_runs, enabled=True, reference_runs={"7": "ref-s7"}, incomparable=())
    ctx.cfg = dataclasses.replace(ctx.cfg, local_runs=lc,
                                  knowledge=dataclasses.replace(ctx.cfg.knowledge, files=(str(k),)))
    store = SqliteStore(tmp_path / "nb.db", ctx.task_id)
    for r in (_row("ref-s7", 7, 0.0, {}, "exp/w1"),
              _row("mni-affine-alignment-s7", 7, 0.002, {"mni_align": [None, "affine"]}, "exp/mni", ["src/nanobrain/mni.py"]),
              _row("register-tokens", 7, -0.004, {"model_kwargs.class_tokens": [0, 4]}, "exp/arch-reg4"),
              _row("drop-path", 7, -0.003, {"model_kwargs.drop_path_rate": [0.0, 0.1]}, "exp/arch-dpath"),
              _row("i-jepa", 7, -0.019, {"objective": [None, "ijepa"]}, "exp/ssl")):
        store.put_local_run(r["name"], r["name"], r, 1.0)
    yield ctx, store
    store.close()


def _cand(title, tag, mech, cat="pretext", **kw):
    return {"title": title, "mechanism_tag": tag, "family": "objective", "category": cat,
            "mechanism": mech + " " + " ".join(f"{tag}p{j}" for j in range(12)), "why": "x", "plausibility": 0.3,
            "probability": 0.2, **kw}


# ---------------------------------------------------------------------------------------------- 1. the memory
def test_every_manual_run_is_in_the_memory_described_and_with_its_result(nb):
    ctx, store = nb
    mem = {e.id: e for e in TS.memory(ctx, store)}
    assert {"local:mni-affine-alignment-s7", "local:register-tokens", "local:drop-path", "local:i-jepa"} <= set(mem)
    assert mem["local:register-tokens"].title.startswith("register tokens")              # the knowledge file's description
    assert "panel" in mem["local:i-jepa"].result and "HARMFUL" in mem["local:i-jepa"].result
    block = TS.brief_block(ctx, store)                       # every group, one line each (no truncated digest)
    assert TS.HEADING in block and all(f"local:{n}" in block for n in ("register-tokens", "drop-path", "i-jepa", "mni-affine-alignment-s7"))
    text = "\n".join(BR.research_blocks(ctx, store))         # ... and every v3 brief carries it
    assert TS.HEADING in text


@pytest.mark.parametrize("title,tag,mech,prior", [
    ("MNI-affine canonical input alignment", "mni_affine_align",
     "Resample each training volume with its shipped _mni.txt affine so positions mean anatomy.", "local:mni-affine-alignment-s7"),
    ("MNI-affine resampling of training inputs", "mni_affine_input",
     "Resample each volume through its MNI affine so tokens are in canonical space.", "local:mni-affine-alignment-s7"),
    ("Global register tokens", "encoder_registers", "Add 4 register tokens to the encoder.", "local:register-tokens"),
    ("Stochastic depth in the encoder", "drop_path_on", "Turn on drop path at rate 0.1.", "local:drop-path"),
    ("EMA latent prediction auxiliary", "ema_latent_aux",
     "Predict the EMA teacher's latents of the masked tokens (I-JEPA style).", "local:i-jepa"),
])
def test_a_reworded_repeat_is_bounced_with_the_prior_result(nb, title, tag, mech, prior):
    ctx, store = nb
    hit = TS.screen(ctx, store, [_cand(title, tag, mech)], "req-a")[0]
    assert hit is not None and hit.kind == "tested" and hit.entry.id == prior, hit
    assert "panel" in hit.cite()


def test_a_new_mechanism_passes_and_a_stated_difference_excuses_a_repeat(nb):
    ctx, store = nb
    new = _cand("Per-collection bias terms in the metadata heads", "collection_bias_heads",
                "Give each metadata head a learned per-collection bias so the pooled vector stops encoding site.")
    assert TS.screen(ctx, store, [new], "req-a") == [None]
    rep = _cand("Global register tokens", "encoder_registers", "Add 4 register tokens to the encoder.")
    excused = {**rep, "differs_from_tested": [{"id": "local:register-tokens", "how": "registers read by a site-adversarial "
                                                                          "probe head that removes scanner identity"}]}
    lazy = {**rep, "differs_from_tested": [{"id": "local:register-tokens", "how": "register tokens encoder"}]}
    assert TS.screen(ctx, store, [excused, lazy], "req-a") == [None, TS.screen(ctx, store, [lazy], "req-a")[0]]
    assert TS.screen(ctx, store, [lazy], "req-a")[0] is not None


def test_the_slate_gate_bounces_repeats_remembers_them_and_cites_the_original_prior(nb):
    ctx, store = nb
    cats = ["pretext", "data", "architecture", "schedule", "context", "pathology"]
    reg = _cand("Encoder register tokens", "encoder_registers", "Add 4 register tokens to the encoder.", "architecture")
    fresh = [_cand(f"novel idea {i} zz{i}", f"novel_{i}", " ".join(f"q{i}w{j}" for j in range(14)), cats[i])
             for i in range(5)]
    g = BOLD.Slate(ctx, store, "req-1", "bold", logger=store.log)
    lim = BOLD.slate_limits(ctx, "bold")
    cands = ([reg] + fresh + [_cand(f"more {i} yy{i}", f"more_{i}", " ".join(f"r{i}w{j}" for j in range(14)), cats[i % 6])
                              for i in range(max(0, lim["k_min"] - 6))])
    out = g(cands)
    assert g.done and g.result["chosen"] != 0, out
    assert "REJECTED" in out and "local:register-tokens" in out                           # bounced, with the prior cited
    rej = store.get_state(TS.K_REJECTS)
    assert rej and rej[0]["prior"] == "local:register-tokens" and rej[0]["id"] == "slate:req-1#0"
    # a later session rewords it: matched against the remembered bounce, and it cites the ORIGINAL prior's result
    again = TS.screen(ctx, store, [_cand("Encoder register tokens for global context", "registers_v2",
                                         "Add 4 register tokens to the encoder.")], "req-2")[0]
    assert again is not None and "local:register-tokens" in again.cite()
    # a slate of nothing but repeats is rejected with every prior named (and costs a try)
    g2 = BOLD.Slate(ctx, store, "req-3", "bold", logger=store.log)
    forms = [("Encoder register tokens", "Add 4 register tokens to the encoder."),
             ("Register tokens for global context", "Learnable register tokens beside the patches."),
             ("Drop path regularisation", "Turn on drop path at 0.1."),
             ("Stochastic depth", "Drop path / stochastic depth in every encoder block."),
             ("I-JEPA objective", "Predict EMA teacher latents of masked blocks, JEPA style."),
             ("EMA latent prediction", "EMA latent prediction of masked tokens."),
             ("MNI affine resampling", "Resample inputs with the MNI affine."),
             ("MNI affine alignment", "Align volumes to MNI152 with the shipped affine.")]
    reps = [_cand(t, f"rep_{i}", m, c) for i, ((t, m), c) in enumerate(zip((forms * 2)[:len(cands)], (cats * 3)))]
    out2 = g2(reps)
    assert not g2.done and out2.startswith("REJECT slate") and "local:drop-path" in out2 and g2.tries == 1
    assert store.events("slate_repeats")                               # logged for the meta report


def test_in_flight_work_is_listed_at_session_start_and_a_duplicate_is_bounced(nb):
    ctx, store = nb
    store.set_state(K_INFLIGHT, [{"id": "req-9"}, {"id": "req-8"}])
    TS.record_slate(ctx, store, "req-9", [_cand("Site-adversarial collection head", "site_adv_head",
                                                "A gradient-reversal head on collection identity.")], [None], 0)
    TS.record_slate(ctx, store, "req-8", [_cand("Anatomical mirror pretext zebra", "mirror_zebra",
                                                "Predict the contralateral hemisphere patch.")], [None], None)
    dup = _cand("Collection adversarial head with gradient reversal", "site_adversary",
                "A gradient-reversal head on collection identity.")
    hit = TS.screen(ctx, store, [dup], "req-7")[0]
    assert hit is not None and hit.kind == "inflight" and hit.entry.request_id == "req-9"
    assert TS.screen(ctx, store, [dup], "req-9") == [None]            # a session never collides with itself
    block = TS.session_block(ctx, store, "req-7")
    assert TS.SESSION_HEADING in block and "IN FLIGHT slate:req-9#0" in block and "proposed slate:req-8#0" in block
    store.set_state(K_INFLIGHT, [])                                   # the session ended: no longer in flight
    assert TS.screen(ctx, store, [dup], "req-7") == [None]


def test_check_idea_and_history_see_the_memory(nb, tmp_path):
    ctx, store = nb
    t = RS.WorktreeTools(None, Worktree(str(tmp_path), "b", "c"), writable=("*",), locked=(), checker=lambda **k: "OK")
    t.memory_check = lambda **kw: TS.check_text(ctx, store, "req-1", **kw)
    t.request_id = "req-1"
    out = t.t_check_idea(mechanism_tag="encoder_registers", claim="add register tokens to the encoder")
    assert out.startswith("REJECT already_tested") and "local:register-tokens" in out
    assert t.t_check_idea(mechanism_tag="collection_bias_heads", claim="per-collection bias in heads").startswith("OK")
    h = HX.Historian(ctx, store)
    assert "local:register-tokens" in h(action="tested") and "tested" in HX.TOOL_DEF["input_schema"]["properties"]["action"]["enum"]
    assert "differs_from_tested" in BOLD.CANDIDATE["properties"]


def test_tested_off_restores_the_old_gate(nb):
    ctx, store = nb
    off = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, tested=dataclasses.replace(ctx.cfg.tested,
                                                                                               enabled=False)))
    assert TS.screen(off, store, [_cand("Global register tokens", "encoder_registers", "Add registers.")]) == [None]
    assert TS.brief_block(off, store) == ""


# ---------------------------------------------------------------------------------------------- 3. the sandbox
@pytest.fixture
def wt(tmp_path):
    root = tmp_path / "wt"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("X = 41\n")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    (root / ".venv").symlink_to(venv)
    return RS.WorktreeTools(None, Worktree(str(root), "b", "c"), writable=("src/**",), locked=(),
                            python=str(venv / "bin" / "python")), root


@pytest.mark.parametrize("cmd", [
    ".venv/bin/python -c 'print(1)'",
    "cd src/pkg && ../../.venv/bin/python -c 'print(2)'",
    "cd src && python -c 'import pkg; print(pkg.X + 1)'",
    "cd src/pkg; grep -n X ../pkg/__init__.py",
    "D=src/pkg; ls $D",
    "awk -F/ '{print $NF}' src/pkg/__init__.py",
    "python3 -c \"print('a/b'.split('/'))\"",
    "grep -n '/def X/' src/pkg/__init__.py 2>/dev/null; echo ok > src/pkg/out.txt",
])
def test_the_project_python_cd_and_local_vars_run_in_the_worktree(wt, cmd):
    tools, root = wt
    out = tools.t_bash(cmd)
    assert out.startswith("exit=0"), out


def test_the_project_python_is_the_one_on_path(wt):
    tools, root = wt
    out = tools.t_bash("cd src && python -c 'import pkg, sys; print(pkg.X + 1, sys.prefix != \"\")'")
    assert out.startswith("exit=0") and "42" in out, out


@pytest.mark.parametrize("cmd,why", [
    ("cd /tmp && ls", "leaves the worktree"),
    ("cd ../.. && ls", "leaves the worktree"),
    ("ls $BRAINMARKS_SMRI_DATA", "environment variable"),
    ("ls $(dirname $PWD)", "command substitution"),
    ("echo x > /tmp/escape.txt", "outside the worktree"),
    ("cp src/pkg/__init__.py .venv/bin/evil", "shared .venv"),
    ("cat /etc/passwd", "outside the worktree"),
    (".venv/bin/python -c 'import urllib.request'", "network"),
    ("python -c 'import subprocess; subprocess.run([\"git\", \"commit\", \"-m\", \"x\"])'", "git state change"),
    ("sbatch job.sh", "SLURM"),
    (".venv/bin/pip install x", "package manager"),
])
def test_the_sandbox_still_refuses_escapes_writes_network_git_and_slurm(wt, cmd, why):
    tools, root = wt
    out = tools.t_bash(cmd)
    assert out.startswith("REFUSED") and why in out, out
    assert not Path("/tmp/escape.txt").exists() or Path("/tmp/escape.txt").read_text() != "x\n"


def test_a_worktree_script_with_network_code_is_refused(wt):
    tools, root = wt
    (root / "src" / "fetch.py").write_text("import requests\nrequests.get('http://x')\n")
    assert "network" in tools.t_bash("cd src && python fetch.py")


def test_the_replayed_11_08_refusals_now_run():
    """The commands the 11-08 sessions were refused for most (relative to their worktree)."""
    for cmd in ["cd src/nanobrain/config && grep -n model_kwargs r12.yaml; grep -rn config ../train.py | head -1",
                "timeout 110 .venv/bin/python -c \"import sys; sys.path.insert(0,'src')\"",
                "cd src/nanobrain; awk -F/ '{print $NF}' config/list.txt | sort | uniq -c"]:
        t = RS.WorktreeTools(None, Worktree("/nonexistent/wt", "b", "c"), writable=(), locked=(),
                             python="/nonexistent/venv/bin/python")
        assert t._confined(cmd) == "", cmd


# ---------------------------------------------------------------------------------------------- 4. knowledge + bar
def test_the_knowledge_file_and_the_bar_reach_every_brief(nb):
    ctx, store = nb
    text = "\n".join(BR.research_blocks(ctx, store))
    assert KN.HEADING in text and "98% linearly decodable" in text and "Walnut gap is adhd" in text
    assert "## Tested mechanisms" not in text                       # the run descriptions feed the memory instead
    sd = ctx.cfg.axis_sd
    assert KN.BAR_HEADING in text and f"{2 * sd['nb_cls_mean']:.4f}" in text and f"{2 * sd['nb_seg_mean']:.4f}" in text
    assert "ONE SPECIFIC AXIS" in text and "+-.004" in text
    floor = RS.LLMResearcher._bar_sizing(SimpleNamespace(ctx=ctx))      # the researcher's sizing line says it too
    assert "ONE named axis" in floor and f"{2 * sd['nb_reg_mean']:.4f}" in floor
    assert KN.descriptions(ctx)["register-tokens"].startswith("register tokens")
    assert KN.describe(["mni-affine-alignment-s7"], KN.descriptions(ctx)).startswith("MNI affine")


def test_the_shipped_nanobrain_knowledge_file_is_found_from_the_domain_dir(tmp_path):
    ctx = load_context(REPO / "tasks" / "nanobrain.yaml", state_dir=tmp_path, require_measured=False)
    assert [p.name for p in KN.paths(ctx)] == ["knowledge.md"]
    d = KN.descriptions(ctx)
    assert {"mni-affine-alignment", "register-tokens", "drop-path", "latent-prediction", "i-jepa"} <= set(d)
    assert "adhd" in KN.block(ctx) and "soop_mrs" in KN.block(ctx)


def test_nanopath_has_no_knowledge_block(tmp_path):
    ctx = load_context(REPO / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path, require_measured=False)
    assert KN.block(ctx) == ""
