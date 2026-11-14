"""loop.localruns: the operator's manual runs imported as external evidence (never ledger nodes), idempotent, gated."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from loop import localruns as LR
from loop.agents import briefs as BR
from loop.spec import load_context
from loop.store import SqliteStore

REPO = Path(__file__).resolve().parents[2]
CLS = ["abide_diagnosis", "adhd_diagnosis", "cnp_diagnosis", "ucsf_idh", "upenn_survival_1y", "soop_mrs_poor"]
REG = ["ixi_age", "openbhb_age"]
SEG = ["brats_enhancing_tumor", "brats_whole_tumor", "soop_lesion"]


def _run(wt: Path, name: str, seed: int, seg: float, cfg_extra: str = "", done: bool = True) -> Path:
    out = wt / "output" / name
    (out / "eval").mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps({"done": done, "step": 30000, "images": 960000,
                                                  "wall_seconds": 3573.0, "git": {"sha": "", "dirty": False}}))
    (out / "config.yaml").write_text(f"name: {name}\nseed: {seed}\nbase_lr: 0.002\n{cfg_extra}")
    agg = {}
    for t in CLS + REG + SEG:
        v = seg if t in SEG else 0.5
        if t in SEG:
            d = {"task": t, "type": "segmentation", "eval_samples": [{"dice": v}]}
        else:
            d = {"task": t, "type": "regression", "target": "y",
                 "eval_samples": [{"y": 0.0, "prediction": 0.0}, {"y": 2.0, "prediction": 1.0}]}
        (out / "eval" / f"{t}.json").write_text(json.dumps(d))
        agg[t] = {"score": v}
    (out / "eval" / "aggregate.json").write_text(json.dumps({"summary": agg}))
    return out


@pytest.fixture
def nb(tmp_path, monkeypatch):
    ctx = load_context(REPO / "tasks" / "nanobrain.yaml", state_dir=tmp_path / "state", require_measured=False)
    wt = tmp_path / "wts" / "w"
    tab = tmp_path / "jobs.tsv"
    tab.write_text("name\tjobid\tworktree\noref\t1\tw\noarm\t2\tw\nosmk\t3\tw\nopend\t4\tw\nogone\t5\tw\n")
    lc = dataclasses.replace(ctx.cfg.local_runs, job_tables=(str(tab),), worktrees_root=str(tmp_path / "wts"),
                             lineages={}, reference_runs={"7": "oref"})
    ctx.cfg = dataclasses.replace(ctx.cfg, local_runs=lc)
    monkeypatch.setattr("loop.board.root_config", lambda _c: {"base_lr": 0.002, "seed": 7338, "name": "r"})
    monkeypatch.setattr(LR._Git, "facts", lambda self, sha: {"lineage": "root", "numstat": {}, "diff": "",
                                                             "code_hash": "", "lineage_numstat": {}})
    _run(wt, "oref", 7, 0.75)
    _run(wt, "oarm", 7, 0.80, "model_kwargs:\n  level_mirror: [1.0]\n")
    smk = _run(wt, "osmk", 7, 0.70)
    (smk / "summary.json").write_text(json.dumps({"done": True, "wall_seconds": 300.0, "git": {}}))
    _run(wt, "opend", 7, 0.7, done=False)
    return ctx, SqliteStore(tmp_path / "nb.db", ctx.task_id), wt


def test_import_is_external_idempotent_and_reaches_briefs(nb):
    ctx, store, wt = nb
    res = LR.import_runs(ctx, store, 100.0)
    assert (res["new"], res["finished"], res["missing"]) == (3, 3, ["ogone"])
    assert any(u.startswith("opend") for u in res["unfinished"])
    assert store.nodes() == [] or all(n.id == "root" for n in store.nodes())     # never ledger nodes
    rows = {r["name"]: r for r in store.local_runs()}
    arm = rows["oarm"]
    assert arm["config_delta"] == {"model_kwargs.level_mirror": [None, [1.0]]} and arm["seed"] == 7
    assert rows["osmk"]["kind"] == "smoke" and arm["kind"] == "full"
    assert abs(arm["scores"]["nb_seg_mean"] - 0.80) < 1e-9 and not arm["integrity"]
    assert res["promising"] == [["oarm"]] and res["banked"] == 1
    again = LR.import_runs(ctx, store, 200.0)                       # nothing changed on disk: nothing re-read
    assert (again["new"], again["updated"], again["unchanged"], again["banked"]) == (0, 0, 3, 0)
    (wt / "output" / "oarm" / "REQUEUED").write_text("restart 1\n")
    assert LR.import_runs(ctx, store, 300.0)["updated"] == 1
    assert {r["name"]: r for r in store.local_runs()}["oarm"]["requeued"]
    text = "\n".join(BR.research_blocks(ctx, store))
    assert LR.HEADING in text and "local:oarm" in text and "NOT CRN-paired" in text and "PROMISING" in text
    assert LR.digest_lines(store)[1].startswith("- PROMISING: oarm")
    bank = [json.loads(x) for x in Path(LR.bank_path(ctx)).read_text().splitlines()]
    assert [b["id"] for b in bank] == ["local-confirm-oarm"] and bank[0]["priority"]


def test_tick_handle_cadence(nb):
    ctx, store, _ = nb
    now = [1000.0]
    h = LR.LocalRuns(ctx, store, None, lambda: now[0])
    acts: list[str] = []
    h.step(acts)
    assert any(a.startswith("local runs:") for a in acts)
    acts.clear()
    now[0] += 60
    h.step(acts)
    assert acts == [] and not h.due(now[0])
    now[0] += 3600
    assert h.due(now[0])


def test_off_for_nanopath(tmp_path):
    ctx = load_context(REPO / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path, require_measured=False)
    assert not LR.enabled(ctx) and LR.block(ctx, None) == ""
    assert LR.read_tables([str(tmp_path / "none.tsv")]) == []


def test_headerless_table(tmp_path):
    p = tmp_path / "w3.tsv"
    p.write_text("a\t1\twt\texp/b\tabc\tacct\tx=1\tarch\n")
    assert LR.read_tables([str(p)])[0] | {} == {"name": "a", "jobid": "1", "worktree": "wt", "branch": "exp/b",
                                                 "sha": "abc", "account": "acct", "overrides": "x=1", "lane": "arch",
                                                 "_table": "w3"}


def test_reroot_turns_the_old_roots_runs_into_context_never_controls(tmp_path, monkeypatch):
    """Re-root 11-08 PM (r12 -> FINO fb): runs on the PREVIOUS root's lineage are diffed against THAT root's
    same-seed replicate (context: config / code vs it, never PROMISING / banked), the new root's arms against the new
    replicate; a stored run is re-read when the root sha changes (its stored delta was against the old root)."""
    ctx = load_context(REPO / "tasks" / "nanobrain.yaml", state_dir=tmp_path / "state", require_measured=False)
    wt = tmp_path / "wts" / "w"
    tab = tmp_path / "jobs.tsv"
    tab.write_text("name\tjobid\tworktree\n" + "".join(f"{n}\t{i}\tw\n" for i, n in
                                                       enumerate(["old_ref", "old_arm", "new_ref", "new_arm"])))
    lc = dataclasses.replace(ctx.cfg.local_runs, job_tables=(str(tab),), worktrees_root=str(tmp_path / "wts"),
                             lineages={"new": "n", "old": "o"}, incomparable=(), reference_runs={"7": "new_ref"},
                             context_refs={"old": {"7": "old_ref"}})
    ctx.cfg = dataclasses.replace(ctx.cfg, local_runs=lc)
    root = {"base_lr": 0.002, "fino3d.enabled": True}                       # the NEW root's config
    monkeypatch.setattr("loop.board.root_config", lambda _c: dict(root))
    lineage = {"old_ref": "old", "old_arm": "old", "new_ref": "new", "new_arm": "new"}
    monkeypatch.setattr(LR._Git, "facts", lambda self, sha: {
        "lineage": lineage[sha], "numstat": {"src/fino.py": [0, 150]} if lineage[sha] == "old" else {}, "diff": "",
        "code_hash": "old" if lineage[sha] == "old" else "", "lineage_numstat": {}})
    for name, seg, extra in (("old_ref", 0.75, "fino3d:\n  enabled: false\n"),
                             ("old_arm", 0.80, "fino3d:\n  enabled: false\nmni_align: rigid\n"),
                             ("new_ref", 0.74, "fino3d:\n  enabled: true\n"),
                             ("new_arm", 0.74, "fino3d:\n  enabled: true\n  age: 1.0\n")):
        out = _run(wt, name, 7, seg, extra)
        s = json.loads((out / "summary.json").read_text())
        (out / "summary.json").write_text(json.dumps({**s, "git": {"sha": name}}))
    store = SqliteStore(tmp_path / "nb.db", ctx.task_id)
    res = LR.import_runs(ctx, store, 100.0)
    g = {x["names"][0]: x for x in res["digest"]["groups"]}
    old = g["old_arm"]
    assert old["context"] and not old["comparable"] and old["verdict"] == "context (previous root)"
    assert abs(old["delta"]["nb_seg_mean"] - 0.05) < 1e-9                  # vs old_ref, NOT vs new_ref (+.06)
    assert old["config_delta"] == {"mni_align": [None, "rigid"]}           # vs its own root replicate's config
    assert res["promising"] == [] and res["banked"] == 0                   # a +.05 seg context run is never banked
    assert g["new_arm"]["comparable"] and g["new_arm"]["config_delta"] == {"fino3d.age": [None, 1.0]}
    text = res["digest"]["text"]
    assert "PREVIOUS ROOT lineage" in text and "local:old_arm" in text and "previous root lineage 'old'" in text
    assert LR.import_runs(ctx, store, 200.0)["unchanged"] == 4             # same root: nothing re-read
    proj = dataclasses.replace(ctx.cfg.project, base_sha="f" * 40)
    ctx.cfg = dataclasses.replace(ctx.cfg, project=proj)
    again = LR.import_runs(ctx, store, 300.0)                               # a re-root: every stored run re-read
    assert again["updated"] == 4 and all(r["root"] == "f" * 40 for r in store.local_runs())


def test_root_replicates_register_identical_runs_as_root_baselines(tmp_path, monkeypatch):
    """11-08 re-root: the fresh root 9035f61 had no seeds, so no noise fit ("no config with >= 2 seeds") and the loop
    could not rank anything. The operator's a root replicate run / a root replicate run / a root replicate run ARE the root (no .py diff vs the root commit,
    the same effective config but seed / name / output): registered once as done root:baseline:<seed> runs, so the
    root has >= 2 seeds; a run with a config / code change, a ckpt, no sha, a smoke, or a seed the root already ran is
    never one."""
    from loop.contracts import ROOT_NODE_ID, Run, RunStatus, Stage
    from loop.core import data as CD
    ctx = load_context(REPO / "tasks" / "nanobrain.yaml", state_dir=tmp_path / "state", require_measured=False)
    names = ["ra", "rb", "rc", "rarm", "rckpt", "rnosha", "rsmk", "rtaken"]
    tab = tmp_path / "jobs.tsv"
    tab.write_text("name\tjobid\tworktree\n" + "".join(f"{n}\t{i}\tw\n" for i, n in enumerate(names)))
    lc = dataclasses.replace(ctx.cfg.local_runs, job_tables=(str(tab),), worktrees_root=str(tmp_path / "wts"),
                             lineages={}, reference_runs={"7": "ra"})
    ctx.cfg = dataclasses.replace(ctx.cfg, local_runs=lc)
    root = ctx.cfg.project.base_sha
    monkeypatch.setattr("loop.board.root_config", lambda _c: {"base_lr": 0.002, "seed": 7338, "name": "r"})
    monkeypatch.setattr(LR._Git, "facts", lambda self, sha: {
        "lineage": "root", "numstat": {"src/x.py": [1, 0]} if sha == "code" else {}, "diff": "",
        "code_hash": "c0de" if sha == "code" else "", "lineage_numstat": {}})
    wt = tmp_path / "wts" / "w"
    for name, seed, seg, extra in (("ra", 7, 0.75, ""), ("rb", 8, 0.74, ""), ("rc", 9, 0.76, ""),
                                   ("rarm", 10, 0.80, "level_weight: 1.0\n"), ("rckpt", 11, 0.7, "ckpt: x.pth\n"),
                                   ("rnosha", 12, 0.7, ""), ("rsmk", 13, 0.7, ""), ("rtaken", 14, 0.7, "")):
        out = _run(wt, name, seed, seg, extra)
        s = json.loads((out / "summary.json").read_text())
        s["git"] = {"sha": "" if name == "rnosha" else "abc"}
        if name == "rsmk":
            s["wall_seconds"] = 300.0
        (out / "summary.json").write_text(json.dumps(s))
    (wt / "output" / "rb" / "REQUEUED").write_text("restart 1\n")
    store = SqliteStore(tmp_path / "nb.db", ctx.task_id)
    store.ensure_root(root)
    store.put_run(Run(id=Run.make_id(ROOT_NODE_ID, Stage.BASELINE, 14), task_id=ctx.task_id, node_id=ROOT_NODE_ID,
                      seed=14, stage=Stage.BASELINE, status=RunStatus.RUNNING))
    res = LR.import_runs(ctx, store, 100.0)
    assert res["root_replicates"] == ["root:baseline:7", "root:baseline:8", "root:baseline:9"]
    got = {r.seed: r for r in store.runs(node_id=ROOT_NODE_ID, status=[RunStatus.DONE])}
    assert sorted(got) == [7, 8, 9] and got[8].resumed == 1 and got[7].stage is Stage.BASELINE
    assert abs(got[7].aggregate - store.local_runs()[0]["scores"]["panel"]) < 1e-12
    assert abs(got[9].scores["nb_seg_mean"] - 0.76) < 1e-12 and got[7].scores["nb_steps"] == 30000.0
    assert got[7].output_dir.endswith("output/ra") and "nb_task_soop_lesion" in got[7].scores
    assert "local runs:" in LR.LocalRuns._line(res) and "root:baseline:7" in LR.LocalRuns._line(res)
    assert [n.id for n in store.nodes()] == [ROOT_NODE_ID]                  # never a ledger node
    led = CD.load(tmp_path / "nb.db", ctx.task_id)                         # the ledger sees 3 root seeds
    assert sorted(o.seed for o in led.obs if o.node == ROOT_NODE_ID) == [7, 8, 9]
    assert LR.import_runs(ctx, store, 200.0)["root_replicates"] == []      # idempotent
    proj = dataclasses.replace(ctx.cfg.project, base_sha="f" * 40)        # another root: nothing registered
    ctx.cfg = dataclasses.replace(ctx.cfg, project=proj)
    store2 = SqliteStore(tmp_path / "nb2.db", ctx.task_id)
    store2.ensure_root(root)
    assert LR.import_runs(ctx, store2, 100.0)["root_replicates"] == []
    off = dataclasses.replace(ctx.cfg, local_runs=dataclasses.replace(lc, root_replicates=False))
    assert LR.register_root_replicates(SimpleNamespace(cfg=off), store2, store2.local_runs(), {}, 1.0) == []


def test_replicate_why_not_names_the_reason():
    row = {"kind": "full", "root": "r", "sha": "s", "comparable": True, "context": False, "code_hash": "",
           "code_numstat": {}, "integrity": [], "seed": 7, "out": "/nonexistent"}
    assert LR.replicate_why_not({**row, "kind": "smoke"}, {"a": 1}, "r") == "not a full run"
    assert "another root" in LR.replicate_why_not(row, {"a": 1}, "q")
    assert "code differs" in LR.replicate_why_not({**row, "code_hash": "x"}, {"a": 1}, "r")
    assert "unreadable" in LR.replicate_why_not(row, {"a": 1}, "r")
    assert "seed" in LR.replicate_why_not({**row, "seed": True}, {"a": 1}, "r")


def test_root_seed_noise_floor_until_k_root_seeds(tmp_path):
    """nanobrain (loop.project.noise_floor_seeds 5): with < 5 done root seeds every fitted per-run sigma is at least
    the spec's measured one (paired sd / sqrt 2); from 5 seeds on the fit stands. nanopath: no floor at all."""
    import math
    from loop import project as P
    from loop.core import model as M
    from loop.core.data import Obs
    ctx = load_context(REPO / "tasks" / "nanobrain.yaml", state_dir=tmp_path / "s", require_measured=False)
    k, floor = P.noise_floor()
    assert k == 5 and abs(floor["panel"] - ctx.cfg.panel_sd / math.sqrt(2)) < 1e-12
    assert abs(floor["cls"] - ctx.cfg.axis_sd["nb_cls_mean"] / math.sqrt(2)) < 1e-12
    tiny = 1e-8
    assert M.floor_sigma2("panel", tiny, 3) == floor["panel"] ** 2
    assert M.floor_sigma2("panel", tiny, 5) == tiny                         # 5 root seeds: the fit
    assert M.floor_sigma2("panel", 1.0, 3) == 1.0                           # a fit above the floor stands
    nz = M.floor_noise(M.Noise({"cls": tiny, "panel": tiny, "other": tiny}, {"cls": 0.3}), 2)
    assert nz.sigma2["cls"] == floor["cls"] ** 2 and nz.sigma2["other"] == tiny and nz.rho == {"cls": 0.3}
    led = SimpleNamespace(base_root="root", obs=[SimpleNamespace(node="root", seed=s) for s in (1, 1, 7, 9)]
                          + [SimpleNamespace(node="n1", seed=3)])
    assert M.root_seeds(led) == 3
    assert ctx.cfg.disk_floor_gb == 15.0
    np_ctx = load_context(REPO / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path / "np", require_measured=False)
    # 11-10 generalization: the root noise floor is a generic default (5 seeds) -- nanopath gets it too
    assert P.noise_floor()[0] == 5 and np_ctx.cfg.project.noise_floor_seeds == 5 and np_ctx.cfg.disk_floor_gb == 0.0
    P.install(np_ctx.cfg.project.__class__(**{**np_ctx.cfg.project.__dict__, "noise_floor_seeds": 0}))   # floor off
    assert M.floor_sigma2("panel", tiny, 0) == tiny and M.floor_noise(nz, 0) is nz
    with pytest.raises(ValueError):
        P.parse_project({"axes": [{"name": "a", "key": "k"}], "noise_floor_seeds": -1})


def test_a_change_is_banked_once_when_a_later_seed_renames_its_group(nb):
    """11-08 live: w4-pdec-s7338 was banked, then w4-pdec-s1 finished, sorted first and banked the SAME change again."""
    ctx, _store, _ = nb
    g = {"verdict": "PROMISING", "comparable": True, "names": ["w4-pdec-s7338"], "seeds": [7338], "branch": "b",
         "sha": "0f822f2", "config_delta": {"model_kwargs.pool_dec_weight": [None, 0.5]}, "code_numstat": {},
         "delta": {"panel": 0.002, "nb_seg_mean": 0.01}, "z": {"panel": 0.5, "nb_seg_mean": 3.0}}
    assert LR._write_bank(ctx, {"groups": [g], "ts": 1.0}) == 1
    two = dict(g, names=["w4-pdec-s1", "w4-pdec-s7338"], seeds=[1, 7338])
    assert LR._write_bank(ctx, {"groups": [two], "ts": 2.0}) == 0
    p = Path(LR.bank_path(ctx))
    with open(p, "a") as fh:                       # the live file already holds the duplicate: compacted on the next pass
        fh.write(json.dumps({"id": "local-confirm-w4_pdec_s1"}) + "\n")
    assert LR._write_bank(ctx, {"groups": [two], "ts": 3.0}) == 0
    assert [json.loads(x)["id"] for x in p.read_text().splitlines()] == ["local-confirm-w4_pdec_s7338"]


# ------------------------------------------------------------------ 11-09: confirm nodes built from the run commit
def test_put_yaml_value_sets_in_place_or_inserts_under_the_deepest_parent():
    import yaml
    text = "name: r\nmodel_kwargs:\n  depth: 12\n  level_weight: 0.5\n\nbatch_size: 32\n"
    out = LR.put_yaml_value(text, "model_kwargs.depth", 6)
    assert "  depth: 6\n" in out and out.count("\n") == text.count("\n")
    out = LR.put_yaml_value(out, "model_kwargs.level_mirror", [1.0, 1.5, 2.0])
    out = LR.put_yaml_value(out, "nuis_dec.enabled", True)
    out = LR.put_yaml_value(out, "nuis_dec.min_count", 50)
    got = yaml.safe_load(out)
    assert got["model_kwargs"] == {"depth": 6, "level_weight": 0.5, "level_mirror": [1.0, 1.5, 2.0]}
    assert got["nuis_dec"] == {"enabled": True, "min_count": 50} and got["batch_size"] == 32
    assert out.index("level_mirror") < out.index("batch_size")       # inside its parent block, not at the end


class _GitAdapter:
    """file_at / changed_files / commit_files over a real git repo (the nanopath_v2 adapter's plumbing contract)."""
    config_rel = "cfg/r.yaml"

    def __init__(self, repo: Path) -> None:
        self.repo = str(repo)

    def _git(self, *a: str, env: dict | None = None) -> str:
        import os
        import subprocess
        r = subprocess.run(["git", "-C", self.repo, "-c", "user.name=t", "-c", "user.email=t@t", *a],
                           capture_output=True, text=True, env={**os.environ, **(env or {})})
        if r.returncode:
            raise RuntimeError(r.stderr)
        return r.stdout

    def file_at(self, commit: str, rel: str) -> str | None:
        try:
            return self._git("show", f"{commit}:{rel}")
        except RuntimeError:
            return None

    def changed_files(self, a: str, b: str) -> list[str]:
        return [x for x in self._git("diff", "--name-only", a, b).splitlines() if x]

    def commit_files(self, parent: str, files: dict, *, message: str) -> str:
        import tempfile
        idx = tempfile.mktemp()
        env = {"GIT_INDEX_FILE": idx}
        self._git("read-tree", parent, env=env)
        for rel, text in files.items():
            p = Path(tempfile.mktemp())
            p.write_text(text)
            sha = self._git("hash-object", "-w", str(p)).strip()
            self._git("update-index", "--add", "--cacheinfo", f"100644,{sha},{rel}", env=env)
        tree = self._git("write-tree", env=env).strip()
        return self._git("commit-tree", tree, "-p", parent, "-m", message).strip()

    def lock_violations(self, commit: str) -> list[str]:
        return []


@pytest.fixture
def forked(tmp_path):
    """A base B; the root R = B + its config change (the loop's root commit); a run branch S forked from B with code,
    a new module, a non-surface file and the OLD config: 'fb + flags' exactly as the operator's fork branches are."""
    import subprocess
    repo = tmp_path / "repo"
    (repo / "cfg").mkdir(parents=True)
    (repo / "src").mkdir()

    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                              capture_output=True, text=True, check=True).stdout.strip()
    git("init", "-q")
    (repo / "cfg" / "r.yaml").write_text("name: r\nmodel_kwargs:\n  depth: 12\nbatch_size: 32\n")
    (repo / "src" / "model.py").write_text("x = 1\n")
    (repo / "top.yaml").write_text("a: 1\n")
    git("add", "-A")
    git("commit", "-qm", "B")
    b = git("rev-parse", "HEAD")
    (repo / "cfg" / "r.yaml").write_text("name: r\nmodel_kwargs:\n  depth: 12\nbatch_size: 64\n")
    git("commit", "-qam", "R")
    r = git("rev-parse", "HEAD")
    git("checkout", "-q", "-b", "run", b)
    (repo / "src" / "model.py").write_text("x = 2\n")
    (repo / "src" / "nuis.py").write_text("y = 3\n")
    (repo / "top.yaml").write_text("a: 2\n")
    git("add", "-A")
    git("commit", "-qm", "S")
    s = git("rev-parse", "HEAD")
    from loop.contracts import ROOT_NODE_ID, Node, NodeStatus, Strategy
    from loop.store import SqliteStore
    store = SqliteStore(tmp_path / "s.db", "t")
    store.put_node(Node(id=ROOT_NODE_ID, task_id="t", commit=r, parent_id=None, strategy=Strategy.BASELINE,
                        operator=None, family="", mechanism_tag="root", diff_hash="", status=NodeStatus.CONFIRMED))
    lc = SimpleNamespace(enabled=True, confirm_nodes=True, confirm_max_open=1)
    ctx = SimpleNamespace(cfg=SimpleNamespace(local_runs=lc, project=SimpleNamespace(base_sha=r),
                                              families={"architecture": None}),
                          spec=SimpleNamespace(repo=str(repo), surface=SimpleNamespace(
                              writable=("cfg/*.yaml", "src/*.py"), locked=("scripts/",))))
    return ctx, store, _GitAdapter(repo), r, s


def _spec(s: str, name: str, panel: float, knobs: dict) -> dict:
    return {"ids": [f"local-confirm-{name}"], "names": [name], "seeds": [7338], "branch": "run", "sha": s,
            "knobs": knobs, "panel": panel, "integrity": []}


def test_a_promising_manual_run_is_built_as_a_queued_priority_node_from_its_commit(forked):
    import yaml
    from loop.contracts import NodeStatus
    from loop.core import acquire as A
    from loop.core.status import Status
    from loop.select import bank as BK
    ctx, store, ad, r, s = forked
    store.set_state(LR.K_CONFIRMS, {"local-confirm-c1": _spec(s, "c1", 0.008, {"model_kwargs.level_mirror": [1.0, 2.0],
                                                                                "nuis_dec.enabled": True}),
                                    "local-confirm-pd": _spec(s, "pd", 0.006, {"model_kwargs.depth": 6})})
    acts = LR.build_confirms(ctx, store, ad, 100.0)
    built = store.get_state(LR.K_BUILT)
    assert list(built) == ["local-confirm-c1"] and len(acts) == 1     # confirm_max_open 1: the best panel delta first
    n = store.get_node(built["local-confirm-c1"])
    assert n.status is NodeStatus.QUEUED and n.parent_id == "root" and "local_confirm=local-confirm-c1" in n.notes
    assert "skipped=top.yaml" in n.notes                               # not on the writable surface
    assert ad.file_at(n.commit, "src/model.py") == "x = 2\n" and ad.file_at(n.commit, "src/nuis.py") == "y = 3\n"
    cfg = yaml.safe_load(ad.file_at(n.commit, "cfg/r.yaml"))
    assert cfg["batch_size"] == 64                                     # the ROOT's config, not the run commit's
    assert cfg["model_kwargs"] == {"depth": 12, "level_mirror": [1.0, 2.0]} and cfg["nuis_dec"] == {"enabled": True}
    assert ad.file_at(n.commit, "top.yaml") == "a: 1\n"
    assert "local-confirm-c1" in store.get_state(BK.K_BANK_USED)        # no session re-implements it
    led = SimpleNamespace(nodes={n.id: n})
    assert A.program_child(n) and A.program_nodes(led, {n.id: Status.QUEUED}) == {n.id}
    assert A.program_nodes(led, {n.id: Status.SCREENING}) == {n.id}    # its seed extensions keep the priority
    assert LR.build_confirms(ctx, store, ad, 101.0) == []               # one open: the next waits
    store.update_node(n.id, status=NodeStatus.SCREENED_OUT)
    LR.build_confirms(ctx, store, ad, 102.0)
    assert set(store.get_state(LR.K_BUILT)) == {"local-confirm-c1", "local-confirm-pd"}


def test_a_run_that_touches_a_file_the_root_changed_is_refused(forked):
    ctx, store, ad, r, s = forked
    ad.config_rel = "src/model.py"            # pretend the loop config is elsewhere: cfg/r.yaml becomes a code file
    ctx.spec.surface.writable = ("cfg/*.yaml", "src/*.py")
    s2 = ad.commit_files(s, {"cfg/r.yaml": "name: r\nmodel_kwargs:\n  depth: 6\nbatch_size: 32\n"}, message="S2")
    got = LR.build_confirm(ctx, store, ad, "local-confirm-x", _spec(s2, "x", 0.01, {}), 1.0)
    assert got.startswith("-: the root changed cfg/r.yaml")
    # a file only the ROOT changed is not the run's change (merge-base..sha, not root..sha): built, root's copy kept
    assert not LR.build_confirm(ctx, store, ad, "local-confirm-z", _spec(s, "z", 0.01, {}), 1.0).startswith("-")
    dirty = dict(_spec(s, "y", 0.01, {}), integrity=["the worktree was DIRTY at training: the code diff may be incomplete"])
    assert LR.build_confirm(ctx, store, ad, "local-confirm-y", dirty, 1.0).startswith("-: the run's worktree was dirty")


def test_confirm_nodes_off_by_default_for_nanopath(tmp_path):
    ctx = load_context(REPO / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path / "s", require_measured=False)
    assert not LR.confirms(ctx) and LR.build_confirms(ctx, None, None, 0.0) == []
    nb = load_context(REPO / "tasks" / "nanobrain.yaml", state_dir=tmp_path / "n", require_measured=False)
    assert nb.cfg.local_runs.confirm_nodes and nb.cfg.local_runs.confirm_max_open == 2


# ------------------------------------------------------------------ 11-09 review of 182a2ab
def _reroot(ctx, store, commit):
    store.update_node("root", commit=commit)
    ctx.cfg.project.base_sha = commit


def test_a_confirm_node_never_screens_on_its_source_runs_seeds_and_pairs_the_root_on_fresh_ones(forked):
    """The root replicate a root replicate run is the reference a root replicate run was SELECTED against (same-seed correlation ~.6): the
    confirm node screens only on seeds none of its source runs used, and the root gets a control run on each fresh seed
    it lacks (CRN pairing)."""
    from loop import tick as TK
    from loop.contracts import Stage
    from loop.core import data as D
    ctx, store, ad, r, s = forked
    spec = dict(_spec(s, "w4", 0.01, {}), seeds=[1, 7338])
    nid = LR.build_confirm(ctx, store, ad, "local-confirm-w4", spec, 1.0)
    n = store.get_node(nid)
    assert "local_seeds=1,7338" in n.notes and TK.local_seeds(n.notes) == {1, 7338}
    assert "FRESH seeds only" in store.get_hypothesis(n.hypothesis_id).falsification

    def run(node, seed, status="done", stage="baseline"):
        return D.Run(f"{node}:{stage}:{seed}", node, seed, stage, status, 0.0, 0.0, 0.0, "", 0)
    core = TK.ModuleCore()
    core._select = lambda s_, blocks: True
    nodes = {"root": SimpleNamespace(parent=None, notes=""), nid: SimpleNamespace(parent="root", notes=n.notes),
             "plain": SimpleNamespace(parent="root", notes="")}
    runs = [run("root", x) for x in (1, 2026, 7338)]
    data = SimpleNamespace(nodes=nodes, runs=runs)
    cfg = SimpleNamespace(seeds=SimpleNamespace(fresh_base=100000, max_seeds_per_node=5))
    specs = core.screen_specs(nid, "root", 3, data, [], cfg)
    mine = [x.seed for x in specs if x.node == nid]
    assert mine == [2026, 2, 3] and not {1, 7338} & set(mine)
    assert {(x.node, x.stage, x.seed) for x in specs if x.node == "root"} == {("root", Stage.BASELINE, 2),
                                                                              ("root", Stage.BASELINE, 3)}
    runs.append(run("root", 2, "running"))                             # a live control is not launched twice
    assert [x.seed for x in core.screen_specs(nid, "root", 3, data, [], cfg) if x.node == "root"] == [3]
    plain = core.screen_specs("plain", "root", 3, data, [], cfg)        # any other node: unchanged (CRN on the root)
    assert [x.seed for x in plain] == [1, 2026, 7338] and all(x.node == "plain" for x in plain)


def test_an_already_built_confirm_node_gets_its_source_seeds_tagged(forked):
    ctx, store, ad, r, s = forked
    spec = _spec(s, "c1", 0.01, {})
    nid = LR.build_confirm(ctx, store, ad, "local-confirm-c1", spec, 1.0)
    store.update_node(nid, notes=store.get_node(nid).notes.replace(" local_seeds=7338", ""))
    store.set_state(LR.K_CONFIRMS, {"local-confirm-c1": spec})
    store.set_state(LR.K_BUILT, {"local-confirm-c1": nid})
    LR.build_confirms(ctx, store, ad, 2.0)
    assert store.get_node(nid).notes.endswith(" local_seeds=7338")


def test_a_code_re_root_diffs_the_run_from_the_merge_base(forked):
    """After the root gained a file the run never touched, root..sha listed it as a deletion: a false refusal."""
    ctx, store, ad, r, s = forked
    r2 = ad.commit_files(r, {"src/other.py": "z = 1\n"}, message="R2")
    _reroot(ctx, store, r2)
    nid = LR.build_confirm(ctx, store, ad, "local-confirm-c1", _spec(s, "c1", 0.01, {}), 1.0)
    assert not nid.startswith("-"), nid
    n = store.get_node(nid)
    assert ad.file_at(n.commit, "src/other.py") == "z = 1\n" and ad.file_at(n.commit, "src/model.py") == "x = 2\n"


def test_a_refusal_is_retried_when_the_root_changes(forked):
    ctx, store, ad, r, s = forked
    ad.config_rel = "src/model.py"            # cfg/r.yaml becomes a code file both changed: refused
    s2 = ad.commit_files(s, {"cfg/r.yaml": "name: r\nmodel_kwargs:\n  depth: 6\nbatch_size: 32\n"}, message="S2")
    store.set_state(LR.K_CONFIRMS, {"local-confirm-x": _spec(s2, "x", 0.01, {})})
    LR.build_confirms(ctx, store, ad, 1.0)
    assert store.get_state(LR.K_BUILT)["local-confirm-x"].startswith("-: the root changed cfg/r.yaml")
    assert LR.build_confirms(ctx, store, ad, 2.0) == []                # same root: not retried
    r2 = ad.commit_files(r, {"cfg/r.yaml": ad.file_at(s, "cfg/r.yaml")}, message="R2")   # root reverts its cfg edit
    assert ad.file_at(r2, "cfg/r.yaml") == ad.file_at(s, "cfg/r.yaml")
    _reroot(ctx, store, r2)
    acts = LR.build_confirms(ctx, store, ad, 3.0)
    got = store.get_state(LR.K_BUILT)["local-confirm-x"]
    assert len(acts) == 1 and not got.startswith("-"), acts


def test_operator_bank_and_root_audit_nodes_are_kept_from_backlog_drops():
    from loop.core import acquire as A
    from loop.spec import AllocCfg
    assert A.operator_bank_node(SimpleNamespace(notes="brief=explore bank=nb-latent-remap-restore cat=x"))
    assert A.operator_bank_node(SimpleNamespace(notes="brief=explore bank=root_audit-a1 root_audit=1"))
    assert not A.operator_bank_node(SimpleNamespace(notes="brief=explore bank=scout-17 cat=x"))
    assert not A.operator_bank_node(SimpleNamespace(notes="brief=bold lane=explore"))
    assert not AllocCfg().keep_bank_nodes                                # nanopath: unchanged


def test_nanobrain_gates_and_nanopath_defaults(tmp_path):
    nb = load_context(REPO / "tasks" / "nanobrain.yaml", state_dir=tmp_path / "n", require_measured=False)
    np_ = load_context(REPO / "tasks" / "nanopath_v2.yaml", state_dir=tmp_path / "s", require_measured=False)
    assert nb.cfg.ideation.pairwise_center and nb.cfg.alloc.backlog_exclude_program and nb.cfg.alloc.keep_bank_nodes
    assert np_.cfg.ideation.pairwise_center                           # 11-10: a generic default (bias fix)
    assert not (np_.cfg.alloc.backlog_exclude_program or np_.cfg.alloc.keep_bank_nodes)
