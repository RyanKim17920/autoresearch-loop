"""v3-c9 C1: the researcher's `history` tool (loop.agents.history) -- a per-node trajectory dashboard computed from the
store and the VISIBLE ledger, table / open / clusters actions, a seeded partial column view per session, read-only,
no official-panel or held-out number, registered for every v3 brief kind (bold included)."""

from __future__ import annotations

import dataclasses
import re
import sqlite3
from dataclasses import replace

import pytest

from loop.agents import history as HX
from loop.agents import researcher as RS
from loop.contracts import ROOT_NODE_ID, CandidateRequest, Operator, Strategy, Worktree
from loop.core import acquire as A
from loop.core import data as D
from loop.core import ideation as ID
from loop.core.seeds import reconstruct
from loop.spec import IdeationCfg
from loop.store import SqliteStore

from .core.conftest import TASK
from .test_tick import _ledger


_REAL_WARM = HX.warm


def _with(ctx, **ideation):
    base = ctx.cfg.ideation or IdeationCfg()
    return dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=dataclasses.replace(base, **ideation)))


def _view(db, ctx, mutate=None):
    store = SqliteStore(db, TASK)
    led = D.load(db, TASK)
    if mutate is not None:
        led = mutate(led)
    post = A.Posterior(led, reconstruct(led))
    return store, ID.Ideation(ctx, store).view(led, post)


@pytest.fixture
def henv(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(HX, "warm", lambda *a, **k: None)             # no background thread in the unit tests
    HX._DIFF.clear()
    ctx = _with(dataclasses.replace(ctx, task_id=TASK, state_dir=str(tmp_path / "state")), history_calls_max=0)
    store, v = _view(_ledger(tmp_path / "ledger.db"), ctx)
    HX.forget()
    HX.remember(ctx, store, v, None)
    yield ctx, store, v
    HX.forget()
    store.close()


def _hist(ctx, store, cols=HX.OPT_COLS):
    return HX.Historian(ctx, store, cols, request_id="req-v3-00001")


def _rows(text):
    """The table's data rows as dicts keyed by the header (the header is the line starting with `id |`)."""
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("id |"))
    head = [h.strip() for h in lines[i].split("|")]
    return [dict(zip(head, [c.strip() for c in l.split(" | ")])) for l in lines[i + 1:] if l and not l.startswith("(")]


# ---------------------------------------------------------------------------------------------- the table
def test_every_ledger_node_is_a_row_with_its_own_increment_and_parent(henv):
    ctx, store, v = henv
    tab = HX.current_table()
    assert {r["id"] for r in tab.rows} == set(v.led.nodes)
    got = next(n for n, b in v.beliefs.items() if b.n >= 3)
    r = tab.by_id[got]
    assert r["m"] == pytest.approx(v.beliefs[got].m) and r["n"] == v.beliefs[got].n
    assert r["sd"] == pytest.approx(v.beliefs[got].v ** 0.5) and r["parent"] == v.led.nodes[got].parent
    assert r["kind"] == "literature" and r["tag"].count("/") == 1 and r["desc"]
    assert tab.by_id[ROOT_NODE_ID]["m"] is None
    assert HX.current_table() is tab                                   # cached per view token


def test_table_text_has_the_key_columns_and_caps_rows_at_forty(henv):
    ctx, store, v = henv
    out = _hist(ctx, store)("table", {"kind": "all"}, "m", 500)
    rows = _rows(out)
    assert len(rows) == 40 < len(v.led.nodes) + 1
    assert {"id", "m", "n", "kind"} <= set(rows[0])
    assert out.splitlines()[0].startswith("history (visible ledger; incumbent ")
    lim = _rows(_hist(ctx, store)("table", {}, "m", 5))
    assert len(lim) == 5
    capped = _rows(_hist(_with(ctx, history_rows_max=7), store)("table", {"kind": "all"}, "m", 40))
    assert len(capped) == 7


def test_filters_and_sorts(henv):
    ctx, store, v = henv
    H = _hist(ctx, store)
    lit = _rows(H("table", {"kind": "literature", "min_n": 0}, "m", 40))
    assert lit and all(r["kind"].startswith("literature") for r in lit)
    ms = [float(r["m"]) for r in lit if r["m"] != "-"]
    assert ms == sorted(ms, reverse=True)                              # m descending when asked
    asc = [float(r["m"]) for r in _rows(H("table", {"kind": "literature", "min_n": 0}, "-m", 40)) if r["m"] != "-"]
    assert asc == sorted(asc)
    band = _rows(H("table", {"kind": "literature", "min_m": -0.001, "max_m": 0.001}, "m", 40))
    assert band and all(-0.001 <= float(r["m"]) <= 0.001 for r in band)
    sub = _rows(H("table", {"kind": "all", "mechanism": "MECH_1"}, "m", 40))          # case-insensitive substring
    assert sub and all("mech_1" in (r["tag"] + r["desc"]).lower() for r in sub)
    failed = _rows(H("table", {"kind": "all", "failed": True}, "recent", 40))
    assert failed and all(r["fail"] != "-" for r in failed)
    ok = _rows(H("table", {"kind": "all", "failed": False}, "m", 40))
    assert ok and all(r["fail"] == "-" for r in ok)
    assert "unknown sort" in H("table", {}, "bogus", 5) and "unknown axis" in H("table", {"axis": "zzz"}, "m", 5)


def test_axis_filter_shows_that_axis_and_sorts_on_it(henv):
    ctx, store, v = henv
    H = _hist(ctx, store)
    out = H("table", {"axis": "cls", "kind": "all"}, "m", 30)
    rows = _rows(out)
    assert rows and "d_cls" in rows[0] and all(r["d_cls"] != "-" for r in rows)
    full = H("table", {"axis": "cls", "kind": "all", "min_n": 0}, "cls", 30)
    assert "WARNING: ranked by cls delta/se" in full                   # an axis sort carries the multiplicity warning
    byaxis = _rows(full)
    z = [float(r["d_cls"].split("+-")[0]) / float(r["d_cls"].split("+-")[1]) for r in byaxis
         if float(r["d_cls"].split("+-")[1]) > 0]
    assert all(a >= b - 0.05 for a, b in zip(z, z[1:]))                # ranked by delta / se (display rounding)
    other = _rows(H("table", {"kind": "all"}, "slide", 10))
    assert other                                                     # an axis name is a sort key too


def test_cluster_and_epoch_filters(henv):
    ctx, store, v = henv
    H = _hist(ctx, store)
    tab = HX.current_table()
    cl = next(c for c in tab.clusters if len(c.members) >= 2)
    rows = _rows(H("table", {"cluster": cl.id, "kind": "all"}, "recent", 40))
    assert {r["id"] for r in rows} == set(cl.members)
    cur = _rows(H("table", {"epoch": "current", "kind": "all"}, "recent", 40))
    assert cur and all(r["epoch"].endswith("*") for r in cur)
    assert "clusters" in H("clusters", limit=3).splitlines()[-1] and cl.id in H("clusters", limit=40)


def test_failure_signatures_name_the_reject_void_veto_and_early_kill(henv):
    from loop.contracts import NodeStatus
    ctx, store, v = henv
    ids = [n for n in v.led.nodes if n != ROOT_NODE_ID][:4]
    store.update_node(ids[0], status=NodeStatus.REJECTED, reject_reason="near_dup_mechanism: x")
    store.update_node(ids[1], status=NodeStatus.VOID)
    store.log("node_transition", node_id=ids[1], frm="screening", to="void", reason="infra_retry_cap")
    store.update_node(ids[2], status=NodeStatus.VETOED)
    store.log("node_transition", node_id=ids[2], frm="screening", to="vetoed", reason="harm veto on ['rob']")
    store.set_state("alloc.early_kill", {ids[3]: {"z": -2.0}})
    HX.forget()
    HX.remember(ctx, store, v, None)
    tab = HX.current_table()
    assert tab.by_id[ids[0]]["fail"] == "rejected: near_dup_mechanism"
    assert tab.by_id[ids[1]]["fail"] == "void: infra_retry_cap"
    assert tab.by_id[ids[2]]["fail"].startswith("harm-guard veto: harm veto on")
    assert tab.by_id[ids[3]]["fail"].endswith("early kill")


# ---------------------------------------------------------------------------------------------- open
def test_open_each_part_and_truncate_the_diff(henv, tmp_path):
    from loop.contracts import HandoffNote, Postmortem
    ctx, store, v = henv
    nid = next(n for n, b in v.beliefs.items() if b.n >= 3)
    diff = tmp_path / "d.diff"
    diff.write_text("\n".join(f"+line {i} of the change" for i in range(2000)))
    store.update_node(nid, diff_path=str(diff))
    store.set_state(f"config_diff:{nid}", {"train": {"lr": 0.001}})
    store.put_handoff(HandoffNote("ho-x", store.task_id, nid, "did a thing", "learned a thing", "worried", 1.0))
    store.put_postmortem(Postmortem("pm-x", store.task_id, nid, "null", "no effect", "the mechanism was inert",
                                    store.get_node(nid).levers, suggested_next="try more", created_ts=1.0))
    H = _with(ctx, history_open_chars=1500) and _hist(_with(ctx, history_open_chars=1500), store)
    d = H("open", node=nid, part="diff")
    assert d.startswith(f"{nid} [") and "--- diff ---" in d and "truncated" in d and len(d) < 2300
    page = H("open", node=nid, part="diff", offset=1500)
    assert "[from character 1500]" in page and "line " in page
    assert "train.lr: 0.001" in H("open", node=nid, part="config")
    h = H("open", node=nid, part="handoff")
    assert "did a thing" in h and "learned a thing" in h and "worried" in h
    pm = H("open", node=nid, part="postmortem")
    assert "no effect" in pm and "try more" in pm
    assert "no node 'n99999'" in H("open", node="n99999", part="diff")
    assert "part must be one of" in H("open", node=nid, part="whatever")
    cl = HX.current_table().by_id[nid]["cluster"]
    if cl:
        assert H("open", node=cl).startswith(f"cluster {cl} ")


def test_per_seed_is_the_visible_composite_only(henv):
    ctx, store, v = henv
    nid = next(n for n, b in v.beliefs.items() if b.n >= 3)
    out = _hist(ctx, store)("open", node=nid, part="per_seed")
    assert out.splitlines()[2].startswith("seed | pool | node composite | parent composite")
    rows = [l.split(" | ") for l in out.splitlines()[3:] if l[:1].isdigit()]
    assert rows and all(len(r) == 5 for r in rows)
    mine = v.led.by_node()[nid]
    seed = int(rows[0][0])
    from loop.core import model as M
    assert float(rows[0][2]) == pytest.approx(M.seed_means(mine, M.COMPOSITE, v.weights)[seed], abs=5e-5)


# ---------------------------------------------------------------------------------------------- blind holdout
def test_no_official_panel_number_reaches_any_output(ctx, tmp_path, monkeypatch):
    """The official `panel` of a visible ledger's observations is the full locked panel (held-out tasks included): the
    dashboard reads axes / the visible composite only. Every observation's panel is a sentinel; it must appear nowhere."""
    monkeypatch.setattr(HX, "warm", lambda *a, **k: None)
    HX._DIFF.clear()
    ctx = _with(dataclasses.replace(ctx, task_id=TASK, state_dir=str(tmp_path / "state")))
    sentinel = 0.987654
    store, v = _view(_ledger(tmp_path / "l.db"), ctx,
                     lambda led: replace(led, obs=tuple(replace(o, panel=sentinel) for o in led.obs)))
    HX.forget()
    HX.remember(ctx, store, v, None)
    H = _hist(ctx, store)
    nid = next(n for n, b in v.beliefs.items() if b.n >= 3)
    texts = [H("table", {"kind": "all"}, "m", 40), H("table", {"kind": "all", "axis": "cls"}, "gain", 40),
             H("table", {"kind": "all"}, "novelty", 40), H("clusters", limit=40), H("open", node=nid, part="per_seed"),
             H("open", node=nid, part="handoff"), H("open", node=nid, part="config")]
    blob = "\n".join(texts)
    # not "0.98": the novelty column (a diff property, set-1 #6: 1 - exp(-points / 5)) can legitimately read 0.98
    assert "0.9876" not in blob and "987654" not in blob and "0.988" not in blob and "0.99 " not in blob
    assert all(r["verdict"] in ("-", "pass", "fail") or r["verdict"].isalpha() for r in _rows(texts[0]))
    HX.forget()
    store.close()


def test_the_test_verdict_is_a_kind_never_a_number(henv):
    ctx, store, v = henv
    tab = HX.current_table()
    kinds = {r["verdict"] for r in tab.rows if r["verdict"]}
    assert kinds and all(re.fullmatch(r"[a-z_]+", k) for k in kinds)


# ---------------------------------------------------------------------------------------------- read-only
def _counts(db):
    con = sqlite3.connect(db)
    try:
        return {t: con.execute(f"SELECT COUNT(*), COALESCE(SUM(LENGTH(CAST(value AS BLOB))), 0) FROM {t}").fetchone()
                if t == "state" else con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
                for t in ("nodes", "runs", "log", "state", "handoffs", "postmortems")}
    finally:
        con.close()


def test_the_tool_never_writes(henv, tmp_path):
    ctx, store, v = henv
    db = tmp_path / "ledger.db"
    files = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    before = _counts(db)
    H = _hist(ctx, store)
    nid = next(n for n, b in v.beliefs.items() if b.n >= 3)
    for args in ({"action": "table"}, {"action": "table", "filter": {"axis": "cls"}, "sort": "cls"},
                 {"action": "clusters"}, {"action": "open", "node": nid, "part": "diff"},
                 {"action": "open", "node": nid, "part": "per_seed"}, {"action": "open", "node": nid, "part": "postmortem"}):
        H(**args)
    assert _counts(db) == before
    assert sorted(p for p in tmp_path.rglob("*") if p.is_file()) == files


# ---------------------------------------------------------------------------------------------- the partial view
def test_columns_are_a_seeded_subset_that_differs_across_sessions_and_branches():
    a = HX.pick_columns("req-v3-00001|", 0.6, recent=())
    assert a == HX.pick_columns("req-v3-00001|", 0.6, recent=())               # seeded: reproducible
    assert len(a) == round(0.6 * len(HX.OPT_COLS)) and set(a) <= set(HX.OPT_COLS)
    assert list(a) == [c for c in HX.OPT_COLS if c in a]                         # in the table's own order
    subs = {HX.pick_columns(f"req-v3-{i:05d}|{b}", 0.6, recent=()) for i in range(20) for b in ("", "3.1", "3.2")}
    assert len(subs) > 40                                                      # sessions / branches differ
    assert HX.pick_columns("x", 1.0) == HX.OPT_COLS and HX.pick_columns("x", 0.0) == ()
    assert HX.pick_columns("x|3.1", 0.6, recent=()) != HX.pick_columns("x|3.2", 0.6, recent=())


def test_concurrent_sessions_get_distinct_subsets(ctx):
    HX._RECENT.clear()
    got = [HX.session_columns(ctx, f"req-v3-{i:05d}", "") for i in range(12)]
    assert len(set(got)) == 12                                                 # a repeat of a recent subset is redrawn
    assert HX.session_columns(ctx, "req-v3-00099", "4.2") not in got[-8:]


def test_a_session_sees_only_its_subset_plus_the_key_columns(henv):
    ctx, store, v = henv
    H = HX.Historian(ctx, store, ("tag", "gain"), request_id="req-v3-00002")
    rows = _rows(H("table", {"kind": "literature"}, "m", 5))
    assert set(rows[0]) == {"id", "m", "sd", "n", "epoch", "kind", "tag", "gain"}
    none = _rows(HX.Historian(ctx, store, (), request_id="r")("table", {"kind": "literature"}, "m", 3))
    assert set(none[0]) == {"id", "m", "sd", "n", "epoch", "kind"}


def test_no_view_yet_is_a_clear_answer(ctx):
    HX.forget()
    assert HX.Historian(ctx, None)("table").startswith("history: no view yet")


# ---------------------------------------------------------------------------------------------- diff statistics
def test_knob_flag_and_ast_novelty_come_from_the_diff(henv, tmp_path):
    ctx, store, v = henv
    ids = [n for n in v.led.nodes if n != ROOT_NODE_ID][:2]
    knob = tmp_path / "k.diff"
    knob.write_text("--- a/config.yaml\n+++ b/config.yaml\n@@ -3 +3 @@\n-  lr: 1.0e-4\n+  lr: 2.0e-4\n")
    struct = tmp_path / "s.diff"
    struct.write_text("--- a/train.py\n+++ b/train.py\n@@ -40 +40,3 @@\n+def neighbour_loss(z, zn):\n"
                      "+    return (z - zn).pow(2).mean()\n")
    store.update_node(ids[0], diff_path=str(knob))
    store.update_node(ids[1], diff_path=str(struct))
    HX._DIFF.clear()
    a, b = HX.diff_stats(store.get_node(ids[0])), HX.diff_stats(store.get_node(ids[1]))
    assert a == (True, 0.0) and b[0] is False and b[1] > 0
    assert HX.diff_stats(replace(store.get_node(ids[0]), id="zzz", diff_path="/nonexistent")) is None
    out = _hist(ctx, store)("table", {"kind": "all"}, "novelty", 40)
    rows = {r["id"]: r for r in _rows(out)}
    if ids[1] in rows:
        assert rows[ids[1]]["knob"] == "struct" and float(rows[ids[1]]["novelty"]) > 0


# ---------------------------------------------------------------------------------------------- the tool wiring
def test_history_is_registered_for_v3_sessions_and_dispatched(tmp_path):
    assert "history" in [t["name"] for t in RS.V3_TOOL_DEFS] and "diagnose" in [t["name"] for t in RS.V3_TOOL_DEFS]
    t = RS.WorktreeTools(None, Worktree(str(tmp_path), "b", "c0"), writable=("*",), locked=(), checker=lambda **k: "OK")
    assert t("history", {"action": "table"}).startswith("ERROR: history is not available")
    t.historian = lambda *a: "H:" + repr(a)
    assert t("history", {"action": "open", "node": "n1", "part": "diff"}) == "H:('open', None, '', 0, 'n1', 'diff', 0)"
    assert t("history", {"action": "bogus_arg", "zz": 1}).startswith("ERROR: bad arguments for history")


@pytest.mark.parametrize("kind", ["explore", "literature", "repair", "program", "kuhn", "readout", "bold"])
def test_every_brief_kind_gets_the_history_tool_and_logs_its_columns(kind):
    from tests.loop.test_agents import DirAdapter, FakeLLMClient, MemStore, _ctx, _snapshot, CONFIG_TEXT
    import tempfile
    from pathlib import Path
    from loop.agents.llm import AgentRun
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tpl = tmp / "tpl"
        tpl.mkdir()
        (tpl / "fake_model.py").write_text("LR = 1e-3\n")
        (tpl / "fake_config.yaml").write_text(CONFIG_TEXT)
        (tpl / "fake_probe.py").write_text("SCORE = 1\n")
        (tmp / "wt").mkdir()
        adapter, store = DirAdapter(tpl, tmp / "wt"), MemStore()
        llm = FakeLLMClient({"researcher": [AgentRun(ok=False, finish=None, turns=0, wall_s=0.0, error="x")]})
        req = CandidateRequest(id="req-v3-00007", task_id=_ctx().task_id, strategy=Strategy.EXPLORE,
                               operator=Operator.DRAFT, island="main", parent_ids=(ROOT_NODE_ID,), brief="b",
                               brief_kind=kind, branch="2.3")
        RS.LLMResearcher(_ctx(), llm, adapter).research(req, store, _snapshot())
        assert "history" in llm.calls[0]["tools"]
        assert ("propose_slate" in llm.calls[0]["tools"]) == (kind in ("explore", "literature", "bold"))     # the repair portfolio is opt-in
        ev = [f for e, f in store.logged if e == "history_view"]
        assert len(ev) == 1 and ev[0]["request_id"] == "req-v3-00007" and ev[0]["branch"] == "2.3"
        assert ev[0]["kind"] == kind and set(ev[0]["columns"]) <= set(HX.OPT_COLS) and ev[0]["columns"]


def test_history_is_off_when_the_config_says_so():
    from tests.loop.test_agents import DirAdapter, FakeLLMClient, MemStore, _ctx, _snapshot, CONFIG_TEXT
    import tempfile
    from pathlib import Path
    from loop.agents.llm import AgentRun
    ctx = _ctx()
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, ideation=IdeationCfg(history=False)))
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "tpl").mkdir()
        (tmp / "tpl" / "fake_model.py").write_text("LR = 1e-3\n")
        (tmp / "tpl" / "fake_config.yaml").write_text(CONFIG_TEXT)
        (tmp / "tpl" / "fake_probe.py").write_text("SCORE = 1\n")
        (tmp / "wt").mkdir()
        adapter, store = DirAdapter(tmp / "tpl", tmp / "wt"), MemStore()
        llm = FakeLLMClient({"researcher": [AgentRun(ok=False, finish=None, turns=0, wall_s=0.0, error="x")]})
        req = CandidateRequest(id="req-v3-00008", task_id=ctx.task_id, strategy=Strategy.EXPLORE,
                               operator=Operator.DRAFT, island="main", parent_ids=(ROOT_NODE_ID,), brief="b",
                               brief_kind="explore")
        t = RS.LLMResearcher(ctx, llm, adapter)
        t.research(req, store, _snapshot())
        assert not [1 for e, _ in store.logged if e == "history_view"]


def test_the_ideation_tick_hands_its_view_to_the_tool(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(HX, "warm", lambda *a, **k: None)
    ctx = _with(dataclasses.replace(ctx, task_id=TASK, state_dir=str(tmp_path / "state")), history_calls_max=0)
    store, v = _view(_ledger(tmp_path / "l.db"), ctx)
    HX.forget()
    ide = ID.Ideation(ctx, store)
    seen = []
    monkeypatch.setattr(HX, "remember", lambda c, s, view, st=None: seen.append((view.incumbent, st)))

    class T:                       # a minimal tick: ideate returns before any session starts
        dry = True
        post = v.post
        research = type("R", (), {"researcher": None, "free": lambda s: 0, "capacity": lambda s: 0})()
        pool = None
        phase_s: dict = {}
    try:
        ide.ideate(T(), [], v.led, {})
    except Exception:  # noqa: BLE001 - only the hand-over matters here
        pass
    assert seen and seen[0][0] == v.incumbent
    store.close()


def test_the_warm_up_thread_analyses_every_diff_once_and_only_when_the_ledger_grew(henv, tmp_path, monkeypatch):
    ctx, store, v = henv
    ids = [n for n in v.led.nodes if n != ROOT_NODE_ID][:3]
    for i, nid in enumerate(ids):
        d = tmp_path / f"w{i}.diff"
        d.write_text("--- a/config.yaml\n+++ b/config.yaml\n@@ -3 +3 @@\n-  lr: 1.0e-4\n+  lr: 2.0e-4\n")
        store.update_node(nid, diff_path=str(d))
    HX._DIFF.clear()
    HX._WARM.update(thread=None, n=None)
    _REAL_WARM(ctx, store, v)
    th = HX._WARM["thread"]
    assert th is not None
    th.join(10)
    assert all(nid in HX._DIFF and HX._DIFF[nid] == (True, 0.0) for nid in ids)
    calls = []
    monkeypatch.setattr(store, "nodes", lambda *a, **k: calls.append(1) or [])
    _REAL_WARM(ctx, store, v)                                       # same ledger size: no re-scan of the store
    assert not calls


# ---------------------------------------------------------------------------------------------- review fixes S1 / S3 / S4
def test_the_default_table_shows_epoch_sd_n_filters_to_the_current_epoch_and_sorts_by_the_shrunk_score(henv):
    ctx, store, v = henv
    H = _hist(ctx, store, ())
    tab = HX.current_table()
    old = next(r for r in tab.rows if r["kind"].startswith("literature") and r["n"] >= 3)
    keep = old["epoch"]
    out = H("table", {}, "", 40)
    assert next(l for l in out.splitlines() if l.startswith("id |")).split(" | ")[:6] == [
        "id", "m", "sd", "n", "epoch", "kind"]                                                 # the key columns
    assert "rows of OLDER base epochs are excluded" in out and "epoch: 'all'" in out
    assert "sorted by shrunk (m - 1.28 sd)" in out
    old["epoch"] = "e0"
    try:
        assert old["id"] not in {r["id"] for r in _rows(H("table", {"kind": "all"}, "", 40))}
        assert old["id"] in {r["id"] for r in _rows(H("table", {"kind": "all", "epoch": "all", "min_n": 0}, "-id", 40))}
        assert "older base epochs are included" in H("table", {"kind": "all", "epoch": "all"}, "", 5)
    finally:
        old["epoch"] = keep
    snodes = {x.id: x for x in store.nodes()}
    rows, _ = HX.select(tab, {"kind": "all"}, "", 1000, snodes)
    solid = [r for r in rows if r["n"] >= 3]
    thin = [r for r in rows if r["n"] < 3]
    assert solid and thin and rows == solid + thin                                              # n<3 rows sort last
    sc = [r["m"] - 1.28 * r["sd"] for r in solid if r["m"] is not None]
    assert sc == sorted(sc, reverse=True)                                                       # the shrunk score
    assert any("n<3" in r["n"] for r in _rows(H("table", {"kind": "all", "max_m": 0.0005}, "", 40)))
    all_n = _rows(H("table", {"kind": "all", "min_n": 0}, "", 40))
    assert not any("n<" in r["n"] for r in all_n)


def test_history_calls_are_capped_per_session(henv):
    ctx, store, v = henv
    c = _with(ctx, history_calls_max=2)
    H = _hist(c, store)
    assert H("table", {}, "", 3).startswith("history (") and H("clusters").startswith("cluster")
    assert "budget of 2 calls is spent" in H("table", {}, "", 3)


def test_the_diff_cache_is_a_bounded_lru(henv, tmp_path, monkeypatch):
    ctx, store, v = henv
    monkeypatch.setattr(HX, "DIFF_CACHE_MAX", 3)
    HX._DIFF.clear()
    ids = [n for n in v.led.nodes if n != ROOT_NODE_ID][:5]
    for i, nid in enumerate(ids):
        d = tmp_path / f"{i}.diff"
        d.write_text(f"--- a/config.yaml\n+++ b/config.yaml\n@@ -3 +3 @@\n-  lr: 1.0e-4\n+  lr: {i + 2}.0e-4\n")
        store.update_node(nid, diff_path=str(d))
        HX.diff_stats(store.get_node(nid))
    assert list(HX._DIFF) == ids[-3:]
    HX.diff_stats(store.get_node(ids[-3]))                                      # a hit refreshes it
    assert list(HX._DIFF)[-1] == ids[-3]


def test_the_table_is_built_outside_the_global_lock(henv, monkeypatch):
    import threading
    ctx, store, v = henv
    HX.forget()
    HX.remember(ctx, store, v, None)
    got = {}
    real = HX.build

    def probe(*a, **k):
        def other():
            got["free"] = HX._LOCK.acquire(timeout=2)
            if got["free"]:
                HX._LOCK.release()
        t = threading.Thread(target=other)
        t.start()
        t.join()
        return real(*a, **k)
    monkeypatch.setattr(HX, "build", probe)
    assert HX.current_table() is not None and got["free"] is True               # remember() / forget() never wait for a build
    assert HX.current_table() is HX.current_table()                             # double-checked cache


# ---------------------------------------------------------------------------------------------- review fix M4: the leak guard
HELD = ["cls:spider_colorectal", "seg:pannuke", "surv:leopard_bcr", "rob:tolkach_esca"]
LEAKS = ("0.8123", "0.0123", "0.7771", "0.6432", "0.5519", "0.9912", "0.4417")


def _plant(ctx, store, v, tmp_path):
    from loop.contracts import HandoffNote, HypothesisKind, HypothesisRecord, Postmortem
    store.set_state("holdout.tasks", HELD)
    nid = next(n for n, b in v.beliefs.items() if b.n >= 3)
    store.put_hypothesis(HypothesisRecord(
        "hy-leak", store.task_id, HypothesisKind.TRIAL,
        "gates the readout; on spider_colorectal it reached auc 0.8123 and the panel delta +0.0123 over the parent",
        "mt", "fam", {}, "falsified if pannuke dice stays below 0.7771"))
    store.update_node(nid, hypothesis_id="hy-leak", diff_path=str(tmp_path / "leak.diff"))
    (tmp_path / "leak.diff").write_text("+# tolkach_esca cindex 0.6432 official panel 0.5519\n+" + "y" * 80 + "\n+x = 1\n")
    store.put_handoff(HandoffNote("ho-leak", store.task_id, nid,
                                  "ran leopard_bcr: c-index 0.6432 and tolkach_esca auc 0.5519",
                                  "labless composite 0.9912 vs 0.4417; panel delta +0.0123",
                                  "pannuke 0.7771 looks high"))
    store.put_postmortem(Postmortem(
        "pm-leak", store.task_id, nid, "null", "official panel delta +0.0123 (spider_colorectal 0.8123)",
        "the head lifts pannuke by 0.7771\nCrash log tail:\nTraceback (most recent call last):\n  File x.py\n"
        "RuntimeError: auc 0.9912 tolkach_esca\n", store.get_node(nid).levers,
        suggested_next="re-run; seg:pannuke was 0.7771, labless composite 0.9912", created_ts=1.0))
    store.set_state(f"config_diff:{nid}", {"probe": {"note": "spider_colorectal 0.8123"}})
    HX.forget()
    HX.remember(ctx, store, v, None)
    return nid


def test_no_held_task_number_or_panel_delta_survives_open_or_the_table(henv, tmp_path):
    ctx, store, v = henv
    nid = _plant(ctx, store, v, tmp_path)
    H = _hist(_with(ctx, history_open_chars=20000), store)
    outs = {p: H("open", node=nid, part=p) for p in HX.PARTS}
    outs["table"] = H("table", {"kind": "all", "epoch": "all", "min_n": 0}, "recent", 40)
    outs["table_axis"] = H("table", {"axis": "cls", "kind": "all", "epoch": "all", "min_n": 0}, "cls", 40)
    outs["clusters"] = H("clusters")
    for part, text in outs.items():
        for leak in LEAKS:
            assert leak not in text, (part, leak)
        assert "+0.0123" not in text and "panel delta +" not in text, part
        for name in ("spider_colorectal", "leopard_bcr", "tolkach_esca", "pannuke"):
            assert name not in text, (part, name)
    pm = outs["postmortem"]
    assert "Traceback" not in pm and "RuntimeError" not in pm and "Crash log" not in pm      # crash-log tails dropped
    assert "suggested next" in pm and "explanation" in pm                                    # the rest stays
    assert "gates the readout" in outs["handoff"] and "ran [held-out task]" in outs["handoff"]
    assert "x = 1" in outs["diff"] and "train.lr" not in outs["config"]


def test_the_sanitizer_keeps_ordinary_numbers_and_ids():
    t = HX.sanitize("node n0655@065b87f trained 12 epochs at lr 0.0003 (seed 3). " + "." * 60 + " panel delta +0.0123", HELD)
    assert "n0655@065b87f" in t and "12 epochs" in t and "0.0003" in t and "+0.0123" not in t
    far = "x" * 60
    assert "0.5" in HX.sanitize(f"auc{far}0.5", HELD)                       # more than 40 characters away: kept
    assert "0.5" not in HX.sanitize(f"auc 0.5 {far}", HELD)
    assert HX.sanitize("tolkach-esca 0.61, TOLKACH ESCA 0.62, tolkachesca 0.63", HELD).count("[n]") == 3


def test_the_sanitizer_is_idempotent_and_leaves_no_number_in_range_of_a_context_word():
    t = "panel " + " ".join(f"{0.1 * i:.4f}" for i in range(1, 16)) + " done"
    once = HX.sanitize(t, HELD)
    assert HX.sanitize(once, HELD) == once
    for m in HX._NUM.finditer(once):                                  # whatever is left is beyond 40 characters of "panel"
        assert m.start() - len("panel") > HX.NUM_NEAR
