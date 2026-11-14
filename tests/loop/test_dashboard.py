"""The HTML dashboard (loop/dashboard.py): renders on an empty and a simulated store, escapes every
stored (LLM-written) text, and the driver writes it atomically.

Run: uv run pytest tests/loop -q --confcutdir=tests/loop
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path

from loop import dashboard
from loop.contracts import (PANEL, AxisPrediction, HandoffNote, HypothesisKind, HypothesisRecord, LeverTags,
                            Node, NodeStatus, Operator, Postmortem, ReviewReport, Strategy)
from loop.spec import DashboardCfg

EVIL = '<script>alert("x")</script><img src=x onerror=alert(1)>'
SECTIONS = ('id="top"', 'id="ideas"', 'id="reviews"', 'id="world"', 'id="ops"')


def _sane(page: str) -> None:
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    assert all(s in page for s in SECTIONS)
    assert page.count("<script") == 2                          # the JSON blob + the page's own inline JS
    assert "http://" not in page and "https://" not in page    # no CDN / network
    assert page.count("</script>") == 2 and "<script src" not in page


def _blob(page: str) -> dict:
    m = re.search(r'<script type="application/json" id="loop-data">(.*?)</script>', page, re.S)
    assert m, "no data blob"
    return json.loads(m.group(1))


def test_renders_on_an_empty_store(ctx, store):
    _sane(dashboard.render(ctx, store, None))
    store.ensure_root("abc123")
    page = dashboard.render(ctx, store, None)
    _sane(page)
    assert "no ideas yet" in page and "no reviews yet" in page and "no world fit yet" in page
    assert _blob(page)["nodes"] == [] and _blob(page)["root"] == "root"


def test_untrusted_text_is_escaped(ctx, store, tmp_path):
    store.ensure_root("abc123")
    diff = tmp_path / "n0001.diff"
    diff.write_text(f"diff --git a/x b/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+{EVIL}\n")
    store.put_hypothesis(HypothesisRecord(
        id="h1", task_id=ctx.task_id, kind=HypothesisKind.TRIAL, claim=EVIL, mechanism_tag="t" + EVIL,
        family="optimization", predictions={PANEL: AxisPrediction(1, 0.002)}, falsification=EVIL,
        levers=LeverTags({"effective_lr": -1})))
    store.put_node(Node(id="n0001", task_id=ctx.task_id, commit="c1", parent_id="root", strategy=Strategy.MOONSHOT,
                        operator=Operator.DRAFT, family="optimization", mechanism_tag="t" + EVIL, diff_hash="h",
                        status=NodeStatus.REJECTED, diff_path=str(diff), hypothesis_id="h1",
                        reject_reason=EVIL, notes=f"request=r1 brief=moonshot{EVIL} backend=opus compound=a+b",
                        created_ts=1.0, updated_ts=1.0))
    store.put_handoff(HandoffNote("ho-n0001", ctx.task_id, "n0001", EVIL, EVIL, EVIL))
    store.put_postmortem(Postmortem("pm1", ctx.task_id, "n0001", "loss", EVIL, EVIL, LeverTags(), (), EVIL))
    store.put_review(ReviewReport(id="rv-1", task_id=ctx.task_id, trigger="first_principles", summary=EVIL,
                                  findings=(EVIL,), meta_hypothesis_ids=(), probe_briefs=((None, EVIL),),
                                  explore_bias={}, n_evals_at=3, selftune=f'[{{"knob": "x", "value": 1, "reason": "{EVIL.replace(chr(34), "")}"}}]',
                                  proposals=(EVIL,)))
    store.log("node_transition", node_id="n0001", frm="queued", to="rejected", reason=EVIL)
    page = dashboard.render(ctx, store, None)
    _sane(page)
    assert "<img" not in page and '<script>alert' not in page
    assert page.count("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;") >= 4   # server-rendered reviews
    n = _blob(page)["nodes"][0]                   # the blob round-trips the raw text; the client uses textContent
    assert n["claim"] == n["fals"] == n["rej"] == EVIL and EVIL in n["diff"] and n["pm"][0][1] == EVIL
    assert n["comp"] == "a+b" and n["arm"] == "moonshot" and n["tr"][-1] == ["rejected", n["tr"][-1][1], EVIL]
    assert "innerHTML" not in page and "insertAdjacentHTML" not in page and "document.write" not in page


def test_blob_cannot_break_out_of_its_script_tag(ctx, store):
    store.ensure_root("abc123")
    evil = "</script><script>alert(1)</script><!-- ]]> \u2028\u2029 &amp;"
    store.put_hypothesis(HypothesisRecord(
        id="h1", task_id=ctx.task_id, kind=HypothesisKind.TRIAL, claim=evil, mechanism_tag="t", family="f",
        predictions={PANEL: AxisPrediction(1, 0.002)}, falsification=evil))
    store.put_node(Node(id="n0001", task_id=ctx.task_id, commit="c1", parent_id="root", strategy=Strategy.EXPLORE,
                        operator=Operator.DRAFT, family="f", mechanism_tag="t", diff_hash="h",
                        status=NodeStatus.QUEUED, hypothesis_id="h1", created_ts=1.0, updated_ts=1.0))
    page = dashboard.render(ctx, store, None)
    _sane(page)
    body = page[page.index('id="loop-data">'):]
    body = body[:body.index("</script>")]
    assert "<" not in body[len('id="loop-data">'):] and "\u2028" not in body.encode().decode("ascii")
    assert _blob(page)["nodes"][0]["claim"] == evil


def test_renders_the_simulated_store_and_driver_writes_atomically(tmp_path, monkeypatch):
    from tests.loop.test_driver import Scripts, _make, _world
    ctx = _world(tmp_path)
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, dashboard=DashboardCfg(every_ticks=2)))
    driver, store, _ = _make(ctx, Scripts())
    seen: list[tuple[str, str, int]] = []
    real = os.replace

    def spy(src: str, dst: str) -> None:                    # the page is complete BEFORE the rename
        seen.append((str(Path(src).parent), Path(src).read_text()[-7:], os.stat(src).st_size))
        real(src, dst)
    monkeypatch.setattr(dashboard.os, "replace", spy)
    driver.run(max_ticks=40)
    out = Path(ctx.state_dir) / "dashboard" / "index.html"
    assert len(seen) == 20                                    # every 2nd of 40 ticks
    assert all(d == str(out.parent) and tail == "</html>" for d, tail, _ in seen)
    assert [p.name for p in out.parent.iterdir()] == ["index.html"]   # no temp file left behind
    driver.write_dashboard()                                  # the last tick may have added nodes
    page = out.read_text()
    _sane(page)
    data = _blob(page)
    assert len(store.nodes()) > 5 and len(data["nodes"]) == len(store.nodes()) - 1
    assert {n["id"] for n in data["nodes"]} == {n.id for n in store.nodes()} - {"root"}
    by = {n.id: n for n in store.nodes()}
    assert all(n["p"] == by[n["id"]].parent_id and n["st"] == by[n["id"]].status.value for n in data["nodes"])
    assert any(n["U"] is not None for n in data["nodes"]) and data["sigma"] is not None
    snap = driver.world.fit(store)
    assert data["incumbent"] == snap.incumbent_id and "Lever trends" in page and 'id="tree"' in page


def _node(ctx, nid, fam, *, parent="root", st=NodeStatus.SCREENED_OUT, strategy=Strategy.EXPLORE, notes="",
          p2=None, tag="t", ts=1.0):
    return Node(id=nid, task_id=ctx.task_id, commit="c", parent_id=parent, strategy=strategy,
                operator=Operator.CROSSOVER if strategy is Strategy.CROSSOVER else None if strategy is Strategy.DESIGN
                else Operator.DRAFT, family=fam,
                mechanism_tag=tag, diff_hash=nid, status=st, second_parent_id=p2,
                notes=f"request=r brief=literature backend=opus {notes}".strip(), created_ts=ts, updated_ts=ts)


def test_root_children_are_grouped_into_family_branches_and_merges_carry_both_parents(ctx, store):
    store.ensure_root("abc123")
    for i in range(8):                                                   # 8 closed -> folded by default
        store.put_node(_node(ctx, f"a{i}", "augmentation", ts=1.0 + i))
    store.put_node(_node(ctx, "o1", "objective", ts=20.0))
    store.put_node(_node(ctx, "o2", "objective", ts=21.0))
    store.put_node(_node(ctx, "o3", "objective", parent="o1", ts=22.0))   # real lineage stays under o1
    for i in range(7):                                                   # 7 but one is active -> open
        store.put_node(_node(ctx, f"r{i}", "regularization", ts=30.0 + i,
                             st=NodeStatus.SCREENING if i == 3 else NodeStatus.SCREENED_OUT))
    store.put_node(_node(ctx, "x1", "objective", strategy=Strategy.CROSSOVER, p2="o2", notes="xover=a3,o2", ts=40.0))
    store.put_node(_node(ctx, "c1", "regularization", notes="compound=augmentation+regularization stack_from=a5",
                         ts=41.0))
    store.put_node(_node(ctx, "m1", "training", strategy=Strategy.MOONSHOT, ts=42.0))
    for i in range(3):
        store.put_node(_node(ctx, f"d1pt{i}", "augmentation", strategy=Strategy.DESIGN, tag=f"d1pt{i}", ts=50.0 + i))
    data = _blob(dashboard.render(ctx, store, None))
    g = {x["name"]: x for x in data["groups"]}
    assert [x["name"] for x in data["groups"]] == ["augmentation", "objective", "regularization", "design",
                                                   "moonshot", "crossover/compound"]
    assert (g["augmentation"]["n"], g["augmentation"]["kids"], g["augmentation"]["open"]) == (8, 8, False)
    assert (g["objective"]["n"], g["objective"]["kids"], g["objective"]["open"]) == (3, 2, True)   # o3 under o1
    assert (g["regularization"]["n"], g["regularization"]["hot"], g["regularization"]["open"]) == (7, True, True)
    assert (g["design"]["n"], g["design"]["kids"]) == (3, 1)                 # one batch node, 3 points
    assert g["moonshot"]["n"] == 1 and g["crossover/compound"]["n"] == 2
    assert data["batches"] == {"batch:d1": {"p": "root", "grp": "design"}}
    by = {n["id"]: n for n in data["nodes"]}
    assert by["o3"]["grp"] is None and by["o3"]["p"] == "o1" and by["o1"]["grp"] == "objective"
    assert by["d1pt0"]["grp"] is None                                      # hangs under its batch
    assert by["x1"]["mp"] == ["a3", "o2"] and by["x1"]["grp"] == "crossover/compound"   # across branches
    assert by["c1"]["mp"] == ["a5"] and by["c1"]["sfi"] is False
    assert by["a0"]["mp"] == [] and by["m1"]["grp"] == "moonshot"


def test_branch_names_and_merge_ids_stay_inert_in_the_blob(ctx, store):
    store.ensure_root("abc123")
    store.put_node(_node(ctx, "n1", EVIL))
    store.put_node(_node(ctx, "n2", "f", strategy=Strategy.CROSSOVER, notes=f"xover=n1,{EVIL.replace(' ', '')}"))
    page = dashboard.render(ctx, store, None)
    _sane(page)
    body = page[page.index('id="loop-data">') + len('id="loop-data">'):]
    assert "<" not in body[:body.index("</script>")]
    data = _blob(page)
    assert EVIL in [x["name"] for x in data["groups"]]
    assert {n["id"]: n["mp"] for n in data["nodes"]}["n2"] == ["n1"]       # unknown ids are dropped


def test_a_bank_node_card_shows_its_bank_id_and_source(ctx, store, tmp_path):
    bank = tmp_path / "bank.jsonl"
    bank.write_text(json.dumps({"id": "j1", "title": "t", "idea": "multi-block predictor", "source": "nanopath-v1"}) + "\n")
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, idea_banks=(str(bank),)))
    store.ensure_root("abc123")
    store.put_node(_node(ctx, "b1", "objective", notes="request=req-1 brief=bank bank=j1", ts=1.0))
    store.put_node(_node(ctx, "o1", "objective", ts=2.0))
    page = dashboard.render(ctx, store, None)
    by = {n["id"]: n for n in _blob(page)["nodes"]}
    assert by["b1"]["bank"] == "j1 (nanopath-v1)" and by["o1"]["bank"] == ""
    assert "' · bank '" in page


def test_arm_shares_count_bandit_draws_only():
    """Shares of research_started by arm: design/tune rows, deterministic debugs, winner crosses and restart
    re-issues are not bandit draws; 24 h and all-time."""
    now = 10 * dashboard.DAY
    ev = [{"event": "research_started", "strategy": "moonshot", "ts": now - 60},
          {"event": "research_started", "strategy": "explore", "ts": now - 60},
          {"event": "research_started", "strategy": "explore", "ts": now - 3 * dashboard.DAY},
          {"event": "research_started", "strategy": "design", "ts": now - 60},
          {"event": "research_started", "strategy": "revisit", "ts": now - 60, "deterministic_debug": True},
          {"event": "research_started", "strategy": "crossover", "ts": now - 60, "winner_cross": True},
          {"event": "research_started", "strategy": "moonshot", "ts": now - 60, "restarted": True},
          {"event": "intake", "strategy": "exploit", "ts": now - 60}]
    assert dashboard.arm_shares(ev, now) == "24h moonshot 50%, explore 50% (n=2); all-time explore 67%, moonshot 33% (n=3)"
    assert dashboard.arm_shares([], now) == "24h none; all-time none"
