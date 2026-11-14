"""loop.core.bar: the external bar = Labless's acceptance rule (P(median of 3 fresh seeds >= leader median + margin)
>= promote_p); the verdict on fresh TEST seeds; the heartbeat line; the meta-report section; IncumbentLabless submits only a beater."""

from __future__ import annotations

import dataclasses

from loop import runner as R
from loop.contracts import RunStatus, Stage
from loop.core import bar as BR
from loop.core import meta as MT
from loop.core.data import AXES, Ledger, Node, Obs
from loop.spec import LablessCfg

from ..conftest import InlineExecutor
from ..test_runner import _node, _Pending

OLD, NEW = "a" * 40, "b" * 40
JITTER = (0.002, -0.002, 0.001, -0.001, 0.0, 0.0015, -0.0015, 0.0005, -0.0005, 0.0)


def _n(nid: str, parent: str | None, base: str, port_of: str = "") -> Node:
    return Node(nid, parent, None, "explore", "f", "explore", "root", 0.0, "", "", base, port_of)


def _obs(node: str, base: str, mean: float, seeds, pool: str = "select") -> list[Obs]:
    st = "confirm" if pool == "test" else "screen"        # v3 review K8: a fresh TEST run is a TEST-stage run
    return [Obs(f"{node}:{st}:{s}", node, s, st, pool, {a: mean + JITTER[i % 10] for a in AXES},
                mean + JITTER[i % 10], 1.0, 0.0, 0, None, base) for i, s in enumerate(seeds)]


def ledger(cands: dict[str, tuple]) -> Ledger:
    """Old base: root .668 and bar .679 on 40 seeds each. New base: root .663 on 5 seeds and ``cands`` = {node:
    (mean level, seeds[, how many of them are fresh TEST-pool seeds: default all])}."""
    nodes = {"root": _n("root", None, OLD), "bar": _n("bar", "root", OLD), "new": _n("new", None, NEW),
             **{c: _n(c, "new", NEW) for c in cands}}
    old = range(1, 41)
    obs = _obs("root", OLD, 0.668, old) + _obs("bar", OLD, 0.679, old) + _obs("new", NEW, 0.663, range(700001, 700006))
    for c, (m, k, *t) in cands.items():
        nt = t[0] if t else k
        obs += _obs(c, NEW, m, range(700001, 700001 + k - nt)) + _obs(c, NEW, m, range(800001, 800001 + nt), "test")
    return Ledger(nodes, (), tuple(obs), frozenset(), ((0.0, "root"),),
                  bases=((0.0, "root", OLD), (1.0, "new", NEW)))


LEADER, T = 0.6810443, 0.6850443
CFG = LablessCfg(bar_node="bar", leader_median=LEADER)          # promote_margin .004, promote_p .7, offset 0


def test_median_of_three_math():
    """P(median of 3 >= T) = 3q^2 - 2q^3; uncertainty in the mean; the needed local mean; offset; p = .5 -> T."""
    assert abs(BR.p_median(T, 0.003, T) - 0.5) < 1e-12
    q = 1 - BR.M._Phi((T - 0.687) / 0.003)
    assert abs(BR.p_median(0.687, 0.003, T) - (3 * q * q - 2 * q ** 3)) < 1e-12
    assert abs(BR.p_promote(T, 0.003, 4, T) - 0.5) < 1e-9                      # symmetric: P = .5 at the target
    assert BR.p_promote(0.687, 0.003, 4, T) < BR.p_median(0.687, 0.003, T)     # an uncertain mean: less sure
    assert abs(BR.need_mean(0.003, 4, T, 0.5) - T) < 1e-7
    n4, n10 = BR.need_mean(0.003, 4, T, 0.7), BR.need_mean(0.003, 10, T, 0.7)
    assert T + 0.001 < n4 < T + 0.0016 and n10 < n4                           # ~ +.0013 at sigma .003, n 4
    assert abs(BR.p_promote(n4, 0.003, 4, T) - 0.7) < 1e-6
    assert BR.need_mean(0.0025, 4, T, 0.7) < n4                               # less noise, less headroom needed
    assert abs(BR.need_mean(0.003, 4, T, 0.7, offset=0.002) - (n4 - 0.002)) < 1e-6
    lv = BR.Level(0.0, 0.0, 5, 0.004)                                          # sigma blend: pooled pulled to own
    assert 0.003 < BR.blend_sigma(0.003, lv) < 0.004 and BR.blend_sigma(0.003, None) == 0.003


def test_bar_from_config_and_back_compat():
    led = ledger({})
    b = BR.bar_from(led, CFG)
    assert b.lab and b.leader == LEADER and abs(b.target - T) < 1e-12 and b.p == 0.7 and b.min_test == 4
    old = BR.bar_from(led, LablessCfg(bar_node="bar", bar_score=0.68143, bar_se=0.0005, bar_margin=0.006))
    assert old.lab and abs(old.target - 0.68743) < 1e-12 and old.p == 0.7       # legacy keys read, same rule
    own = BR.bar_from(led, LablessCfg(bar_node="bar"))               # no Labless number: its local mean (+ bar_margin)
    assert not own.lab and abs(own.leader - 0.679) < 1e-9 and own.margin == 0.0 and own.tag == "bar"
    assert BR.bar_from(led, LablessCfg(bar_node="nope")) is None
    assert BR.configured(CFG) and BR.configured(LablessCfg(leader_median=0.68)) and not BR.configured(LablessCfg())


def test_verdict_is_p_median_of_three_on_fresh_test_seeds():
    led = ledger({"big": (0.690, 5), "near": (0.686, 5), "at": (0.685, 5), "few": (0.690, 1)})
    b = BR.bar_from(led, CFG)
    x = {n: BR.beat(led, b, n) for n in ("big", "near", "at", "few")}
    assert x["big"].beats and x["big"].p_test > 0.99 and abs(x["big"].d - (0.690 - T)) < 1e-9
    assert x["near"].beats and 0.7 <= x["near"].p_test < 0.85                  # just past the needed mean
    assert not x["at"].beats and abs(x["at"].p_test - 0.5) < 0.05               # AT the target median: a coin flip
    assert x["near"].need > T and x["at"].mean < x["at"].need < x["near"].mean
    assert not x["few"].beats and x["few"].p_test is None                       # one seed: provisional only
    assert BR.beat(led, b, "new") is None and BR.beat(led, b, "bar") is None    # the base root; an old-epoch node
    # verdicts first, then P; "few" (1 seed, none shared with the incumbent) never ranks: an unpaired seed offset
    assert [y.node for y in BR.ranked(led, b)] == ["big", "near", "at"]


def test_heartbeat_summary():
    led = ledger({"big": (0.690, 5)})
    assert BR.summary(led, LablessCfg()) == ""
    s = BR.summary(led, CFG, {"big": "065b87fdeadbeef"})
    assert s.startswith("bar=Labless median\u2265.6850 (leader .6810+.004) need local mean \u2265 .68")
    assert "; best=big@065b87f mean=.6900 need .68" in s and "P(promote)=1.00 test 5s" in s and s.endswith("*")
    assert BR.summary(led, LablessCfg(bar_node="bar")).startswith("bar=bar local .6790+.000\u2192.6790")
    assert BR.summary(led, LablessCfg(bar_node="nope")) == "bar=nope (not measurable)"
    assert BR.summary(ledger({}), CFG).endswith("best=none")


def test_verdict_uses_only_fresh_test_seeds_and_at_least_four():
    """Winner's-curse guard: a node screened on 5 select seeds looks as good as one tested, but is only provisional."""
    led = ledger({"screened": (0.690, 5, 0), "three": (0.690, 5, 3), "four": (0.690, 5, 4), "tested": (0.690, 5)})
    b = BR.bar_from(led, CFG)
    x = {n: BR.beat(led, b, n) for n in ("screened", "three", "four", "tested")}
    assert x["screened"].p > 0.9 and x["screened"].p_test is None and not x["screened"].beats
    assert x["three"].p > 0.9 and x["three"].n_test == 3 and x["three"].p_test is None and not x["three"].beats
    assert x["four"].n_test == 4 and x["four"].p_test is not None and x["four"].beats
    assert x["tested"].beats and x["tested"].n_test == 5
    low = ledger({"lucky": (0.679, 5, 4)})                      # a test block that disagrees: the verdict follows it
    assert not BR.beat(low, BR.bar_from(low, CFG), "lucky").beats
    s = BR.summary(ledger({"screened": (0.690, 5, 0)}), CFG)
    assert "provisional (5s, 0/4 test)" in s and not s.endswith("*")
    sec = MT.bar_section(led, CFG)
    assert sorted(x.split()[0] for x in sec["beats"]) == ["four", "tested"]
    assert BR.beat(led, BR.bar_from(led, dataclasses.replace(CFG, min_test_seeds=5)), "four").p_test is None


def test_meta_report_section_names_the_real_improvements():
    sec = MT.bar_section(ledger({"big": (0.690, 5), "same": (0.682, 5)}), CFG)
    assert sec["text"].startswith("bar=Labless median\u2265.6850 (leader .6810+.004) need local mean")
    assert sec["best"].startswith("big local mean .6900 (5 fresh TEST seeds") and "reached" in sec["best"]
    assert len(sec["beats"]) == 1 and sec["alpha"] == 0.7
    assert " short;" in MT.bar_section(ledger({"same": (0.682, 5)}), CFG)["best"]
    md = MT.render_markdown(MT.MetaReport(ts=1e9, since=0.0, bar=sec))
    assert "## External bar" in md and "beats the Labless bar (submission gate): big" in md
    sw = MT.render_markdown(MT.MetaReport(ts=1e9, since=0.0, bar=sec, switches=["big +0.0100 (n=5)"]))
    assert "pause-rule improvement (incumbent switches in window): big +0.0100 (n=5)" in sw
    assert "pause-rule improvement (incumbent switches in window): none" in md
    assert "beats the Labless bar (submission gate): none" in MT.render_markdown(
        MT.MetaReport(ts=1e9, since=0.0, bar=MT.bar_section(ledger({"same": (0.682, 5)}), CFG)))


def test_incumbent_labless_submits_only_a_node_that_beats_the_bar(ctx, store):
    ctx = dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, labless=dataclasses.replace(ctx.cfg.labless, enabled=True, bar_node="bar")))
    calls = []
    lab = R.IncumbentLabless(ctx, store, InlineExecutor(), clock=lambda: 1000.0,
                             submit=lambda ctx_, out, name, text: calls.append(text) or "sub-1")
    node = _node(store, "big")
    R.RunBook(ctx, store, _Pending()).launch(node, Stage.CONFIRM, 100_000)
    store.update_run("big:confirm:100000", status=RunStatus.DONE)
    assert lab.step("big") == []                                # a bar but no ledger view: nothing is submitted
    low = ledger({"big": (0.679, 5)})
    w = lab.step("big", low)
    assert w[0].startswith("labless: incumbent big not submitted, does not beat bar=bar local .6790") and not calls
    assert lab.step("big", low) == []                           # said once per seed count
    assert [e for e in store.events() if e["event"] == "labless_below_bar"]
    assert lab.step("big", ledger({"big": (0.690, 5)})) == ["labless: submitting incumbent big from big:confirm:100000"]
    assert calls == ["v3 incumbent big: passed the fresh-seed test and beats the bar bar"]


def test_ranked_uses_the_paired_posterior_and_skips_wave_points_and_unpaired_nodes():
    """v3-c7a A1: "best=w013.p03 .6858" was a wave point (4 identical replicates at one seed, no control). The ranking
    now skips wave points and unpaired nodes and ranks on incumbent level + the shrunk own paired gain; the heartbeat
    line and the meta report name the same best."""
    nodes = {"root": _n("root", None, OLD), "new": _n("new", None, NEW), "inc": _n("inc", "new", NEW),
             "good": _n("good", "inc", NEW), "meh": _n("meh", "inc", NEW), "unp": _n("unp", "inc", NEW),
             "grand": _n("grand", "good", NEW),
             "w9.p03": dataclasses.replace(_n("w9.p03", "inc", NEW), notes="wave=w9 point=3 block=0 seed=900")}
    inc_seeds = range(700001, 700011)
    obs = _obs("new", NEW, 0.663, range(700001, 700006)) + _obs("inc", NEW, 0.680, inc_seeds)
    obs += _obs("good", NEW, 0.6835, range(700001, 700005)) + _obs("meh", NEW, 0.6805, range(700001, 700005))
    obs += _obs("unp", NEW, 0.690, [990001]) + _obs("w9.p03", NEW, 0.6858, [900] * 4)
    obs += _obs("grand", NEW, 0.6845, [990002, 990003])                      # pairs through its parent only
    obs += _obs("grand", NEW, 0.6845, [])
    obs += [dataclasses.replace(o, node="grand", run_id=o.run_id.replace("good", "grand"),
                                panel=o.panel + 0.001) for o in _obs("good", NEW, 0.6835, [990002, 990003])]
    obs += _obs("good", NEW, 0.6835, [990002, 990003])
    led = Ledger(nodes, (), tuple(obs), frozenset(), ((0.0, "root"),), bases=((0.0, "root", OLD), (1.0, "new", NEW)))
    b = BR.bar_from(led, CFG)
    r = BR.ranked(led, b, incumbent="inc")
    names = [x.node for x in r]
    assert "w9.p03" not in names and "unp" not in names                     # a wave point; 1 unpaired seed
    assert names.index("good") < names.index("meh") and "grand" in names    # grand: paired via good
    g = next(x for x in r if x.node == "good")
    assert g.n_pair == 4 and g.inc == "inc" and 0 < g.gain < 0.0035 + 1e-9  # shrunk (or at most raw +.0035)
    assert abs(g.post - (0.680 + g.gain)) < 1e-9 and 0 < g.p_post < 1
    s = BR.summary(led, CFG, incumbent="inc")
    assert f"best={r[0].node} post=" in s and "provisional" in s
    sec = MT.bar_section(led, CFG, incumbent="inc")
    assert sec["best"].startswith(f"{r[0].node} local mean") and "paired posterior" in sec["best"]
    pv = BR.paired(led, "inc", eb_min_nodes=0)                              # flat: the raw paired gain
    assert abs(pv.gains["good"][0] - 0.0035) < 1e-9 and pv.gains["inc"] == (0.0, 0.0, 10)


def test_meta_bar_section_ranks_against_the_ticks_incumbent_like_the_heartbeat():
    """06:30Z live: the report's best candidate used the view's replay incumbent while the hb used the tick's status
    incumbent -> two different 'best'. bar_incumbent prefers state tick.incumbent; both rank with bar.ranked."""
    class St:
        def __init__(self, v):
            self.v = v

        def get_state(self, k):
            return self.v if k == "tick.incumbent" else None
    assert MT.bar_incumbent(St("inc"), "other") == "inc" and MT.bar_incumbent(St(None), "other") == "other"
    assert MT.bar_incumbent(St(""), "") is None
    nodes = {"root": _n("root", None, OLD), "new": _n("new", None, NEW), "inc": _n("inc", "new", NEW),
             "good": _n("good", "inc", NEW), "stale": _n("stale", "new", NEW)}
    obs = _obs("new", NEW, 0.663, range(700001, 700006)) + _obs("inc", NEW, 0.680, range(700001, 700011))
    obs += _obs("good", NEW, 0.6835, range(700001, 700005)) + _obs("stale", NEW, 0.640, range(700001, 700005))
    led = Ledger(nodes, (), tuple(obs), frozenset(), ((0.0, "root"),), bases=((0.0, "root", OLD), (1.0, "new", NEW)))
    b = BR.bar_from(led, CFG)
    inc = MT.bar_incumbent(St("inc"), "stale")
    hb = BR.summary(led, CFG, incumbent=inc)
    sec = MT.bar_section(led, CFG, incumbent=inc)
    best = BR.ranked(led, b, incumbent=inc)[0].node
    assert hb.split("best=")[1].startswith(best) and sec["best"].startswith(best)


def test_meta_report_bar_section_ranks_on_the_official_ledger_not_the_visible_one(store, monkeypatch):
    """Live 10-14 (DB-copy shape): the tick hands meta the holdout's VISIBLE-composite ledger (every panel on another
    scale); the bar section ranked on it -> 'best w019.x local mean .6715' while the heartbeat's best= (the full ledger)
    was n2678 post .6805. report(bar_led=...) ranks the bar section on the official ledger; without it, the old drift."""
    nodes = {"root": _n("root", None, OLD), "new": _n("new", None, NEW), "inc": _n("inc", "new", NEW),
             "good": _n("good", "inc", NEW), "weak": _n("weak", "inc", NEW)}
    seeds = range(100098, 100103)
    obs = _obs("new", NEW, 0.663, range(700001, 700006)) + _obs("inc", NEW, 0.680, seeds)
    obs += _obs("good", NEW, 0.6812, seeds) + _obs("weak", NEW, 0.678, seeds)
    full = Ledger(nodes, (), tuple(obs), frozenset(), ((0.0, "root"),), bases=((0.0, "root", OLD), (1.0, "new", NEW)))
    # visible composite: held-out tasks removed -> a lower panel for every run, and a different node order
    shift = {"inc": -0.0090, "good": -0.0100, "weak": -0.0085, "new": -0.009}
    vis = dataclasses.replace(full, obs=tuple(dataclasses.replace(o, panel=o.panel + shift[o.node]) for o in full.obs))
    monkeypatch.setattr(MT.ID, "plateau", lambda *a, **k: None)      # the toy ledger has no GPU-h history
    store.set_state("tick.incumbent", "inc")                          # what the tick records for the hb
    hb = BR.summary(full, CFG, incumbent="inc")
    best = hb.split("best=")[1].split()[0]
    off, _ = MT.report(vis, store, now=2e9, bar=CFG, bar_led=full)
    assert off.bar, off.errors
    assert off.bar["best"].startswith(best) and best == "good"
    assert f"{BR.beat_text(BR.ranked(full, BR.bar_from(full, CFG), incumbent='inc')[0], BR.bar_from(full, CFG))}" == off.bar["best"]
    op, llm = MT.render_markdown(off), MT.render_markdown(off, llm=True)      # operator copy official, LLM copy visible
    assert off.bar["best"] in op and off.bar["best"] not in llm and off.bar_vis["best"] in llm
    assert "VISIBLE-task composite only" in llm and "VISIBLE-task composite only" not in op   # 10-22: review vs meta line
    off.bar_vis = {}                                                          # visible section failed: omitted, never leaked
    assert "External bar" not in MT.render_markdown(off, llm=True)
    drift, _ = MT.report(vis, store, now=2e9, bar=CFG)                 # no bar_led: the visible ledger's own level
    assert drift.bar["best"] != off.bar["best"]
