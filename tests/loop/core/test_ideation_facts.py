"""The computed facts every brief and the plateau review read (fix batch B, 10-12): axis trades, crash-censored
regions, per-key coverage on the carrier lineage, harm-blocked nodes; the plateau review redone at tick 0 on new
inputs, its hypotheses taking the first LLM slots and its knobs joining the incumbent's wave bank."""

from __future__ import annotations

from types import SimpleNamespace

from loop.core import data as D
from loop.core import ideation as ID

from ..conftest import InlineExecutor
from .test_ideation import _Research, env  # noqa: F401  (the fixture)


def _n(nid, parent, notes=""):
    return D.Node(nid, parent, None, "explore", "f", "explore", "root", 0.0, "", notes, "b", "")


def _r(node, seed, status):
    return D.Run(f"{node}:{seed}", node, seed, "screen", status, 1.0, 0.0, 0.0, "", 0)


class _Store:
    def __init__(self, state):
        self.state = state

    def get_state(self, k, default=None):
        return self.state.get(k, default)


def test_crash_censored_regions_are_unmeasured_not_negative_and_coverage_lists_the_lineage():
    nodes = {n.id: n for n in [_n("root", None), _n("inc", "root"), *(_n(f"c{i}", "inc") for i in range(7))]}
    cfg = {"c0": {"train": {"batch_size": 128}}, "c1": {"train": {"batch_size": 144}},
           "c2": {"train": {"batch_size": 192}}, "c3": {"train": {"batch_size": 256}},
           "c4": {"model": {"type": "big"}}, "c5": {"model": {"type": "big"}}, "c6": {"dino": {"lr": 3e-4}}}
    runs = (_r("c0", 1, "done"), _r("c1", 1, "done"), _r("c2", 1, "crash"), _r("c3", 1, "crash"),
            _r("c3", 2, "crash"), _r("c4", 1, "crash"), _r("c5", 1, "crash"), _r("c6", 1, "done"), _r("c6", 2, "done"))
    led = D.Ledger(nodes, runs, (), frozenset(), ((0.0, "root"),))
    store = _Store({f"config_diff:{k}": v for k, v in cfg.items()})
    cen = ID.censored(store, led)
    assert any(x.startswith("train.batch_size > 144: tried [192, 256] on 2 nodes") and "unmeasured, not negative" in x
               for x in cen)
    assert any(x.startswith('model.type in ["\\"big\\""]') or x.startswith("model.type in") for x in cen)
    assert not any(x.startswith("dino.lr") for x in cen)                            # it finished: measured
    cov = ID.coverage(store, led, "inc")
    assert cov[0].startswith("lineage inc > root")
    assert any(x.startswith("train.batch_size: ") and "128 (1 nodes / 1 seeds)" in x for x in cov)
    assert any(x.startswith("dino.lr: 0.0003 (1 nodes / 2 seeds)") for x in cov)


def test_axis_trades_and_the_facts_block_on_the_ledger_fixture(env):  # noqa: F811
    ctx, store, led, ide = env
    v = ide.view(led)
    rows = ID.trades(v, store)
    assert 0 < len(rows) <= ID.TRADE_K and all({"cls", "rob", "seg", "composite", "keys"} <= set(r) for r in rows)
    assert [abs(r["cls"]) for r in rows] == sorted((abs(r["cls"]) for r in rows), reverse=True)
    text = ide.facts(v)
    for head in ("### Axis trades", "### Crash-censored regions", "### Per-key coverage", "### Harm-blocked"):
        assert head in text
    b = ide.brief_for("explore", led, v)
    assert "## Computed facts" in b.text and rows[0]["node"] in b.text
    review = ide.review_prompt(v, ID.plateau(led, v.judged), {})
    assert "### Axis trades" in review and "{" + "facts}" not in review


def test_a_review_on_older_inputs_is_redone_at_once_and_its_briefs_take_the_first_slots(env, monkeypatch):  # noqa: F811
    ctx, store, led, ide = env
    object.__setattr__(ctx.cfg.ideation, "kind_weight", False)        # the fixed rotation (A3 kind weights off)
    v = ide.view(led)
    monkeypatch.setattr(ID, "plateau", lambda led, judged: ID.Plateau(True, 123.0, 50.0, 10.0, 2))
    old = {"active": True, "since": 123.0, "ts": 1e18, "served": 1, "diagnosis": "d",   # recent, served, no inputs
           "hypotheses": [{"claim": "c0", "evidence": "e", "decisive_experiment": "x0", "prediction": {}}]}
    store.set_state(ID.K_PLATEAU, old)
    ide.clock = lambda: 1e18

    class LLM:
        def complete_json(self, role, *, system, prompt, schema):
            return {"diagnosis": "d", "hypotheses": [
                {"claim": f"c{i}", "evidence": "e", "decisive_experiment": f"x{i}", "prediction": {},
                 "knobs": [{"key": "dino.clip_grad", "lo": 3.0, "hi": 9.0, "scale": "log"}]} for i in (1, 2)]}
    ide.llm = LLM()
    tick = SimpleNamespace(research=_Research(0), pool=InlineExecutor(), dry=False)
    assert "review started" in " ".join(ide._plateau(tick, v, {}))
    assert "2 hypotheses" in " ".join(ide._plateau(tick, v, {}))
    assert store.get_state(ID.K_PLATEAU)["inputs"] == ID.PLATEAU_INPUTS
    tick = SimpleNamespace(research=_Research(3), pool=InlineExecutor(), dry=False)
    ide.ideate(tick, [SimpleNamespace(kind="idea", lane=l) for l in ("exploit", "crossover", "exploit")], led, {})
    kinds = [r.brief_kind for r in tick.research.started]
    assert kinds == ["kuhn", "kuhn", "repair"]                            # the review's hypotheses go first (kuhn)
    assert "c1" in tick.research.started[0].brief and "c2" in tick.research.started[1].brief


def test_plateau_knobs_join_the_incumbents_wave_bank(ctx, tmp_path):
    from loop import tick as TK
    from loop.core import factorial as F

    from ..test_port import _two_epochs
    from ..test_waves import _Post
    sim, store = _two_epochs(ctx, tmp_path)
    core = TK.ModuleCore()
    tick = TK.Tick(ctx, store, sim, core, db_path=tmp_path / "loop.db", clock=lambda: 50.0)
    led = D.load(tmp_path / "loop.db", ctx.task_id)
    store.set_state(core.K_BANK + "inc@new", {"knobs": [], "lines": [], "tau2": 2e-6, "effects": 0, "used": False,
                                              "v": core.RETRO_V})
    store.set_state(ID.K_PLATEAU, {"active": True, "hypotheses": [
        {"claim": "c", "knobs": [{"key": "train.weight_decay", "lo": 0.05, "hi": 0.1, "scale": "log"},
                                 {"key": "train.lr", "lo": 1.0, "hi": 2.0}]}]})        # lr: the carrier is not at 1.0
    post = _Post()
    post.est, post.by = {}, {}
    bank = core._carrier_bank(tick, led, post, "inc@new", [])
    assert [(b.knob.key, b.source) for b in bank] == [("train.weight_decay", "plateau")]
    store.set_state(core.K_USED + "inc@new", [bank[0].signature()])
    assert core._carrier_bank(tick, led, post, "inc@new", []) == []                      # consumed by its wave
    assert isinstance(bank[0], F.BankKnob)
    tick.close()
    store.close()
