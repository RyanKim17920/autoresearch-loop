"""10-22: the plateau review never triggered under a flat best (last review 231 h earlier): the spend rule (> 2 x the
median GPU-h between passes) stayed off at 1338 of 1851 GPU-h after 9 days with the best unchanged. A wall-of-ledger-time
idle rule (no PASS for PLATEAU_IDLE_H with >= PLATEAU_IDLE_GPU_X x the median spent since) now triggers it."""

from __future__ import annotations

from types import SimpleNamespace

from loop.core import ideation as ID

from ..conftest import InlineExecutor

H = 3600.0


def _led(passes_h, runs):
    return (SimpleNamespace(runs=tuple(SimpleNamespace(done_ts=t * H, gpu_h=g) for t, g in runs)),
            [SimpleNamespace(ts=t * H, switched=True) for t in passes_h])


def _flat(extra_gpu=1338.0, last_h=280.0):
    """4 passes with ~900 GPU-h between them (median 900), then a flat best."""
    runs = [(10 * k - 1, 900.0) for k in (1, 2, 3, 4)]
    return _led([10, 20, 30, 40], runs + [(41.0, extra_gpu / 2), (last_h, extra_gpu / 2)])


def test_the_spend_rule_alone_misses_a_flat_best_and_the_idle_rule_catches_it():
    led, judged = _flat()
    p = ID.plateau(led, judged)
    assert p.active and p.why == "idle" and p.idle_h > ID.PLATEAU_IDLE_H and p.median == 900.0
    assert p.gpu_h < ID.PLATEAU_FACTOR * p.median                      # the old rule: 1338 < 1800 -> inactive
    big = ID.plateau(*_flat(extra_gpu=2400.0, last_h=60.0))
    assert big.active and big.why == "gpu"                              # the spend rule still fires by itself


def test_the_idle_rule_needs_both_time_and_spend():
    assert not ID.plateau(*_flat(last_h=60.0)).active                   # 20 h since the last pass: young
    assert not ID.plateau(*_flat(extra_gpu=300.0)).active               # old but nothing spent: a stopped loop
    led, judged = _led([], [(1.0, 5.0)])
    assert not ID.plateau(led, judged).active                          # no pass yet: nothing to compare with


def test_a_flat_best_starts_the_plateau_review_through_the_tick_hook(env):
    from .test_fnf import _Research
    ctx, store, led, ide = env
    flat, judged = _flat()
    v = SimpleNamespace(led=flat, judged=judged)
    seen = []

    class LLM:
        def complete_json(self, role, **kw):
            seen.append(role)
            return {"diagnosis": "d", "hypotheses": [{"claim": "c"}]}
    ide.llm = LLM()
    ide.review_prompt = lambda *a: "prompt"
    tick = SimpleNamespace(research=_Research(0), pool=InlineExecutor(), dry=False)
    acts = ide._plateau(tick, v, {})
    assert acts and acts[0].startswith("plateau (idle)") and ide._review is not None
    acts = ide._plateau(tick, v, {})
    st = store.get_state(ID.K_PLATEAU)
    assert seen == ["review"] and st["active"] and st["since"] == 40 * H and acts == ["plateau review: 1 hypotheses"]


from .test_fnf import env  # noqa: E402,F401  (the fixture)
