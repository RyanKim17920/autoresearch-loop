"""10-22: the bold knob judge was qwen-only (183 errors vs 3 answers in 7 days); a deterministic AST / novelty judge
decides whenever qwen is unavailable, and every call logs which judge decided."""

from __future__ import annotations

from types import SimpleNamespace

from loop.core import bold as BOLD

from .test_c8b_bold import CONST, NEWDEF, NEWFILE, NEWLINE, benv  # noqa: F401  (the fixture)

SWITCH = ("--- a/train.py\n+++ b/train.py\n@@ -9 +9 @@\n-    t = sched('cosine')\n+    t = sched('linear')\n"
          "+    z = f(g(t))\n")                        # a switched option: one or two new calls, no new def / class
MODULE = ("--- a/model.py\n+++ b/model.py\n@@ -1,2 +1,5 @@\n class Net:\n+class Ctx(nn.Module):\n+    def forward(self, x):\n"
          "+        return x\n")


def test_deterministic_verdicts_follow_the_ast_and_the_novelty_parts():
    assert BOLD.det_knob_verdict(CONST)[0] is True                          # constants only
    assert BOLD.det_knob_verdict(NEWDEF)[0] is False                        # a new def
    assert BOLD.det_knob_verdict(NEWFILE)[0] is False or BOLD.det_knob_verdict(NEWFILE)[1]
    assert BOLD.det_knob_verdict(MODULE)[0] is False                        # a new nn.Module
    ok, why = BOLD.det_knob_verdict("not a diff at all")
    assert isinstance(ok, bool) and why
    ok, why = BOLD.det_knob_verdict(NEWLINE)
    assert ok is (BOLD.novelty_parts(NEWLINE)["other"] <= BOLD.DET_OTHER_MAX
                  and not any(BOLD.novelty_parts(NEWLINE)[k] for k in ("defs", "classes", "modules", "losses")))


def test_qwen_decides_when_up_and_the_deterministic_judge_when_down_and_each_call_is_logged():
    log: list[tuple[str, dict]] = []
    rec = lambda ev, **kw: log.append((ev, kw))                          # noqa: E731

    def down(d):
        raise RuntimeError("qwen down")
    j = BOLD.chained_knob_judge(lambda d: True, log=rec)                 # qwen says knob on a structural diff: it stands
    assert j(NEWDEF) is True and j.last == "qwen"
    assert log[-1][0] == "bold_knob_judge" and log[-1][1]["judge"] == "qwen" and log[-1][1]["det"] is False
    j = BOLD.chained_knob_judge(down, log=rec)
    assert j(NEWDEF) is False and j.last == "deterministic" and j(CONST) is True
    assert [e for e, _ in log].count("bold_knob_judge_error") == 2
    assert log[-1][1]["judge"] == "deterministic" and log[-1][1]["knob_only"] is True
    j = BOLD.chained_knob_judge(lambda d: None, log=rec)                 # an unparseable answer is no verdict either
    assert j(CONST) is True and j.last == "deterministic"
    j = BOLD.chained_knob_judge(None, log=rec)
    assert j(NEWDEF) is False and j.last == "deterministic"


def test_ideation_never_holds_a_node_for_a_knob_verdict_any_more(benv):
    ctx, store, led, ide = benv

    class Down:
        def complete_json(self, *a, **k):
            raise RuntimeError("qwen down")
    ide.llm = Down()
    cand = SimpleNamespace(diff_text=CONST, hypothesis=SimpleNamespace(claim="x"))
    assert ide._bold_rule(cand)[0] == "bold_knob_only"                   # no AST-only structural fallthrough, no hold
    cand = SimpleNamespace(diff_text=NEWDEF, hypothesis=SimpleNamespace(claim="x"))
    assert ide._bold_rule(cand) == ("", "")
    ev = store.events("bold_knob_judge")
    assert ev and all(e["judge"] == "deterministic" for e in ev)
    ide.llm = None                                                      # no LLM wired at all: still answers
    assert ide._bold_rule(SimpleNamespace(diff_text=NEWDEF, hypothesis=cand.hypothesis)) == ("", "")


CURRICULUM = ("--- a/train.py\n+++ b/train.py\n@@ -1,6 +1,8 @@\n import torch\n def train_step(self, x, step):\n"
              "-    z = self.student(x)\n+    if step < 1000:\n+        x = x[:, ::2]\n+    z = self.student(x)\n"
              "     t = self.teacher(x)\n")
DETACH = ("--- a/train.py\n+++ b/train.py\n@@ -1,6 +1,8 @@\n import torch\n def train_step(self, x, step):\n"
          "     z = self.student(x)\n-    t = self.teacher(x)\n+    t = self.student(x).detach()\n")
RETARGET = ("--- a/train.py\n+++ b/train.py\n@@ -1,6 +1,8 @@\n import torch\n def train_step(self, x, step):\n"
            "     z = self.student(x)\n     u = self.teacher(x)\n-    t = self.teacher(x)\n+    t = u\n")
KNOB = ("--- a/train.py\n+++ b/train.py\n@@ -1,6 +1,8 @@\n import torch\n def train_step(self, x, step):\n"
        "-    loss_weight = 0.5\n+    loss_weight = 0.7\n     z = self.student(x)\n")
PARENT = "import torch\ndef train_step(self, x, step):\n    z = self.student(x)\n    t = self.teacher(x)\n    u = self.teacher(x)\n"


def test_a_curriculum_branch_a_detached_target_and_a_retargeted_loss_are_structural_but_a_constant_edit_is_knob():
    """S5: DET_OTHER_MAX was 2, so one new branch (curriculum) or the stop-grad target swap passed as knob-only."""
    assert BOLD.DET_OTHER_MAX == 0
    parent = lambda path: PARENT                                                        # noqa: E731
    for name, d in (("curriculum", CURRICULUM), ("detach", DETACH), ("retarget", RETARGET)):
        ok, why = BOLD.det_knob_verdict(d, parent)
        assert ok is False, (name, why)
    assert BOLD.det_knob_verdict(KNOB, parent)[0] is True                              # a loss weight value: still a knob
    assert BOLD.det_knob_verdict(CONST)[0] is True


def test_names_the_parent_already_has_do_not_count_as_new_items():
    parent = lambda path: PARENT + "    y = self.student(self.teacher(x))\n"             # noqa: E731
    d = ("--- a/train.py\n+++ b/train.py\n@@ -1,6 +1,8 @@\n import torch\n def train_step(self, x, step):\n"
         "-    z = self.student(x)\n+    z = self.student(x)  # same call\n")
    assert BOLD.det_knob_verdict(d, parent)[0] is True
