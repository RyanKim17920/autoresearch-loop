"""v3-c5b M3: eval-only eligibility vs the TRAINING path. train.py's EMA teacher runs ``forward`` in eval mode during
training, so ``if not self.training`` bodies of a training span (and every span it calls) run on the teacher; an
existing buffer / attribute's initial value (``rn_fitted = True``) switches the teacher's readout on mid-training."""

from __future__ import annotations

import pytest

from loop.adapters.nanopath_v2 import train_reachable
from loop.spec import EvalOnlyCfg, FamilyRule
from tests.loop.test_adapters import git, make_np, np_ctx_with_locks
from tests.loop.test_c5_evalonly_lane import commit_on
from tests.loop.test_evalonly import TRAIN, repo  # noqa: F401  (the fixture)

VIT = '''import torch


class ViT:
    def __init__(self, robust):
        if robust:
            self.register_buffer("rn_fitted", torch.zeros((), dtype=torch.bool))
            self.register_buffer("rn_mu", torch.zeros(3))
        else:
            self.rn_fitted = self.pf_fitted = False

    def _suppress(self, x, mean):
        return x - mean

    def forward(self, x):
        if not self.training and self.rn_fitted:
            x = self._suppress(x, self.rn_mu)
        return self._forward_tokens(x)

    def _forward_tokens(self, x):
        return x * 2

    def probe_features(self, x):
        return self._suppress(self(x), self.rn_mu) * 1.0

    def build_blocks(self):
        return 1
'''
SPANS = ("model.py::ViT.forward", "model.py::ViT._forward_tokens", "model.py::ViT._suppress",
         "model.py::ViT.probe_features", "train.py::# Fit scanner-response directions...model.ct_ms.copy_")
TRAIN_SPANS = ("model.py::ViT.forward", "model.py::ViT._forward_tokens")


def adapter(tmp_path, repo, train=TRAIN_SPANS):   # noqa: F811
    fams = {"eval_readout": FamilyRule(paths=SPANS, combine=True), "architecture": FamilyRule(paths=("model.py",))}
    ctx = np_ctx_with_locks(tmp_path / "state", ("probe.py",), families=fams,
                            evalonly=EvalOnlyCfg(enabled=True, train_spans=train, readout_keys=True))
    return make_np(ctx, repo)[0]


@pytest.fixture
def vit(repo):   # noqa: F811
    base = git(repo, "rev-parse", "main")
    return commit_on(repo, base, "vit", **{"model.py": VIT, "train.py": TRAIN})


def test_reachability_follows_self_calls():
    got = train_reachable([VIT], SPANS, TRAIN_SPANS)
    assert got == {"model.py::ViT.forward", "model.py::ViT._forward_tokens", "model.py::ViT._suppress"}
    assert train_reachable([VIT], SPANS, ()) == set()          # no train_spans: the old lane's behaviour


@pytest.mark.parametrize("edit,ok,why", [
    # a readout-only span: eligible
    (("* 1.0", "* 0.5"), True, ""),
    # inside `if not self.training` of forward: the eval-mode teacher runs it during training
    (("x = self._suppress(x, self.rn_mu)", "x = self._suppress(x, self.rn_mu) * 0.9"), False, "training-path"),
    # a span forward reaches (the teacher calls it through forward)
    (("return x - mean", "return x - 0.5 * mean"), False, "reached from"),
    # the rn_fitted=True attack: an existing attribute's initial value flips the teacher's readout on
    (("self.rn_fitted = self.pf_fitted = False", "self.rn_fitted = self.pf_fitted = True"), False, "initial value"),
    (('"rn_fitted", torch.zeros((), dtype=torch.bool)', '"rn_fitted", torch.ones((), dtype=torch.bool)'), False,
     "initial value"),
    (('"rn_mu", torch.zeros(3)', '"rn_mu", torch.ones(3)'), False, "initial value"),
    # an added fitted buffer read only by the readout-only span: eligible (a shim)
    (('self.register_buffer("rn_mu", torch.zeros(3))\n',
      'self.register_buffer("rn_mu", torch.zeros(3))\n            self.register_buffer("pf_z", torch.zeros(3))\n'),
     True, ""),
])
def test_training_path_and_buffer_init(tmp_path, repo, vit, edit, ok, why):   # noqa: F811
    child = commit_on(repo, vit, "c", **{"model.py": VIT.replace(*edit)})
    got = adapter(tmp_path, repo).eval_only(vit, child)
    assert got[0] is ok, got
    assert why in got[1]


def test_added_buffer_read_by_the_teacher_path_is_training(tmp_path, repo, vit):   # noqa: F811
    text = (VIT.replace('self.register_buffer("rn_mu", torch.zeros(3))\n',
                        'self.register_buffer("rn_mu", torch.zeros(3))\n'
                        '            self.register_buffer("pf_z", torch.zeros(3))\n'))
    ok_child = commit_on(repo, vit, "ok", **{"model.py": text.replace("* 1.0", "* self.pf_z")})
    bad_child = commit_on(repo, vit, "bad", **{"model.py": text.replace("return x * 2", "return x * 2")
                                               .replace("return x - mean", "return x - mean - self.pf_z")})
    ad = adapter(tmp_path, repo)
    assert ad.eval_only(vit, ok_child)[0] is True
    ok, why, _ = ad.eval_only(vit, bad_child)
    assert not ok and ("training-path" in why or "readout-only" in why), why


def test_constant_read_inside_a_train_span_is_training(tmp_path, repo, vit):   # noqa: F811
    text = VIT.replace("class ViT:\n", "class ViT:\n    SHRINK = 0.5\n\n")
    base = commit_on(repo, vit, "k0", **{"model.py": text.replace("return x * 2", "return x * self.SHRINK")})
    child = commit_on(repo, base, "k1", **{"model.py": text.replace("SHRINK = 0.5", "SHRINK = 0.25")
                                           .replace("return x * 2", "return x * self.SHRINK")})
    ok, why, _ = adapter(tmp_path, repo).eval_only(base, child)
    assert not ok and "readout-only" in why, why
