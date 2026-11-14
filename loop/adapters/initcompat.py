"""loop.adapters.initcompat — the hard init-compatibility check (the carried lesson).

An architecture change that breaks model init makes the model much worse
(DESIGN D0.5). This module is the check that runs BEFORE any GPU spend.

Split in two so the decision is testable without torch:

* ``SCRIPT`` is a CPU script run with the *worktree's* python. It only
  MEASURES: builds the child model (worktree files) and the parent model (files
  extracted from the parent commit), runs a dummy batch through both, and prints
  one JSON object.
* :func:`evaluate` applies the thresholds (:class:`Limits`) to that JSON and
  returns ``contracts.InitCompat`` with per-check detail.

Checks: (1) construction, (2) finite forward, (3) initial loss (and feature
scale) within a factor of the parent's, (4) parameter-count ratio bounds,
(5) the parent's init/checkpoint still loads with no missing/unexpected keys --
except CHILD-ONLY keys (new parameters / buffers the parent never had) when the
child is IDENTITY AT INIT: loaded with the parent's state, its features and
proxy loss equal the parent's within ``identity_tol`` (zero-initialised
residual branches, gates at 0, new heads beside the pretrained ones). Live
2026-10-06: 75 init_incompatible rejections, 66 added a module / parameter and
26 a buffer, 36 of them zero-init. A resized / renamed / dropped parent tensor
(unexpected keys, a shape mismatch) still fails.

INIT SWAP (operator 2026-10-07: "different init models ... it should be able to do whatever it wants"): a child
whose ``model.type`` or declared checkpoint (``model.pretrained_url``) differs from the parent's is a NEW init,
so (3)-(5) -- all relative to the parent -- do not apply. Its checks (:data:`SWAP_CHECKS`): (1) construction,
(2) a finite forward, ``init_legal`` (the source and type match no ``Limits.deny`` pattern: pathology-trained
and GenBio checkpoints may never initialise or guide the model; nothing else is restricted), ``load`` (the
child's own ``load_pretrained`` runs and the loaded model's forward is finite; missing / unexpected keys are
reported, never judged) and ``flop_budget`` (the run must reach the sample budget within the FLOP cap: the
smoke's measured FLOPs per sample x ``train.max_train_samples`` / ``train.max_train_flops`` <= ``flop_cap``;
pending until a smoke measured it). :attr:`InitCompat.rule` names the failure: ``init_illegal``,
``over_flop_cap`` or ``init_incompatible``.

"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from loop.contracts import InitCompat

CHECKS = ("construct", "forward_finite", "initial_loss", "param_ratio", "checkpoint_keys")
SWAP_CHECKS = ("construct", "forward_finite", "init_legal", "load", "flop_budget")


@dataclass(frozen=True)
class Limits:
    loss_factor: float = 2.0            # child/parent initial loss within [1/f, f]
    loss_floor: float = 0.05            # added to both losses: a near-zero proxy loss must not make the ratio noisy
    feature_factor: float = 4.0         # child/parent feature rms within [1/f, f]
    param_ratio_min: float = 0.5
    param_ratio_max: float = 2.0
    image_size: int = 112
    batch: int = 4
    seed: int = 0
    timeout_s: float = 900.0
    model_file: str = "model.py"
    config_rel: str = "configs/main.yaml"
    identity_tol: float = 1e-4          # child-only keys pass when max|z_child - z_parent| <= tol x rms(z_parent)
    deny: tuple[str, ...] = ()          # init swap: regexes (case-insensitive) an init source / type may not match
    allow: tuple[str, ...] = ()         # init swap: the weights-source ALLOWLIST (see :func:`allowed`); () = no allowlist
    flop_cap: float = 0.97              # init swap: max projected FLOP fraction at the full sample budget


# ---------------------------------------------------------------------------
# The measuring script (runs in the worktree's venv; torch + yaml available there)
# ---------------------------------------------------------------------------

FETCH_SPY = r'''
def spy_fetches(fetches, mod=None):
    """S1: record every repo id / url / model name the loader passes to a weights fetcher (hf_hub_download,
    snapshot_download, timm.create_model(pretrained=True), torch.hub.load / load_state_dict_from_url), then call
    through. Returns the restore callables."""
    undo = []

    def first(a, k, *names):
        return a[0] if a else next((k[n] for n in names if n in k), None)

    def wrap(owner, name, ident):
        orig = getattr(owner, name, None)
        if orig is None:
            return

        def f(*a, **k):
            i = ident(a, k)
            if i:
                fetches.append(str(i))
            return orig(*a, **k)
        setattr(owner, name, f)
        undo.append(lambda: setattr(owner, name, orig))
        for key, val in list(vars(mod).items()) if mod is not None else []:
            if val is orig:                      # `from x import fn` bound at the model module's import
                setattr(mod, key, f)
                undo.append(lambda key=key, val=val: setattr(mod, key, val))

    try:
        import huggingface_hub as hh
        for n in ("hf_hub_download", "snapshot_download"):
            wrap(hh, n, lambda a, k: first(a, k, "repo_id"))
    except ImportError:
        pass
    try:
        import timm
        wrap(timm, "create_model", lambda a, k: first(a, k, "model_name") if k.get("pretrained") else None)
    except ImportError:
        pass
    try:
        import torch
        wrap(torch.hub, "load", lambda a, k: first(a, k, "repo_or_dir"))
        wrap(torch.hub, "load_state_dict_from_url", lambda a, k: first(a, k, "url"))
    except ImportError:
        pass
    return undo
'''

SCRIPT = r'''
import argparse, importlib.util, json, math, sys, traceback
ap = argparse.ArgumentParser()
ap.add_argument("--child-dir", required=True)
ap.add_argument("--parent-dir", required=True)
ap.add_argument("--model-file", default="model.py")
ap.add_argument("--config", default="configs/main.yaml")
ap.add_argument("--image-size", type=int, default=112)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
import torch, yaml
import torch.nn.functional as F
from pathlib import Path


def load(root, tag):
    path = Path(root) / a.model_file
    spec = importlib.util.spec_from_file_location("loop_" + tag + "_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(root))
    return mod


def folded_type(m):
    """The port's train.py folds `type: timm` + `timm_name: <id>` into `timm:<id>`; check what actually loads."""
    t = m.get("type")
    return "timm:" + str(m["timm_name"]) if t == "timm" and m.get("timm_name") else t


def build(root, tag):
    cfg = yaml.safe_load((Path(root) / a.config).read_text())
    mod = load(root, tag)
    torch.manual_seed(a.seed)
    cfg["model"]["type"] = folded_type(cfg["model"])
    model = mod.ViT(variant=cfg["model"]["type"],
                    drop_path_rate=float((cfg.get("dino") or {}).get("drop_path_rate", 0.0)))
    return model, cfg


def features(model, x):
    enc = getattr(model, "encode_image", model)
    z = enc(x)
    if isinstance(z, dict):
        z = next(iter(z.values()))
    if isinstance(z, (tuple, list)):
        z = z[0]
    return z.flatten(1).float()


def measure(root, tag):
    out = {"constructed": False}
    try:
        model, cfg = build(root, tag)
    except Exception:
        out["error"] = traceback.format_exc().strip().splitlines()[-1]
        return out, None, None
    out["constructed"] = True
    out["params"] = sum(p.numel() for p in model.parameters())
    out["type"] = str((cfg.get("model") or {}).get("type") or "")
    out["source"] = str(getattr(model, "pretrained_url", "") or "") or (
        "timm/" + str(getattr(model, "timm_name", "")) if getattr(model, "timm_name", "") else "")
    try:
        model.eval()
        g = torch.Generator().manual_seed(a.seed)
        x = torch.randn(a.batch, 3, a.image_size, a.image_size, generator=g)
        x2 = x + 0.1 * torch.randn(x.shape, generator=g)
        with torch.no_grad():
            z1, z2 = features(model, x), features(model, x2)
        out["finite"] = bool(torch.isfinite(z1).all() and torch.isfinite(z2).all())
        out["feature_rms"] = float(z1.pow(2).mean().sqrt())
        n1, n2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        logits = n1 @ n2.t() / 0.1
        out["loss"] = float(F.cross_entropy(logits, torch.arange(a.batch)))
    except Exception:
        out["forward_error"] = traceback.format_exc().strip().splitlines()[-1]
    return out, model, cfg


def parent_state(model, cfg):
    train, mdl = cfg.get("train") or {}, cfg.get("model") or {}
    for path in (train.get("resume"), mdl.get("init_from")):
        if path and Path(path).is_file():
            blob = torch.load(path, map_location="cpu", weights_only=False)
            for key in ("student_backbone", "backbone", "model", "state_dict"):
                if isinstance(blob, dict) and isinstance(blob.get(key), dict):
                    blob = blob[key]
                    break
            return blob, str(path)
    return model.state_dict(), "parent_init"


def identity(cmodel, pmodel, state):
    """Child (loaded with the parent's state, its own new tensors at their init) vs parent (same state): the
    largest feature gap relative to the parent's feature rms, and both proxy losses -- 0 for an identity-at-init
    addition."""
    pmodel.load_state_dict(state, strict=False)
    cmodel.eval()
    pmodel.eval()
    g = torch.Generator().manual_seed(a.seed + 1)
    x = torch.randn(a.batch, 3, a.image_size, a.image_size, generator=g)
    x2 = x + 0.1 * torch.randn(x.shape, generator=g)
    with torch.no_grad():
        zc, zp = features(cmodel, x), features(pmodel, x)
        zc2, zp2 = features(cmodel, x2), features(pmodel, x2)
    if zc.shape != zp.shape:
        return {"shape": [list(zc.shape), list(zp.shape)]}
    rms = float(zp.pow(2).mean().sqrt()) or 1.0
    gap = max(float((zc - zp).abs().max()), float((zc2 - zp2).abs().max())) / rms
    loss = lambda u, v: float(F.cross_entropy(F.normalize(u, dim=1) @ F.normalize(v, dim=1).t() / 0.1,
                                              torch.arange(a.batch)))
    return {"rel_gap": gap, "loss_child": loss(zc, zc2), "loss_parent": loss(zp, zp2)}


#FETCH_SPY#


def swap_load(model):
    """An init swap: the child's own load_pretrained (absent: a random init), each load_state_dict it makes
    recorded, then a forward of the loaded model."""
    mod = sys.modules.get("loop_child_model")
    out = {"loader": hasattr(mod, "load_pretrained")}
    if not out["loader"]:
        return out
    seen, orig = [], model.load_state_dict
    out["fetches"] = fetches = []
    undo = spy_fetches(fetches, mod)

    def spy(state, *args, **kw):
        res = orig(state, *args, **kw)
        seen.append({"keys": len(state), "n_missing": len(res.missing_keys), "missing": list(res.missing_keys)[:10],
                     "n_unexpected": len(res.unexpected_keys), "unexpected": list(res.unexpected_keys)[:10]})
        return res
    model.load_state_dict = spy
    try:
        got = mod.load_pretrained(model)
    except Exception:
        out["error"] = traceback.format_exc().strip().splitlines()[-1]
        return out
    finally:
        model.__dict__.pop("load_state_dict", None)
        for restore in undo:
            restore()
    model = got if isinstance(got, torch.nn.Module) else model
    out["loads"] = seen
    try:
        model.eval()
        g = torch.Generator().manual_seed(a.seed)
        with torch.no_grad():
            z = features(model, torch.randn(a.batch, 3, a.image_size, a.image_size, generator=g))
        out["finite"] = bool(torch.isfinite(z).all())
    except Exception:
        out["error"] = traceback.format_exc().strip().splitlines()[-1]
    return out


child, cmodel, _ = measure(a.child_dir, "child")
parent, pmodel, pcfg = measure(a.parent_dir, "parent")
keys = {"checked": False}
if cmodel is not None and pmodel is not None and (child["type"], child["source"]) != (parent["type"], parent["source"]):
    try:
        keys = {"swap": True, **swap_load(cmodel)}
    except Exception:
        keys = {"swap": True, "loader": True, "error": traceback.format_exc().strip().splitlines()[-1]}
elif cmodel is not None and pmodel is not None:
    try:
        state, source = parent_state(pmodel, pcfg)
        res = cmodel.load_state_dict(state, strict=False)
        keys = {"checked": True, "source": source, "missing": list(res.missing_keys)[:20],
                "unexpected": list(res.unexpected_keys)[:20]}
        if res.missing_keys and not res.unexpected_keys:
            try:
                keys["identity"] = identity(cmodel, pmodel, state)
            except Exception:
                keys["identity"] = {"error": traceback.format_exc().strip().splitlines()[-1]}
    except Exception:
        keys = {"checked": True, "source": "error", "error": traceback.format_exc().strip().splitlines()[-1]}
print(json.dumps({"child": child, "parent": parent, "keys": keys}))
'''

SCRIPT = SCRIPT.replace("#FETCH_SPY#", FETCH_SPY)


def folded_type(model: Mapping[str, Any]) -> Any:
    """``model.type`` as the port's train.py folds it: ``type: timm`` + ``timm_name: <id>`` -> ``timm:<id>``."""
    t = model.get("type")
    return f"timm:{model['timm_name']}" if t == "timm" and model.get("timm_name") else t


def parse_measurements(stdout: str) -> Mapping[str, Any] | None:
    """The last JSON line of the script's stdout carrying child + parent, else None."""
    for line in reversed((stdout or "").splitlines()):
        try:
            obj = json.loads(line.strip()) if line.strip().startswith("{") else None
        except ValueError:
            continue
        if isinstance(obj, dict) and "child" in obj and "parent" in obj:
            return obj
    return None


def _within(child: float, parent: float, factor: float, floor: float = 1e-12) -> tuple[bool, float]:
    ratio = (abs(child) + floor) / (abs(parent) + floor)
    return math.isfinite(ratio) and 1.0 / factor <= ratio <= factor, ratio


def identical_at_init(ident: Any, tol: float) -> bool:
    """The child with the parent's state computes the parent's features (relative gap <= ``tol``) and proxy loss:
    its parent-missing tensors are an identity-at-init addition."""
    if not isinstance(ident, Mapping) or "error" in ident or "rel_gap" not in ident:
        return False
    gap, lc, lp = (float(ident.get(k, math.nan)) for k in ("rel_gap", "loss_child", "loss_parent"))
    return math.isfinite(gap) and gap <= tol and math.isfinite(lc) and abs(lc - lp) <= tol * max(1.0, abs(lp))


def denied(text: str, deny: tuple[str, ...]) -> str:
    """The first ``deny`` pattern ``text`` matches (case-insensitive), or ''."""
    return next((p for p in deny if re.search(p, text or "", re.I)), "")


def allowed(source: str, allow: tuple[str, ...]) -> bool:
    """The WEIGHTS ALLOWLIST (operator 2026-10-22: generic Meta DINOv2 / DINOv3 weights legal, everything else not).
    No allowlist configured, or no weights source at all (a random-init architecture): True. Else the source must match
    one ``allow`` regex (case-insensitive). ``deny`` is checked separately and always wins."""
    src = (source or "").strip()
    return not allow or not src or any(re.search(p, src, re.I) for p in allow)


def flop_budget(flops: Mapping[str, Any] | None, cap: float) -> dict[str, Any]:
    """``flops`` = {per_sample (None: the smoke printed none), max_samples, max_flops}; None = no smoke yet."""
    if not flops:
        return {"ok": True, "pending": "no smoke of this code / config yet: the smoke measures FLOPs per sample"}
    per, n, cap_flops = flops.get("per_sample"), flops.get("max_samples"), flops.get("max_flops")
    if not all(isinstance(x, (int, float)) and x > 0 for x in (per, n, cap_flops)):
        return {"ok": False, "detail": "the smoke printed no measured_flops_per_step", **dict(flops)}
    frac = per * n / cap_flops
    return {"ok": frac <= cap, "projected": round(frac, 4), "cap": cap, "per_sample": per, "max_samples": n,
            "max_flops": cap_flops}


def _swap(m: Mapping[str, Any], limits: Limits) -> InitCompat:
    child, parent, keys = m.get("child") or {}, m.get("parent") or {}, m.get("keys") or {}
    src, typ = str(child.get("source") or ""), str(child.get("type") or "")
    pat = denied(f"{src} {typ}", limits.deny)
    off_list = not pat and not allowed(src, limits.allow)       # a source outside the weights allowlist
    fetches = [str(f) for f in (keys.get("fetches") or [])]
    bad_fetch = next((f for f in fetches if denied(f, limits.deny) or not allowed(f, limits.allow)), "")
    checks: dict[str, dict[str, Any]] = {
        "construct": {"ok": bool(child.get("constructed")), "error": child.get("error", "")},
        "forward_finite": {"ok": bool(child.get("finite")) and "forward_error" not in child,
                           "error": child.get("forward_error", "")},
        "init_legal": {"ok": not pat and not off_list and not bad_fetch, "source": src, "type": typ,
                       "fetches": fetches, **({"bad_fetch": bad_fetch} if bad_fetch else {}),
                       **({"pattern": pat} if pat else {}), **({"not_allowlisted": True} if off_list else {})},
        "load": {"ok": "error" not in keys and (not keys.get("loader") or bool(keys.get("finite"))),
                 "loader": bool(keys.get("loader")), "error": keys.get("error", ""), "loads": keys.get("loads", [])},
        "flop_budget": flop_budget(m.get("flops"), limits.flop_cap)}
    failed = [n for n in SWAP_CHECKS if not checks[n]["ok"]]
    fb = checks["flop_budget"]
    if pat:
        reason = (f"init_illegal: the init {src or typ!r} matches the deny pattern {pat!r}: pathology-trained and "
                  f"GenBio checkpoints may not initialise or guide the model (any other source is allowed)")
    elif off_list:
        reason = (f"init_illegal: the init source {src!r} is not on the weights allowlist: allowlisted generic "
                  f"(non-pathology) weights: see init_compat.allow")
    elif bad_fetch:
        reason = (f"init_illegal: the loader fetched {bad_fetch!r} (hf_hub_download / timm pretrained / torch.hub), which is "
                  f"denied or not on the weights allowlist: allowlisted generic (non-pathology) weights: see init_compat.allow")
    elif not fb["ok"] and "projected" in fb:
        reason = (f"over_flop_cap: projected flop_fraction {fb['projected']:.3f} > {fb['cap']} at the full "
                  f"{fb['max_samples']:,} samples ({fb['per_sample']:.4g} FLOPs/sample measured by the smoke x "
                  f"{fb['max_samples']:,} / {fb['max_flops']:.3g}): the run would stop on the FLOP cap before the "
                  f"sample budget; cut per-sample compute until the projection is <= {fb['cap']}")
    else:
        reason = "init-swap check failed: " + "; ".join(
            f"{n} {json.dumps(checks[n], default=str)[:200]}" for n in failed)
    swap = {"from": {"type": parent.get("type"), "source": parent.get("source")}, "to": {"type": typ, "source": src}}
    return InitCompat(not failed, reason if failed else "", {**checks, "init_swap": swap})


def evaluate(m: Mapping[str, Any], limits: Limits = Limits()) -> InitCompat:
    """Apply :class:`Limits`; fail closed (a check that cannot be evaluated failed). An init swap (the script's
    ``keys.swap``) takes :data:`SWAP_CHECKS` instead of the parent-relative ones."""
    child, parent, keys = m.get("child") or {}, m.get("parent") or {}, m.get("keys") or {}
    if keys.get("swap"):
        return _swap(m, limits)
    built, pbuilt = bool(child.get("constructed")), bool(parent.get("constructed"))
    fwd = built and bool(child.get("finite")) and "forward_error" not in child
    checks: dict[str, dict[str, Any]] = {
        "construct": {"ok": built and pbuilt, "error": child.get("error", ""),
                      "parent_error": parent.get("error", "")},
        "forward_finite": {"ok": fwd, "error": child.get("forward_error", "")},
        "initial_loss": {"ok": False, "detail": "not measurable"},
        "param_ratio": {"ok": False, "detail": "not measurable"},
        "checkpoint_keys": {"ok": False, "detail": keys.get("error", "not measurable")}}
    if fwd and "loss" in parent and "feature_rms" in parent:
        lok, lr = _within(child["loss"], parent["loss"], limits.loss_factor, limits.loss_floor)
        fok, fr = _within(child["feature_rms"], parent["feature_rms"], limits.feature_factor)
        checks["initial_loss"] = {"ok": lok and fok and math.isfinite(child["loss"]),
                                  "loss_ratio": lr, "feature_rms_ratio": fr}
    if built and pbuilt and parent.get("params"):
        pr = child["params"] / parent["params"]
        checks["param_ratio"] = {"ok": limits.param_ratio_min <= pr <= limits.param_ratio_max, "ratio": pr}
    if keys.get("checked") and "error" not in keys:
        missing, unexpected = list(keys.get("missing") or []), list(keys.get("unexpected") or [])
        same = identical_at_init(keys.get("identity"), limits.identity_tol)
        checks["checkpoint_keys"] = {"ok": not unexpected and (not missing or same), "source": keys.get("source"),
                                     "missing": missing, "unexpected": unexpected}
        if missing:
            checks["checkpoint_keys"]["identity_at_init"] = {"ok": same, **(keys.get("identity") or {})}
    failed = [n for n in CHECKS if not checks[n]["ok"]]
    reason = "; ".join(f"{n} {json.dumps(checks[n], default=str)[:200]}" for n in failed)
    return InitCompat(not failed, f"init-compat failed: {reason}" if failed else "", checks)


def run_init_check(*, runner: Any, worktree_path: str, parent_commit: str, state_dir: str,
                   limits: Limits = Limits(), python: str = "python",
                   flops: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None) -> InitCompat:
    """Extract the parent's model/config, run :data:`SCRIPT` with ``python``, evaluate (``flops``: the smoke's
    measurement of this worktree, see :func:`flop_budget`). ``runner.run(argv, cwd=, timeout=) -> (rc, stdout,
    stderr)``."""
    work = Path(state_dir) / "initcompat" / parent_commit[:12]
    for rel in (limits.model_file, limits.config_rel):
        rc, out, err = runner.run(["git", "-C", worktree_path, "show", f"{parent_commit}:{rel}"], timeout=60)
        if rc != 0:
            return InitCompat(False, f"init-compat: cannot read {rel} at parent {parent_commit[:12]}: "
                                     f"{(err or out).strip()[-200:]}", {"construct": {"ok": False}})
        (work / "parent" / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / "parent" / rel).write_text(out)
    (work / "check_init.py").write_text(SCRIPT)
    rc, out, err = runner.run(
        [python, str(work / "check_init.py"), "--child-dir", worktree_path, "--parent-dir",
         str(work / "parent"), "--model-file", limits.model_file, "--config", limits.config_rel,
         "--image-size", str(limits.image_size), "--batch", str(limits.batch), "--seed", str(limits.seed)],
        cwd=worktree_path, timeout=limits.timeout_s, **({"env": dict(env)} if env else {}))
    measured = parse_measurements(out)
    if measured is None:
        return InitCompat(False, f"init-compat script produced no verdict (rc={rc}): "
                                 f"{(err or out).strip()[-300:]}", {"construct": {"ok": False}})
    return evaluate({**measured, "flops": flops}, limits)
