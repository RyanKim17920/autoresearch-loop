"""loop.adapters.feature_diag — the researcher's FEATURE DIAGNOSTIC job (``diagnose(what="features")``, v3-c4i).

Runs INSIDE a short SLURM job (primary pool) in the incumbent's node worktree venv (stdlib + the worktree's torch /
yaml; it never imports ``loop``). It loads one of the incumbent's finished checkpoints exactly as the locked
``probe.py`` worker does (the same loading as ``loop.adapters.probe_smoke``) and, READ-ONLY, uses ``probe.py``'s own
datasets and transforms: for each configured classification dataset it embeds the first N val tiles (the manifest
split is truncated IN MEMORY, as the probe smoke does; probe.py is never modified) through

* ``model.probe_features`` (what every classification head is fitted on), and
* the plain forward's final ``cls`` token,

under the probe's fp16 autocast, and reports per dataset: the norms of both, their cosine on the shared width, and
every per-tile scalar a ``sigmoid`` produced inside ``probe_features`` (a gate: its mean / p10 / p50 / p90) --
captured generically by a ``TorchFunctionMode``, so no readout internals are named here.

    python feature_diag.py <worktree> <config yaml> <checkpoint> <out.json> [--n=32] [--device=cuda]
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

N_TILES = 32
MARK = "feature_diag:"


def _bound(spec: dict, n: int) -> None:
    for key in ("images", "labels", "indices"):
        if isinstance(spec.get(key), list):
            spec[key] = spec[key][:n]


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def _stats(xs: list[float]) -> dict:
    return {"n": len(xs), "mean": sum(xs) / len(xs) if xs else None, "p10": _q(xs, 0.1), "p50": _q(xs, 0.5),
            "p90": _q(xs, 0.9)}


def run(worktree: str, cfg_path: str, ckpt_path: str, device_name: str = "cuda", n: int = N_TILES) -> dict:
    os.chdir(worktree)
    sys.path.insert(0, worktree)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import importlib
    from pathlib import Path

    import torch
    import torch.nn.functional as F
    import yaml
    from torch.overrides import TorchFunctionMode

    import probe                                   # the locked probe.py of the checkout, unmodified
    from model import ViT

    cfg = yaml.safe_load(Path(cfg_path).read_text())
    pcfg = cfg["probe"]
    probe.DATASET_ROOTS.clear()
    probe.DATASET_ROOTS.update({k: Path(v) for k, v in pcfg["dataset_roots"].items()})
    device = torch.device(device_name)
    if pcfg.get("model_loader"):
        module, fn = pcfg["model_loader"].split(":")
        model = getattr(importlib.import_module(module), fn)(ckpt_path, device)
    else:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ViT(variant=cfg["model"]["type"]).to(device).eval()
        model.load_state_dict(ck[{"ema": "model_ema", "model": "model"}[str(pcfg.get("model_weights", "ema"))]],
                              strict=True)
        del ck
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)
    transform, _ = probe.worker_probe_transforms(cfg)
    sig = {torch.sigmoid, torch.Tensor.sigmoid, F.sigmoid}

    class Gates(TorchFunctionMode):
        """Every per-tile scalar (shape (B,) or (B, 1)) a sigmoid returns while it is active."""

        def __init__(self, b: int) -> None:
            super().__init__()
            self.b, self.got = b, []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if func in sig and isinstance(out, torch.Tensor) and out.shape[:1] == (self.b,) and out.numel() == self.b:
                self.got.append(out.detach().float().reshape(-1).cpu())
            return out

    use_amp = device.type == "cuda"
    report: dict = {"datasets": {}, "n_tiles": n}
    for ds in list(pcfg.get("datasets") or []):
        t = time.monotonic()
        _bound(probe.THUNDER_V2["classification"][ds]["val"], n)
        tiles = probe.ClassificationDataset(ds, "val", transform)
        k = min(len(tiles), n)
        if k == 0:
            continue
        x = torch.stack([tiles[i][0] for i in range(k)]).to(device)
        xn = (x - mean) / std
        cap = Gates(k)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            with cap:
                pf = model.probe_features(xn)
            raw = model(xn)["cls"]
        pf, raw = pf.float(), raw.float()
        w = min(pf.shape[-1], raw.shape[-1])
        gates = [g for g in cap.got]
        report["datasets"][ds] = {
            "tiles": k, "probe_features_dim": int(pf.shape[-1]), "cls_dim": int(raw.shape[-1]),
            "norm_probe_features": _stats(pf.norm(dim=-1).tolist()),
            "norm_probe_features_head": _stats(pf[:, :w].norm(dim=-1).tolist()),
            "norm_cls": _stats(raw.norm(dim=-1).tolist()),
            "cos_head_vs_cls": _stats(F.cosine_similarity(pf[:, :w], raw[:, :w], dim=-1).tolist()),
            "gates": [_stats(g.tolist()) for g in gates][:4],
            "seconds": round(time.monotonic() - t, 1)}
        print(f"{MARK} {ds}: {k} tiles, {len(gates)} gate(s)", flush=True)
    return report


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in argv[1:] if a.startswith("--") and "=" in a)
    if len(args) != 4:
        print(f"{MARK} usage: feature_diag.py <worktree> <config> <checkpoint> <out.json> [--n=32] [--device=cuda]",
              file=sys.stderr)
        return 2
    wt, cfg, ckpt, out = args
    try:
        rep = run(wt, cfg, ckpt, opts.get("device", "cuda"), int(opts.get("n", N_TILES)))
    except BaseException as exc:                   # noqa: BLE001 - the failure is the report
        traceback.print_exc()
        rep = {"error": f"{type(exc).__name__}: {str(exc)[:600]}"}
    if "error" in rep:            # a failure is never the cached report: loop.agents.diagnose shows it, then resubmits
        out = os.path.join(os.path.dirname(out), "features.error.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rep, f, indent=1)
    os.replace(tmp, out)
    print(f"{MARK} wrote {out}", flush=True)
    return 0 if "error" not in rep else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
