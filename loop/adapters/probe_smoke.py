"""loop.adapters.probe_smoke — the PROBE leg of a nanopath smoke (NanopathV2Adapter.smoke, ``slurm.smoke_probe``).

Runs INSIDE the smoke's SLURM job, after the training smoke, in the worktree's own venv (stdlib + the worktree's
torch / yaml only: it never imports ``loop``). It loads the smoke's end-of-run ``latest.pt`` exactly as the locked
``probe.py`` worker does (``run_probe_job``: deterministic algorithms, ViT(variant) strict-loading the probe weights,
``worker_probe_transforms``) and drives the candidate's readout through ``probe.py``'s OWN entry points on a few
real probe tiles per dataset:

* ``probe.embed_classification_dataset`` on every configured classification dataset's val split (``probe_features``
  under the probe's fp16 autocast -- where live a node died, ``linalg.eigh`` in a covariance readout, after 2.4 GPU-h);
* ``probe.embed_segmentation_dataset`` on every configured manifest-backed segmentation set (``encode_image``);
* the slide / AUC / survival tile call (``probe_features`` on the patch transform) and the robustness call
  (``model(x)`` -> ``cls`` / ``patches``, inline_pathorob's two lines) on one real tile batch.

Bounding: each manifest split is truncated IN MEMORY (``probe.THUNDER_V2``, a module dict) to N_TILES entries; no
file is written and probe.py is never modified (it is locked). Every embedding must be finite. A failure prints the
traceback plus one ``probe_smoke: FAILED <Type>: <msg>`` line on stderr and exits 1, which fails the smoke job, so
the traceback lands in the smoke's log_tail (loop.design.smoke_done / the researcher treat it as a crash).

    python probe_smoke.py <worktree> <smoke config yaml> <latest.pt> [--device cuda]
"""

from __future__ import annotations

import os
import sys
import time
import traceback

N_TILES = 64                    # per manifest split: 1-2 loader batches' worth of real tiles, seconds per dataset
MARK = "probe_smoke:"
FAILED = f"{MARK} FAILED"
SKIP_SEG = ("pannuke",)         # an .npy split (np.load mmap): no manifest list to bound; segpath covers encode_image


def _finite(name: str, value) -> None:
    import numpy as np
    import torch
    arr = value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if arr.ndim < 2 or arr.shape[0] == 0:
        raise ValueError(f"{name}: readout returned shape {tuple(arr.shape)}, expected (tiles, features)")
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise FloatingPointError(f"{name}: {bad} non-finite readout values of {arr.size} (shape {tuple(arr.shape)})")


def _bound(spec: dict, n: int) -> None:
    """Truncate a manifest split's parallel lists (images / labels / indices) to their first ``n`` entries."""
    for key in ("images", "labels", "indices"):
        if isinstance(spec.get(key), list):
            spec[key] = spec[key][:n]


def run(worktree: str, cfg_path: str, ckpt_path: str, device_name: str = "cuda", n: int = N_TILES) -> list[str]:
    os.chdir(worktree)
    sys.path.insert(0, worktree)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"            # as run_probe_job
    import importlib
    import random
    from pathlib import Path

    import torch
    import yaml

    # the locked probe.py of the checkout (cwd), unmodified
    import probe
    from model import ViT

    random.seed(probe.THUNDER_PROBE_SEED)
    torch.manual_seed(probe.THUNDER_PROBE_SEED)
    torch.cuda.manual_seed_all(probe.THUNDER_PROBE_SEED)
    torch.use_deterministic_algorithms(True)
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
        model.load_state_dict(ck[{"ema": "model_ema", "model": "model"}[str(pcfg["model_weights"])]], strict=True)
        del ck
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)
    transform, patch_transform = probe.worker_probe_transforms(cfg)
    done: list[str] = []

    seg = [d for d in pcfg.get("segmentation_datasets") or [] if d not in SKIP_SEG]
    for ds in seg:                                                # first, as in run_probe_job
        _bound(probe.THUNDER_V2["segmentation"][ds]["val"], n)
        t = time.monotonic()
        feats, scales, _ = probe.embed_segmentation_dataset(model, mean, std, ds, "val", device, patch_transform)
        _finite(f"encode_image[{ds}] scales", scales)
        done.append(f"seg:{ds} {tuple(feats.shape)} {time.monotonic() - t:.1f}s")
        print(f"{MARK} encode_image {ds}: feats {tuple(feats.shape)}", flush=True)
    classification = list(pcfg.get("datasets") or [])
    for ds in classification:
        _bound(probe.THUNDER_V2["classification"][ds]["val"], n)
        t = time.monotonic()
        embs, _ = probe.embed_classification_dataset(model, mean, std, ds, "val", device, transform)
        _finite(f"probe_features[{ds}]", embs)
        done.append(f"cls:{ds} {tuple(embs.shape)} {time.monotonic() - t:.1f}s")
        print(f"{MARK} probe_features {ds}: {tuple(embs.shape)}", flush=True)
    if classification:              # slide / auc / survival tiles (patch transform) and pathorob's forward
        tiles = probe.ClassificationDataset(classification[0], "val", patch_transform)
        x = torch.stack([tiles[i][0] for i in range(min(len(tiles), 16))]).to(device)
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
        with torch.no_grad(), autocast:
            pf = model.probe_features((x - mean) / std)
            o = model((x - mean) / std)
            feat = torch.cat([o["cls"], o["patches"].mean(dim=1)], dim=-1)
        _finite("probe_features[patch transform]", pf)
        _finite("forward[cls+patches]", feat)
        done.append(f"patch:{tuple(pf.shape)} forward:{tuple(feat.shape)}")
        print(f"{MARK} probe_features (patch transform) {tuple(pf.shape)}; forward cls+patches {tuple(feat.shape)}",
              flush=True)
    return done


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--device")]
    device = next((a.split("=", 1)[1] for a in argv[1:] if a.startswith("--device=")), "cuda")
    if len(args) != 3:
        print(f"{FAILED} usage: probe_smoke.py <worktree> <config> <latest.pt> [--device=cuda]", file=sys.stderr)
        return 2
    started = time.monotonic()
    try:
        done = run(*args, device_name=device)
    except BaseException as exc:                                  # noqa: BLE001 — every failure is the verdict
        traceback.print_exc()
        print(f"{FAILED} {type(exc).__name__}: {str(exc)[:400]}", file=sys.stderr, flush=True)
        return 1
    print(f"{MARK} ok, {len(done)} readout calls in {time.monotonic() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
