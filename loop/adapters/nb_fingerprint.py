"""loop.adapters.nb_fingerprint — the NO-OP KNOB FINGERPRINT of a nanobrain config (``intake.noop_fingerprint``).

Run as a SCRIPT under the project's python, inside a worktree (``python nb_fingerprint.py <worktree> <config> <out>``):
it imports the worktree's own ``nanobrain`` package, resolves the config exactly as ``nanobrain.train`` does
(default_pretrain.yaml unsafe-merged with the config; lr from base_lr; epoch-keyed step counts from the filelist
size), and records, on CPU in float32 with fixed seeds and one fixed synthetic batch:

- ``resolved``: the resolved config (sorted JSON);
- ``lr``: the LR schedule sampled at fixed steps / wall-clock fractions, and ``lr_final`` (the LR at the last step);
- ``loss0`` / ``grad0``: the loss and global grad norm of one forward / backward;
- ``loss1`` / ``param1``: after one optimizer step (grad clipped like training) at the step-1 LR, the loss on the same
  batch and the parameter sum.

Each part is computed independently; a part that raises is recorded under ``errors`` (never a crash). The loop
(``NanobrainAdapter.noop_fingerprint``) compares the parent's and the child's parts: identical within tolerance on
every part both computed => the change is numerically inert. Nothing here touches data, GPUs or the network.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path


def _lines(path: Path) -> int:
    with open(path) as fh:
        return sum(1 for ln in fh if ln.strip())


def main(worktree: str, config: str, out: str) -> int:
    import os
    t0 = time.monotonic()
    os.environ.setdefault("BRAINMARKS_SMRI_DATA", "/nonexistent-nb-fingerprint")   # read at import; never used here
    os.environ.setdefault("WANDB_MODE", "disabled")
    wt = Path(worktree).resolve()
    sys.path.insert(0, str(wt / "src"))
    res: dict = {"errors": {}}
    try:
        import torch
        from omegaconf import OmegaConf
        torch.set_num_threads(int(os.environ.get("NB_FP_THREADS", "8")))
        torch.use_deterministic_algorithms(True, warn_only=True)
        default = wt / "src" / "nanobrain" / "config" / "default_pretrain.yaml"
        cfg = OmegaConf.unsafe_merge(OmegaConf.load(default), OmegaConf.load(config))
        tb = cfg.batch_size * cfg.num_samples * cfg.get("accum_iter", 1)
        if not cfg.get("lr"):
            cfg.lr = cfg.base_lr * tb / 256
        if cfg.get("epochs"):
            try:
                spe = _lines(wt / str(cfg.train_filelist)) / (cfg.batch_size * cfg.get("accum_iter", 1))
            except OSError:
                spe = 1000.0
            cfg.total_steps = round(cfg.epochs * spe)
            cfg.warmup_steps = round(cfg.warmup_epochs * spe)
            cfg.decay_steps = round(cfg.decay_epochs * spe)
        c = OmegaConf.to_container(cfg, resolve=True)
        for k in ("name", "notes", "output_dir", "wandb", "seed"):
            c.pop(k, None)
        res["resolved"] = json.dumps(c, sort_keys=True, default=str)
    except Exception as exc:  # noqa: BLE001
        res["errors"]["resolve"] = repr(exc)[:400]
        Path(out).write_text(json.dumps(res))
        return 0
    try:
        import nanobrain.train as T
        if cfg.get("time_schedule"):
            limit = float(cfg.time_limit) * 60.0
            pts = [(s, f * limit) for s in (0, 1, 10, 100, 1000) for f in (0.0,)] + \
                  [(10 ** 6, f * limit) for f in (0.1, 0.5, 0.79, 0.81, 0.9, 0.95, 0.99, 1.0)]
            res["lr"] = [float(T.time_keyed_lr(cfg, s, w, limit, 0.0)) for s, w in pts]
        else:
            sch = T.make_lr_schedule(cfg.lr, cfg.min_lr, cfg.total_steps, cfg.warmup_steps, cfg.decay_steps)
            idx = sorted({0, 1, 10, 100, 1000, len(sch) // 4, len(sch) // 2, 3 * len(sch) // 4, len(sch) - 1})
            res["lr"] = [float(sch[min(i, len(sch) - 1)]) for i in idx]
        res["lr_final"] = res["lr"][-1]
        lr1 = res["lr"][1] if res["lr"][1] > 0 else float(cfg.lr)
    except Exception as exc:  # noqa: BLE001
        res["errors"]["lr"] = repr(exc)[:400]
        lr1 = float(cfg.lr)
    try:
        import torch
        from nanobrain.model import ViTMAE3D
        torch.manual_seed(0)
        model = ViTMAE3D.from_config(OmegaConf.to_container(cfg, resolve=True)).float()
        model.train()
        g = torch.Generator().manual_seed(1234)
        B = 1
        if cfg.native:
            shape, sp = (96, 120, 96), (2.0, 2.0, 2.0)
        else:
            shape, sp = tuple(cfg.grid_size), None
        xs = [torch.linspace(-1, 1, n) for n in shape]
        X, Y, Z = torch.meshgrid(*xs, indexing="ij")
        brain = (X ** 2 / 0.8 + Y ** 2 / 0.9 + Z ** 2 / 0.7) < 1.0
        imgs = [(torch.randn(shape, generator=g) * 0.5 + torch.cos(3 * X + k) * torch.sin(2 * Y) + Z) * brain
                for k in range(B)]
        masks = [brain.clone() for _ in range(B)]
        kw = dict(num_visible=cfg.num_visible, num_predict=cfg.num_predict, num_samples=cfg.num_samples,
                  with_state=False)
        if cfg.native:
            batch = (imgs, masks)
            kw["spacing"] = [sp] * B
        else:
            batch = (torch.stack(imgs), torch.stack(masks).float())

        def loss_of() -> "torch.Tensor":
            torch.manual_seed(7)
            return model(*batch, **kw)
        loss = loss_of()
        loss.backward()
        res["loss0"] = float(loss.detach())
        res["grad0"] = float(math.sqrt(sum(float((p.grad.double() ** 2).sum())
                                           for p in model.parameters() if p.grad is not None)))
        try:
            import nanobrain.misc as misc
            import nanobrain.train as T
            opt = T.make_optimizer(cfg, model)
            misc.update_lr(getattr(opt, "param_groups", []), lr1)
            if cfg.get("clip_grad"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.clip_grad))
            opt.step()
            model.zero_grad(set_to_none=True)
            res["param1"] = float(sum(float(p.detach().double().sum()) for p in model.parameters()))
            with torch.no_grad():
                res["loss1"] = float(loss_of())
        except Exception as exc:  # noqa: BLE001
            res["errors"]["step"] = repr(exc)[:400]
    except Exception as exc:  # noqa: BLE001
        res["errors"]["forward"] = repr(exc)[:400]
    res["seconds"] = round(time.monotonic() - t0, 1)
    Path(out).write_text(json.dumps(res))
    return 0


PARTS = ("resolved", "lr", "loss0", "grad0", "param1", "loss1")


def same(a: dict, b: dict, rtol: float = 1e-6) -> tuple[bool, list[str]]:
    """(identical, the parts compared). Fails OPEN: no numeric part on both sides -> not identical."""
    seen, num = [], 0
    for k in PARTS:
        if k not in a or k not in b:
            continue
        seen.append(k)
        x, y = a[k], b[k]
        if k == "resolved":
            continue                       # the configs differ by construction; the numbers decide
        xs, ys = (x, y) if isinstance(x, list) else ([x], [y])
        if len(xs) != len(ys):
            return False, seen
        for u, v in zip(xs, ys):
            if not (math.isfinite(u) and math.isfinite(v)) or abs(u - v) > rtol * max(1.0, abs(u), abs(v)):
                return False, seen
        num += 1
    return num >= 3 and "loss0" in seen and "grad0" in seen, seen


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
