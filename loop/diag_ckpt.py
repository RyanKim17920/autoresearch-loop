"""loop.diag_ckpt -- STANDALONE checkpoint + metrics diagnostics (torch + stdlib). NOT imported by loop code: the
loop launches it as a subprocess under a torch-capable python (loop.diagnostics.launch).

    python diag_ckpt.py --run-dir <output_dir> --init <init.pth> --out <json> [--run-id ID] [--threads 4]
                        [--model-dir <the node's worktree>]

The init the drift features are measured from is the run's OWN (an init swap trains from another checkpoint):
with ``--model-dir``, the checkpoint the node's model.py declares for the run's ``model.type``
(``ViT(variant=...).pretrained_url``) in the torch hub cache -- none (no drift features) when it is not cached;
``--init`` when the node's code is unavailable.

Reads ``<run-dir>/latest.pt`` (mmap) and ``<run-dir>/metrics.jsonl`` and writes
``{"run_id"?, "features": {...}, "version": 1, "secs": t}``: ~100 floats. A missing piece omits its keys; nothing
here raises past ``main`` (a section that fails is skipped).

Feature keys (block index ``i``):
  b{i}_tdrift / b{i}_st            teacher drift from init ||W-W0||/||W0|| / student-teacher distance (matrix weights)
  b{i}_grms                        sqrt(v_hat) RMS of the AdamW second moment over the block's 4 matrices (grad RMS)
  b{i}_upd                         lr * rms(m_hat/(sqrt(v_hat)+eps)) / rms(W): the per-step update/weight ratio
  b{i}_lr_end                      the optimizer group's lr at the checkpoint
  b{i}_{qkv,fc2}_{erank_d,sigma1_r}  teacher effective-rank change / sigma1 ratio vs init (first, middle, last block)
  patch_embed_drift, pos_embed_drift, cls_token_drift, ...   teacher drift of the embeddings
  n_blocks, step, train_flops, examples_seen, clip_grad, layerwise_decay
  lr_peak, lr_end_frac, flop_frac_end, gn_early, gn_late, gn_late_over_clip, gn_clip_frac,
  val_<k>_min, val_<k>_end, val_<k>_rise, val_<k>_argmin_frac
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

VERSION = 1
SUBS = ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")
SHORT = {"attn.qkv.weight": "qkv", "attn.proj.weight": "proj", "mlp.fc1.weight": "fc1", "mlp.fc2.weight": "fc2"}
EMBED_KEYS = ("patch_embed.proj.weight", "pos_embed", "cls_token", "register_tokens", "norm.weight")
LATE_FRAC = 0.25
EARLY_FRAC = 0.10
SVD_BLOCKS = 3                                   # first, middle, last


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _put(f: dict, key: str, val) -> None:
    v = _num(val)
    if v is not None:
        f[key] = v


# ------------------------------------------------------------------ metrics.jsonl
def metric_features(run_dir: str, clip: float | None) -> dict:
    f: dict = {}
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.isfile(path):
        return f
    train, val = [], []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "event" in d:
                continue
            if "grad_norm" in d and "lr" in d:
                train.append(d)
            elif any(k.startswith("val_") for k in d):
                val.append(d)
    if train:
        n = len(train)
        lrs = [_num(d["lr"]) or 0.0 for d in train]
        peak = max(lrs)
        f["n_train_logs"] = float(n)
        _put(f, "lr_peak", peak)
        if peak > 0:
            f["lr_end_frac"] = lrs[-1] / peak
        _put(f, "flop_frac_end", train[-1].get("flop_fraction"))
        gn = [g for g in (_num(d["grad_norm"]) for d in train) if g is not None]
        if gn:
            late = gn[int(len(gn) * (1 - LATE_FRAC)):] or gn[-1:]
            early = gn[:max(1, int(len(gn) * EARLY_FRAC))]
            f["gn_late"], f["gn_early"] = sum(late) / len(late), sum(early) / len(early)
            if clip and clip > 0:
                f["gn_late_over_clip"] = f["gn_late"] / clip
                f["gn_clip_frac"] = sum(g > clip for g in gn) / len(gn)
                f["gn_clip_frac_late"] = sum(g > clip for g in late) / len(late)
    if len(val) >= 2:
        steps = [_num(d.get("step")) or 0.0 for d in val]
        smax = max(steps) or 1.0
        for k in sorted({k for d in val for k in d if k.startswith("val_")}):
            pts = [(s, _num(d.get(k))) for s, d in zip(steps, val)]
            pts = [(s, v) for s, v in pts if v is not None]
            if len(pts) < 3:
                continue
            vs = [v for _, v in pts]
            i = min(range(len(vs)), key=vs.__getitem__)
            tail = vs[-3:]
            f[f"{k}_min"], f[f"{k}_end"] = vs[i], sum(tail) / len(tail)
            f[f"{k}_rise"] = f[f"{k}_end"] - vs[i]
            f[f"{k}_argmin_frac"] = pts[i][0] / smax
    return f


# ------------------------------------------------------------------ checkpoint
def _erank(torch, w):
    s = torch.linalg.svdvals(w.float())
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum())), float(s[0])


def _block_params(opt: dict, shapes: tuple, depth: int, ld) -> dict[int, list]:
    """block -> [(param id, group), ...] of its 4 matrices (qkv, proj, fc1, fc2), found by SHAPE in the param
    groups: consecutive params whose state shapes are the 4 block matrices. With layerwise decay (<1) the block is
    read off the group's lr_mult (a lone group of 4); otherwise (or in the lr_mult 1.0 group that also holds the
    heads) tuples are counted in order from the first match."""
    state, out, seq = opt["state"], {}, 0
    for g in opt["param_groups"]:
        if g.get("last_layer") or g.get("wd_mult", 1.0) != 1.0:
            continue
        ps = list(g["params"])
        shp = [tuple(state[p]["exp_avg"].shape) if p in state and "exp_avg" in state[p] else None for p in ps]
        j = 0
        while j + 4 <= len(ps):
            if tuple(shp[j:j + 4]) == shapes:
                if ld and 0 < ld < 1:
                    lm = float(g.get("lr_mult", 1.0))
                    blk = depth - 1 - round(math.log(lm) / math.log(ld))
                else:
                    blk, seq = seq, seq + 1
                if 0 <= blk < depth and blk not in out:      # a repeat (head / predictor tuple) is not the backbone
                    out[blk] = [(ps[j + k], g) for k in range(4)]
                j += 4
            else:
                j += 1
    return out


def resolve_init(torch, model_dir: str, cfg: dict, default: str) -> str:
    """The init file of a run with config ``cfg``: the checkpoint the node's model.py (in ``model_dir``) declares for
    its ``model.type``, from the torch hub cache ('' when not cached or not constructible); ``default`` without the
    node's code."""
    typ, path = (cfg.get("model") or {}).get("type"), os.path.join(model_dir or "", "model.py")
    if not typ or not os.path.isfile(path):
        return default
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location("diag_node_model", path)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, model_dir)
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path.remove(model_dir)
        with torch.device("meta"):
            url = str(getattr(mod.ViT(variant=typ), "pretrained_url", "") or "")
    except Exception:                                             # noqa: BLE001 -- its init is unknown: no drift
        return ""
    if not url or os.path.isfile(url):
        return url
    cached = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(url.split("?")[0]))
    return cached if os.path.isfile(cached) else ""


def checkpoint_features(torch, run_dir: str, init_path: str, model_dir: str = "") -> dict:
    f: dict = {}
    path = os.path.join(run_dir, "latest.pt")
    if not os.path.isfile(path):
        return f
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    cfg = ck.get("config") or {}
    init_path = resolve_init(torch, model_dir, cfg, init_path)
    dino = cfg.get("dino") or {}
    for k in ("step", "train_flops", "examples_seen"):
        _put(f, k, ck.get(k))
    _put(f, "clip_grad", dino.get("clip_grad"))
    _put(f, "layerwise_decay", dino.get("layerwise_decay"))
    _put(f, "cfg_lr", dino.get("lr"))
    S, T = ck.get("model") or {}, ck.get("model_ema") or {}
    W0 = {}
    if init_path and os.path.isfile(init_path):
        W0 = torch.load(init_path, map_location="cpu", weights_only=False, mmap=True)
        if isinstance(W0, dict) and "model" in W0 and isinstance(W0["model"], dict):
            W0 = W0["model"]
    depth = 0
    while f"blocks.{depth}.attn.qkv.weight" in S:
        depth += 1
    f["n_blocks"] = float(depth)

    def rel(a, b, ref):
        return float((a.float() - b.float()).norm() / ref.float().norm())

    for name in EMBED_KEYS:                                       # embeddings vs init
        try:
            if name in S and name in W0 and S[name].shape == W0[name].shape:
                key = name.split(".")[0]
                f[f"{key}_drift"] = rel(T.get(name, S[name]), W0[name], W0[name])
                f[f"{key}_sdrift"] = rel(S[name], W0[name], W0[name])
        except Exception:                                         # noqa: BLE001
            pass

    svd_at = sorted({0, depth // 2, depth - 1}) if depth else []
    for i in range(depth):                                        # weights: drift, student-teacher, SVD
        try:
            dt = ds = w0n = 0.0
            for sub in SUBS:
                k = f"blocks.{i}.{sub}"
                if k not in W0 or k not in S or S[k].shape != W0[k].shape:
                    continue
                w0, t = W0[k].float(), (T[k] if k in T else S[k]).float()
                s = S[k].float()
                dt += float((t - w0).norm() ** 2)
                ds += float((s - t).norm() ** 2)
                w0n += float(w0.norm() ** 2)
                if i in svd_at and sub in ("attn.qkv.weight", "mlp.fc2.weight"):
                    er, s1 = _erank(torch, t)
                    er0, s10 = _erank(torch, w0)
                    f[f"b{i}_{SHORT[sub]}_erank_d"], f[f"b{i}_{SHORT[sub]}_sigma1_r"] = er - er0, s1 / s10
            if w0n > 0:
                f[f"b{i}_tdrift"], f[f"b{i}_st"] = math.sqrt(dt / w0n), math.sqrt(ds / w0n)
        except Exception:                                         # noqa: BLE001
            continue

    try:                                                          # optimizer: grad RMS and update/weight per block
        opt = ck["opt"]
        shapes = tuple(tuple(S[f"blocks.0.{s}"].shape) for s in SUBS)
        betas = (opt["param_groups"][0].get("betas") or (0.9, float(dino.get("adam_beta2", 0.999))))
        b1, b2 = float(betas[0]), float(betas[1])
        eps = float(opt["param_groups"][0].get("eps", 1e-8))
        ld = dino.get("layerwise_decay")
        for blk, plist in _block_params(opt, shapes, depth, ld).items():
            gs, us, ws = [], [], []
            lr = None
            for j, (p, g) in enumerate(plist):
                st = opt["state"].get(p)
                if st is None:
                    continue
                stp = float(st["step"])
                v, m = st["exp_avg_sq"].float() / (1 - b2 ** stp), st["exp_avg"].float() / (1 - b1 ** stp)
                gs.append(float(v.mean()))
                us.append(float(((m / (v.sqrt() + eps)) ** 2).mean()))
                ws.append(float((S[f"blocks.{blk}.{SUBS[j]}"].float() ** 2).mean()))
                lr = float(g.get("lr", 0.0))
            if gs and lr is not None:
                f[f"b{blk}_grms"] = math.sqrt(sum(gs) / len(gs))
                f[f"b{blk}_lr_end"] = lr
                if sum(ws) > 0:
                    f[f"b{blk}_upd"] = lr * math.sqrt(sum(us) / len(us)) / math.sqrt(sum(ws) / len(ws))
    except Exception:                                             # noqa: BLE001
        pass
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--model-dir", default="")
    a = ap.parse_args(argv)
    t0 = time.time()
    feats: dict = {}
    clip = None
    try:
        import torch
        torch.set_num_threads(max(1, a.threads))
        feats.update(checkpoint_features(torch, a.run_dir, a.init, a.model_dir))
        clip = feats.get("clip_grad")
    except Exception:                                             # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
    try:
        feats.update(metric_features(a.run_dir, clip))
    except Exception:                                             # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
    doc = {"features": feats, "version": VERSION, "secs": time.time() - t0}
    if a.run_id:
        doc["run_id"] = a.run_id
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = f"{a.out}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, sort_keys=True)
    os.replace(tmp, a.out)
    return 0 if feats else 1


if __name__ == "__main__":
    sys.exit(main())
