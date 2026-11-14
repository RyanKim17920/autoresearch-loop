"""loop.effective -- the EFFECTIVE resolved config of a task (``python -m loop effective-config --task T``).

One JSON document: the parsed ``loop:`` block (every default filled in), the per-process project registry it installs
(axes, probe keys, weights, target / guard, noise floor), the spec's noise model, the LLM backend and the adapter's
class. Two dumps diff cleanly (sorted keys), so a change to a generic default shows, per task, exactly which value
moved. Read-only: no state dir is created, nothing is written."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import tempfile
from typing import Any, Sequence


def _clean(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return repr(x)
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x if isinstance(x, (str, int, float, bool, type(None))) else repr(x)


def effective(task: str) -> dict[str, Any]:
    from loop import project as P
    from loop.spec import encode, load_context
    with tempfile.TemporaryDirectory() as tmp:
        ctx = load_context(task, state_dir=tmp, require_measured=False)
    nm = ctx.noise
    s = P._S
    registry = {"axes": list(s.axes), "probe_keys": dict(s.probe_keys), "weights": dict(s.weights),
                "names": dict(s.names), "target": s.target, "guard": list(s.guard), "harm_h": dict(s.harm_h),
                "axis_exclude": list(s.axis_exclude), "composite": s.composite,
                "throughput": [s.throughput_key, s.throughput_tol], "noise_floor_seeds": s.noise_floor_seeds,
                "cell_tau2": s.cell_tau2, "noise_floor": dict(s.noise_floor)}
    cfg = encode(ctx.cfg)
    cfg.pop("state_dir", None)
    return _clean({"task": ctx.task_id, "adapter": ctx.cfg.execution_adapter,
                   "llm_default_backend": ctx.spec.llm_default_backend, "cfg": cfg, "registry": registry,
                   "noise": {f.name: getattr(nm, f.name) for f in dataclasses.fields(nm)}})


def flat(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict) and d:
        out: dict[str, Any] = {}
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): d}


def diff(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """``key: before -> after`` lines for every flattened key that differs (added / removed included)."""
    a, b = flat(before), flat(after)
    miss = "<absent>"
    return [f"{k}: {json.dumps(a.get(k, miss))} -> {json.dumps(b.get(k, miss))}"
            for k in sorted(set(a) | set(b)) if a.get(k, miss) != b.get(k, miss)]


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m loop effective-config")
    ap.add_argument("--task", required=True)
    ap.add_argument("--diff", default=None, help="a previous dump (JSON): print only the changed keys")
    args = ap.parse_args(list(argv))
    cur = effective(args.task)
    if args.diff:
        with open(args.diff) as fh:
            print("\n".join(diff(json.load(fh), cur)) or "(no change)")
    else:
        json.dump(cur, sys.stdout, indent=1, sort_keys=True)
        print()
    return 0
