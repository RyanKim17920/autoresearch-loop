"""loop.adapters — ExecutionAdapter implementations: ``sim.SimAdapter`` (offline
simulator), ``nanopath_v2.NanopathV2Adapter`` (git worktrees + the repo's own sbatch
launcher) and ``initcompat`` (the hard init-compat check). The package init stays light:
nothing here imports the root ``config`` (D0.4)."""

from __future__ import annotations

from typing import Any, Mapping


def flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested mapping -> ``{"a.b.c": leaf}``; lists and empty mappings are leaves."""
    out: dict[str, Any] = {}
    for key, val in tree.items():
        out.update(flatten(val, f"{prefix}{key}.") if isinstance(val, Mapping) and val
                   else {f"{prefix}{key}": val})
    return out


def set_dotted(tree: dict[str, Any], key: str, value: Any) -> None:
    """``set_dotted(d, "train.lr", 1)`` creates intermediate dicts as needed."""
    *parents, leaf = key.split(".")
    for part in parents:
        if not isinstance(tree.get(part), dict):
            tree[part] = {}
        tree = tree[part]
    tree[leaf] = value


MISSING = object()                 # merge_config: a key the side lacks (a None value is a real YAML null)


def subsumed(base: Any, theirs: Any, ours: Any) -> bool:
    """``ours`` already holds ``theirs``' move from ``base``: a number moved the same way at least as far (a node's lr
    1.25e-4 -> 1.875e-4 is inside a node's 2.344e-4)."""
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)     # noqa: E731
    if not (num(base) and num(theirs) and num(ours)) or theirs == base:
        return False
    return (theirs - base) * (ours - base) > 0 and abs(ours - base) >= abs(theirs - base)


def merge_config(base: Mapping[str, Any], theirs: Mapping[str, Any], ours: Mapping[str, Any]
                 ) -> tuple[dict[str, Any], list[str], list[str]]:
    """Per flattened key, ``theirs``' change over ``base`` merged onto ``ours``: (keys to set on ours -> value,
    :data:`MISSING` = delete -- ours left them, or moved a number the same way less far; keys ``ours`` already holds
    -- equal or :func:`subsumed`; conflicting keys -- ours moved them elsewhere)."""
    put, held, clash = {}, [], []
    for k in sorted(set(base) | set(theirs)):
        b, t, o = (d.get(k, MISSING) for d in (base, theirs, ours))
        if b == t:
            continue
        if o == t or subsumed(b, t, o):
            held.append(k)
        elif o == b or subsumed(b, o, t):
            put[k] = t
        else:
            clash.append(k)
    return put, held, clash
