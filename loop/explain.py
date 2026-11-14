"""loop.explain — the one hook for "why did this node close the way it did" (it replaced the Opus postmortem).

:func:`explain` asks ``loop.diagnostics.explain(node, store, snapshot)`` (the computed residual / diagnostic
explanation: loop.world.predictive.explain_text) and returns None while that is unavailable. Prompts that used to quote
postmortems render :func:`block`: the computed explanation where there is one, else the node's stored (historical)
postmortem, else nothing. No LLM is ever called here."""

from __future__ import annotations

from typing import Any, Iterable


def explain(node: Any, store: Any, snapshot: Any = None) -> str | None:
    """The computed explanation of ``node``'s close, or None (no diagnostics module / nothing to say / it raised)."""
    import importlib
    try:                                               # through sys.modules (a stubbed / absent module is honoured)
        diagnostics = importlib.import_module("loop.diagnostics")
    except ImportError:
        return None
    fn = getattr(diagnostics, "explain", None)
    if fn is None:
        return None
    try:
        out = fn(node, store, snapshot)
    except Exception:  # noqa: BLE001 - an explanation never fails a prompt
        return None
    return " ".join(str(out).split()) or None if out else None


def why(node: Any, store: Any, snapshot: Any = None) -> str | None:
    """:func:`explain`, else the node's latest stored (historical) postmortem, else None."""
    if (text := explain(node, store, snapshot)) is not None or not hasattr(store, "postmortems"):
        return text
    pm = max(store.postmortems(node_id=node.id), key=lambda p: p.created_ts or 0.0, default=None)
    if pm is None or not pm.explanation:
        return None
    from loop.core.closed_pm import is_computed
    return f"{pm.explanation[:400]} (computed postmortem)" if is_computed(pm) else f"{pm.explanation[:300]} (stored postmortem)"


def block(store: Any, nodes: Iterable[Any], snapshot: Any = None, limit: int = 40) -> str:
    """``- [a node screened_out] <explanation>`` for the last ``limit`` of ``nodes`` that have one (:func:`why`)."""
    rows = []
    for n in list(nodes)[-limit:]:
        if text := why(n, store, snapshot):
            rows.append(f"- [{n.id} {n.status.value}] {text[:400]}")
    return "\n".join(rows) or "(none)"
