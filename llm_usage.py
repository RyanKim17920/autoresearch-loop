"""overfitX SDK — LLM usage ledger (append-only JSONL).

Thread-safe, fire-and-forget logging of every LLM call. Each row carries
timestamp, role, resolved model, backend path, prompt size, token usage,
duration, success flag, and attempt number.

Ledger path resolution (same pattern as config.OVERFITX_ROOT):
    1. <OVERFITX_HOME>/llm_usage.jsonl
    2. <SDK_ROOT>/.llm_usage.jsonl  (fallback when OVERFITX_HOME is unresolvable)

Write failures are swallowed with a single warning per process — the ledger
must NEVER break the call path.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ledger path resolution
# ---------------------------------------------------------------------------

def _resolve_ledger_path() -> Path:
    """Resolve the JSONL ledger path, falling back gracefully."""
    # Primary: OVERFITX_HOME/llm_usage.jsonl
    try:
        from config import OVERFITX_ROOT
        parent = OVERFITX_ROOT
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        return parent / "llm_usage.jsonl"
    except Exception:
        pass
    # Fallback: repo-local (SDK root)
    try:
        sdk_root = Path(__file__).resolve().parent
        return sdk_root / ".llm_usage.jsonl"
    except Exception:
        # Last resort: CWD (should never happen)
        return Path(".llm_usage.jsonl")


_LEDGER_PATH = _resolve_ledger_path()

# ---------------------------------------------------------------------------
# Thread-safe singleton writer
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_writer_started = False
_warned = False


def _write_once_init() -> None:
    """Ensure the parent directory exists (idempotent, best-effort)."""
    global _writer_started
    try:
        _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    _writer_started = True


def record(
    *,
    role: str,
    model: str,
    backend: str,
    prompt_chars: int,
    duration_ms: float,
    ok: bool,
    attempt: int = 1,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    estimated: bool = False,
    error: str | None = None,
    ts: str | None = None,
) -> None:
    """Append one ledger row (thread-safe, fire-and-forget).

    Parameters
    ----------
    role :
        The SDK role (analyst, l2, l3, ideator, etc.).
    model :
        The *resolved* model string, e.g. "Qwen/Qwen3.6-27B" or
        "claude-opus-5-5".
    backend :
        Backend path: "qwen-proxy", "anthropic-oauth", "agent-sdk-web",
        "anthropic-api_key".
    prompt_chars :
        Estimated prompt size in characters.
    duration_ms :
        Wall-clock duration of the call in milliseconds.
    ok :
        True if the call returned a usable response.
    attempt :
        Attempt number (1-based; retries each get their own row).
    input_tokens :
        Actual input token count when the response carries a ``usage`` field.
        If both input_tokens and output_tokens are None, tokens are estimated
        from prompt_chars.
    output_tokens :
        Actual output token count from response.usage.
    estimated :
        True when tokens are estimated (chars/4) rather than from the
        response.usage field.
    error :
        Error message string when ok=False.
    ts :
        ISO-8601 timestamp (defaults to now if not provided).
    """
    global _warned
    if not _writer_started:
        _write_once_init()

    if input_tokens is None and output_tokens is None:
        # Estimate from characters
        input_tokens = max(1, prompt_chars // 4)
        output_tokens = 0
        estimated = True

    row = {
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "role": role,
        "model": model,
        "backend": backend,
        "prompt_chars": prompt_chars,
        "duration_ms": round(duration_ms, 1),
        "ok": ok,
        "attempt": attempt,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated": estimated,
    }
    if error:
        row["error"] = error

    with _lock:
        try:
            with open(_LEDGER_PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            if not _warned:
                _warned = True
                logger.warning(
                    "llm_usage: cannot write to %s; "
                    "all subsequent ledger writes will be silently dropped",
                    _LEDGER_PATH)


# ---------------------------------------------------------------------------
# Ledger reader (for CLI report / tests)
# ---------------------------------------------------------------------------

def read_ledger(path: str | Path | None = None) -> list[dict]:
    """Read all rows from the ledger file. Returns empty list if missing."""
    p = Path(path) if path else _LEDGER_PATH
    if not p.exists():
        return []
    rows = []
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:
        pass
    return rows


# ---------------------------------------------------------------------------
# Aggregation (for CLI report / tests)
# ---------------------------------------------------------------------------

def _empty_stats() -> dict:
    return {"calls": 0, "ok": 0, "errors": 0, "input_tokens": 0,
            "output_tokens": 0, "est_calls": 0, "total_ms": 0.0}


def aggregate(rows: list[dict] | None = None, *, since: str | None = None,
              hourly: bool = False) -> dict:
    """Aggregate the usage ledger by model, role, backend and (optionally) hour.

    ``rows=None`` reads the append-only ledger. ``est_calls`` is carried in every
    bucket on purpose: a row whose tokens were ESTIMATED from characters is not
    a measurement, and a rollup that hides how many of its tokens were guessed
    is the same failure as an unpaired delta reported as a result.

    The rollup lives HERE, beside the ledger it reads, and ``agent.llm`` exports
    it as ``usage_rollup``. The other arrangement — body in ``agent.llm``, shim
    here — is what made ``llm_usage`` import back into ``agent.llm`` from inside
    this function and closed an ``agent.llm -> llm -> llm_usage`` loop.
    """
    if rows is None:
        rows = read_ledger()
    if since:
        rows = [r for r in rows if r.get("ts", "") >= since]

    by_model: dict[str, dict] = {}
    by_role: dict[str, dict] = {}
    by_backend: dict[str, dict] = {}
    buckets: dict[str, dict] = {}
    for row in rows:
        keyed = ((by_model, row.get("model", "unknown")),
                 (by_role, row.get("role", "unknown")),
                 (by_backend, row.get("backend", "unknown")))
        for group, key in keyed:
            s = group.setdefault(key, _empty_stats())
            s["calls"] += 1
            s["ok" if row.get("ok") else "errors"] += 1
            s["input_tokens"] += row.get("input_tokens", 0)
            s["output_tokens"] += row.get("output_tokens", 0)
            if row.get("estimated"):
                s["est_calls"] += 1
            s["total_ms"] += row.get("duration_ms", 0)
        if hourly:
            s = buckets.setdefault(row.get("ts", "")[:13], _empty_stats())
            s["calls"] += 1
            s["input_tokens"] += row.get("input_tokens", 0)
            s["output_tokens"] += row.get("output_tokens", 0)

    result = {
        "total": {
            "calls": len(rows),
            "input_tokens": sum(r.get("input_tokens", 0) for r in rows),
            "output_tokens": sum(r.get("output_tokens", 0) for r in rows),
            "est_calls": sum(1 for r in rows if r.get("estimated")),
        },
        "by_model": by_model, "by_role": by_role, "by_backend": by_backend,
    }
    if hourly:
        result["hourly"] = buckets
    return result


# ---------------------------------------------------------------------------
# Report formatting (shared between CLI and programmatic use)
# ---------------------------------------------------------------------------

def format_report(aggr: dict) -> str:
    """Format an aggregate dict as a human-readable text report."""
    lines = []
    lines.append("=" * 72)
    lines.append("LLM USAGE LEDGER REPORT")
    lines.append("=" * 72)

    total = aggr.get("total", {})
    lines.append(f"Total calls     : {total.get('calls', 0)}")
    lines.append(f"Total in tokens : {total.get('input_tokens', 0):,}")
    lines.append(f"Total out tokens: {total.get('output_tokens', 0):,}")
    lines.append(f"Estimated calls : {total.get('est_calls', 0)}")
    lines.append("")

    lines.append("--- BY MODEL ---")
    lines.append(f"{'model':<30} {'calls':>6} {'ok':>5} {'err':>4} "
                 f"{'in_tok':>10} {'out_tok':>10} {'est':>5} {'ms':>10}")
    lines.append("-" * 72)
    for model, st in sorted(aggr.get("by_model", {}).items(),
                            key=lambda kv: kv[0] or ""):
        lines.append(
            f"{str(model or 'None'):<30} {st['calls']:>6} {st['ok']:>5} {st['errors']:>4} "
            f"{st['input_tokens']:>10,} {st['output_tokens']:>10,} "
            f"{st['est_calls']:>5} {st['total_ms']:>10,.1f}")
    lines.append("")

    lines.append("--- BY ROLE ---")
    lines.append(f"{'role':<20} {'calls':>6} {'ok':>5} {'err':>4} "
                 f"{'in_tok':>10} {'out_tok':>10}")
    lines.append("-" * 55)
    for role, st in sorted(aggr.get("by_role", {}).items(),
                           key=lambda kv: kv[0] or ""):
        lines.append(
            f"{str(role or 'None'):<20} {st['calls']:>6} {st['ok']:>5} {st['errors']:>4} "
            f"{st['input_tokens']:>10,} {st['output_tokens']:>10,}")
    lines.append("")

    lines.append("--- BY BACKEND ---")
    lines.append(f"{'backend':<25} {'calls':>6} {'ok':>5} {'err':>4} "
                 f"{'in_tok':>10} {'out_tok':>10}")
    lines.append("-" * 58)
    for backend, st in sorted(aggr.get("by_backend", {}).items(),
                              key=lambda kv: kv[0] or ""):
        lines.append(
            f"{str(backend or 'None'):<25} {st['calls']:>6} {st['ok']:>5} {st['errors']:>4} "
            f"{st['input_tokens']:>10,} {st['output_tokens']:>10,}")
    lines.append("")

    if "hourly" in aggr:
        lines.append("--- HOURLY ---")
        lines.append(f"{'hour':<15} {'calls':>6} {'in_tok':>10} {'out_tok':>10}")
        lines.append("-" * 40)
        for hour, st in sorted(aggr["hourly"].items()):
            lines.append(
                f"{hour:<15} {st['calls']:>6} "
                f"{st['input_tokens']:>10,} {st['output_tokens']:>10,}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point (used by main.py)
# ---------------------------------------------------------------------------

def cli_report(args: Any = None) -> int:
    """Run the usage report CLI. Returns exit code."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="usage", description="LLM usage ledger report")
    parser.add_argument(
        "--ledger", default=None,
        help="Path to ledger JSONL (default: auto-resolved)")
    parser.add_argument(
        "--since", default=None,
        help="ISO-8601 timestamp; only show rows at or after this time")
    parser.add_argument(
        "--hourly", action="store_true",
        help="Bucket results by hour")
    parser.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Output as JSON instead of human-readable text")
    parser.add_argument(
        "--model", default=None,
        help="Filter to a single model name (prefix match)")

    parsed = parser.parse_args(args)

    rows = read_ledger(parsed.ledger)
    if not rows:
        print("No ledger entries found.")
        return 0

    # Filter by model prefix
    if parsed.model:
        rows = [r for r in rows
                if parsed.model.lower() in r.get("model", "").lower()]

    aggr = aggregate(rows, since=parsed.since, hourly=parsed.hourly)

    if parsed.as_json:
        print(json.dumps(aggr, indent=2, default=str))
    else:
        print(format_report(aggr))
    return 0
