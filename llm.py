"""overfitX SDK — LLM client (bare `anthropic` messages API, sync).

Routing (explicit per-role tiers from config.ROLE_TIERS — no string matching):
- qwen tier (analyst, L2, L3): the local NO-FALLBACK hybrid proxy
  (default http://127.0.0.1:12346: a local, no-fallback proxy you run yourself).
- opus tier (the 2 ideators only): config.OPUS_PATH selects the billing path
  {api_key | credit_pool | qwen}; default qwen when no API key is configured.
  The active path is logged prominently at startup (log_active_backends()).

Qwen preemption handling (spec §8): on 503 / connection-refused we
PAUSE-AND-WAIT — poll a REAL serving signal (GET /v1/models; RUNNING in
squeue != serving while vLLM loads weights), unbounded by default
(OVFX_QWEN_WAIT_MAX=0), logging each wait tick. The qwen tier NEVER falls
back to a paid route. Qwen has no rate limits: any rate-limit-shaped error
means requests are hitting the OLD fallback proxy — that raises hard.

WHAT THIS MODULE IS NOW. Plan commit 10 moved the tool loop, the outage hold's
declared bound, role->transport resolution and the usage rollup into
``agent/llm.py``, where there is ONE loop over TWO transports. What is left here
is the backend PRIMITIVES the transports stand on — client construction, the
serving probe, the 503 episode and backoff, the error classifiers, the opus
billing-path resolver, ``query_tier``/``query_role``/``query_json`` — plus the
OAuth session driver ``sdk_session``, which lives with the six names it reads
rather than above them, and which ``agent.llm`` imports and re-exports.
``opus_sdk_agent_loop`` is an alias of it for the callers that still spell it
that way (ideator, and the billing assertions in tests/test_l3_opus_sdk.py).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import urllib.request
import uuid
from typing import Any

import anthropic

import config
from config import (
    OPUS_MODEL,
    OPUS_PATH,
    OPUS_TIMEOUT,
    OPUS_URL,
    QWEN_MODEL,
    QWEN_POLL_INTERVAL,
    QWEN_PROBE_TIMEOUT,
    QWEN_PROXY_URL,
    QWEN_WAIT_MAX,
    tier_for_role,
)

# Ledger — best-effort; never breaks the call path
try:
    from llm_usage import record as _ledger_record
except Exception:
    def _ledger_record(**_kw):  # type: ignore[no-redef]
        pass

logger = logging.getLogger(__name__)

_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "ratelimit", "429",
                       "overloaded_error", "usage limit")


class QwenUnavailableError(RuntimeError):
    """Qwen backend did not recover within OVFX_QWEN_WAIT_MAX.

    Raised INSTEAD of a paid fallback when the bounded pause-and-wait poll
    expires (the default is unbounded, so this only fires when the operator
    set OVFX_QWEN_WAIT_MAX). Callers surface it as a clean message — never a
    stack-trace spew, never a fallback route.
    """


class NonJSONResponseError(RuntimeError):
    """The model returned non-JSON after all bounded re-ask attempts."""


class OpusTimeoutError(RuntimeError):
    """A single opus-via-OAuth (claude_agent_sdk) call exceeded OVFX_OPUS_TIMEOUT.

    The bundled Claude Code CLI subprocess can spawn then hang or die silently
    (observed 7-15 min freezes emitting no directions); anyio.run() over its
    async query would block forever. We put a hard wall-clock timeout on the
    call so a hang becomes a CAUGHT error the caller can retry / degrade on,
    rather than an unbounded block on the whole research loop.
    """


class LLMOutageError(RuntimeError):
    """The Claude ACCOUNT refused the call: ``kind`` 'quota' (usage / rate limit; ``resets_at`` = unix ts
    when the window resets, None when unknown) or 'auth' (logged out). Not a flaky call: never retried
    here; the loop pauses all LLM work instead (``loop.agents.llm``)."""

    def __init__(self, kind: str, resets_at: float | None, msg: str) -> None:
        super().__init__(msg)
        self.kind, self.resets_at = kind, resets_at


#: AssistantMessage.error values of an account-level refusal (live 2026-10-03: "rate_limit" with the text
#: "You've hit your weekly limit · resets 12am (UTC)", then "authentication_failed" "Not logged in"; the SDK
#: then raises only "Claude Code returned an error result: success").
_SDK_OUTAGE_ERRORS = {"rate_limit": "quota", "billing_error": "quota", "authentication_failed": "auth"}
_RESETS_RE = re.compile(r"resets\s+(?:([A-Z][a-z]{2})\s+(\d{1,2}),\s*)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*\(UTC\)",
                        re.I)


def _resets_from_text(text: str, now: float | None = None) -> float | None:
    """'resets 12am (UTC)' / 'resets Oct 5, 7am (UTC)' / 'resets 11:30pm (UTC)' -> the next such unix ts."""
    import calendar
    import datetime as _dt
    m = _RESETS_RE.search(text or "")
    if not m:
        return None
    mon, day, hh, mm, ap = m.groups()
    h = int(hh) % 12 + (12 if ap.lower() == "pm" else 0)
    t = _dt.datetime.fromtimestamp(time.time() if now is None else now, _dt.timezone.utc)
    try:
        if mon:
            when = t.replace(month=list(calendar.month_abbr).index(mon.title()), day=int(day), hour=h,
                             minute=int(mm or 0), second=0, microsecond=0)
            if when < t - _dt.timedelta(days=180):
                when = when.replace(year=when.year + 1)
        else:
            when = t.replace(hour=h, minute=int(mm or 0), second=0, microsecond=0)
            if when <= t:
                when += _dt.timedelta(days=1)
    except ValueError:
        return None
    return when.timestamp()


def _sdk_outage(msg: Any, box: dict[str, Any]) -> None:
    """Note an account refusal seen in the SDK message stream into ``box`` (kind, resets_at)."""
    import claude_agent_sdk as cas
    if isinstance(msg, cas.AssistantMessage) and (kind := _SDK_OUTAGE_ERRORS.get(msg.error or "")):
        box["kind"] = box.get("kind") or kind
        text = " ".join(b.text for b in msg.content if isinstance(b, cas.TextBlock))
        box["resets_at"] = box.get("resets_at") or _resets_from_text(text)
    elif isinstance(msg, cas.RateLimitEvent) and msg.rate_limit_info.status == "rejected":
        box["kind"] = "quota"
        box["resets_at"] = msg.rate_limit_info.resets_at or box.get("resets_at")
    elif isinstance(msg, cas.ResultMessage) and msg.is_error and getattr(msg, "api_error_status", None) == 429:
        box["kind"] = box.get("kind") or "quota"


class QwenFallbackProxyError(RuntimeError):
    """A rate-limit-shaped error came back on the qwen tier.

    Qwen (local vLLM) has NO rate limits — this error means the SDK is talking
    to the OLD fallback proxy (port 12345, hybrid_proxy.py), which silently
    falls back to a PAID OpenCode route when qwen is preempted. Fix
    OVFX_QWEN_PROXY_URL to point at the no-fallback proxy (port 12346).
    """

class PromptTooLongError(Exception):
    """The estimated prompt exceeds the per-model character budget.

    Raised BEFORE any network call so callers (briefs.py) can shrink evidence
    and retry instead of blowing up the model's context window. Carries
    ``prompt_chars``, ``limit_chars``, and ``model`` attributes.
    """

    def __init__(self, prompt_chars: int, limit_chars: int, model: str) -> None:
        self.prompt_chars = prompt_chars
        self.limit_chars = limit_chars
        self.model = model
        super().__init__(
            f"prompt too long for {model}: {prompt_chars} chars "
            f"(budget {limit_chars})")



class WebQueryError(RuntimeError):
    """The opus-tier web-enabled ideator call failed.

    Raised by ``query_ideator_with_web`` when the claude_agent_sdk session
    with WebSearch/WebFetch tools fails (backend unavailable, tool error,
    timeout, etc.). The caller (briefs.py) catches this to fall back to the
    arXiv-API path. Carries an ``attempted_tools`` attribute listing which
    tools were requested.
    """

    def __init__(self, msg: str, attempted_tools: list[str] | None = None) -> None:
        self.attempted_tools = attempted_tools or []
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Per-model prompt character budgets
# ---------------------------------------------------------------------------
# Derived from known context windows minus the default max_tokens (16000),
# converted to chars at ~3 chars/token, with 10% headroom.
#   Qwen/Qwen3.6-27B: 131K context - 16K output = 115K input tokens
#     115072 * 3 * 0.9 ~ 310K chars
#   Claude Opus 4:    200K context - 16K output = 184K input tokens
#     184000 * 3 * 0.9 ~ 496K chars, rounded up to 600K for safety margin

_QWEN_CONTEXT_TOKENS = 131_072  # Qwen3.6-27B context window
_DEFAULT_MAX_TOKENS = 32_000    # default max_tokens in query_tier

_PROMPT_CHAR_BUDGETS: dict[str, int] = {
    "qwen": (_QWEN_CONTEXT_TOKENS - _DEFAULT_MAX_TOKENS) * 3 * 9 // 10,
    # ~310_694 chars  (115072 prompt tokens * 3 * 0.9)
    "opus": 600_000,
}


def _get_prompt_budget(model: str) -> int:
    """Return the prompt character budget for a model string."""
    lower = model.lower()
    if "qwen" in lower:
        return _PROMPT_CHAR_BUDGETS["qwen"]
    if "opus" in lower:
        return _PROMPT_CHAR_BUDGETS["opus"]
    return _PROMPT_CHAR_BUDGETS["qwen"]


def _block_chars(block) -> int:
    if isinstance(block, str):
        return len(block)
    if isinstance(block, dict):
        return len(block.get("text", block.get("content", "")))
    return len(str(block))


def _msg_chars(msg: dict) -> int:
    content = msg.get("content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(map(_block_chars, content))
    return len(str(content))


def _estimate_prompt_chars(system: str | None, messages: list[dict]) -> int:
    """Roughly estimate total prompt size in characters (system + message content)."""
    total = len(system or "")
    return total + sum(map(_msg_chars, messages))


def _check_prompt_budget(model: str, system: str | None,
                         messages: list[dict]) -> None:
    """Raise PromptTooLongError if the estimated prompt exceeds budget."""
    chars = _estimate_prompt_chars(system, messages)
    limit = _get_prompt_budget(model)
    if chars > limit:
        raise PromptTooLongError(chars, limit, model)



# ---------------------------------------------------------------------------
# Serving-signal probes (NOT squeue: RUNNING != serving while vLLM loads)
# ---------------------------------------------------------------------------


def probe_served_model(base_url: str = QWEN_PROXY_URL,
                       timeout: float = 10.0) -> str | None:
    """GET <base_url>/v1/models; return the first served model id, or None."""
    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        models = data.get("data", [])
        return models[0].get("id") if models else None
    except Exception as e:
        logger.debug("probe %s failed: %s", url, str(e)[:120])
        return None


def tiny_completion_ok(base_url: str = QWEN_PROXY_URL,
                       model: str = QWEN_MODEL,
                       timeout: float = QWEN_PROBE_TIMEOUT) -> bool:
    """One tiny completion through the proxy — the strongest serving signal.

    Bounded by ``timeout`` with max_retries=0, so a hung POST fails the probe
    instead of hanging the caller.
    """
    try:
        client = anthropic.Anthropic(base_url=base_url, api_key="local-qwen",
                                     max_retries=0, timeout=timeout)
        resp = client.messages.create(
            model=model, max_tokens=8,
            messages=[{"role": "user", "content": "ping"}])
        return bool(resp.content)
    except Exception as e:
        logger.debug("tiny completion failed: %s", str(e)[:120])
        return False


def qwen_serving_model() -> str | None:
    """The REAL serving signal: the model id iff the proxy also COMPLETES.

    GET /v1/models alone is not a serving signal on this proxy — it answers with
    a STATIC 200 model list even while POST /v1/messages hangs or 503s (the
    reason run_cycle.hold_for_backend never engaged across all of cycle 8, and
    the reason the 503 episode logic below refuses to treat a healthy GET as
    recovery). So: GET first (cheap; no model listed -> don't pay for a POST),
    then one bounded tiny completion.
    """
    model = probe_served_model()
    if model is None:
        return None
    if not tiny_completion_ok():
        logger.debug("qwen proxy lists %s but does not complete — NOT serving",
                     model)
        return None
    return model


def is_qwen_healthy() -> bool:
    """True iff the qwen proxy lists a model AND answers a tiny completion."""
    return qwen_serving_model() is not None


# Longest single sleep inside a poll interval. The wait is deliberately
# unbounded, but an abort request must be OBSERVED promptly rather than up to a
# full poll_interval late — so the interval is slept in slices.
_ABORT_CHECK_SLICE_S = 1.0


def wait_for_qwen(max_wait: int | float | None = None,
                  poll_interval: int | float = QWEN_POLL_INTERVAL,
                  should_abort=None) -> bool:
    """PAUSE-AND-WAIT for the qwen backend (preemption handling).

    Polls the REAL serving signal (qwen_serving_model: /v1/models + a bounded
    tiny completion), NOT the model list alone. Unbounded by default (max_wait
    None -> config OVFX_QWEN_WAIT_MAX, and 0 there means wait forever). Logs
    every tick. Never falls back to a paid route.

    ``should_abort`` is an optional predicate consulted before every probe and
    between sleep slices; when it returns True the wait gives up immediately and
    returns False. It exists ONLY for shutdown (SIGTERM/SIGINT): the live
    incident was an operator's SIGTERM that could not take because this loop
    slept through it. Callers that pass nothing keep the unbounded behaviour
    exactly — a down backend must never void a lineage.
    """
    if max_wait is None:
        max_wait = QWEN_WAIT_MAX  # 0 = unbounded
    elapsed = 0.0
    tick = 0
    while True:
        if should_abort is not None and should_abort():
            logger.warning("qwen wait ABORTED by shutdown request after %ds",
                           elapsed)
            return False
        model = qwen_serving_model()
        if model is not None:
            logger.info("qwen serving again (model=%s) after %ds", model, elapsed)
            return True
        tick += 1
        logger.warning(
            "qwen wait tick %d: proxy %s not serving (%ds elapsed%s) — "
            "likely qwen-serve preempted / vLLM loading; pausing, NOT failing",
            tick, QWEN_PROXY_URL, elapsed,
            "" if not max_wait else f"/{max_wait}s max")
        if max_wait and elapsed >= max_wait:
            logger.error("qwen did not recover within %ds", max_wait)
            return False
        if should_abort is None:
            time.sleep(poll_interval)
        else:
            remaining = float(poll_interval)
            while remaining > 0:
                if should_abort():
                    logger.warning("qwen wait ABORTED by shutdown request "
                                   "after %ds", elapsed)
                    return False
                slice_s = min(_ABORT_CHECK_SLICE_S, remaining)
                time.sleep(slice_s)
                remaining -= slice_s
        elapsed += poll_interval



# ---------------------------------------------------------------------------
# Qwen 503 outage EPISODE + mandatory backoff.
#
# Fix for a live busy-loop: the no-fallback proxy can answer GET /v1/models
# as healthy while POST /v1/messages 503s, so the old
# "503 -> wait_for_qwen() -> healthy -> retry" cycle span ~4 req/s with NO
# sleep, and every phantom "recovery" reset the wait accounting so
# OVFX_QWEN_WAIT_MAX never triggered.
#
# An EPISODE starts at the first 503 on the REAL request and ends ONLY when a
# real completion succeeds (200). A healthy health probe never resets the
# episode clock or the consecutive-failure counter, and never bypasses the
# minimum inter-retry sleep.
# ---------------------------------------------------------------------------

QWEN_BACKOFF_MIN = 5.0      # minimum sleep before retrying after a 503
QWEN_BACKOFF_CAP = 60.0     # exponential backoff cap (5 -> 10 -> 20 -> 40 -> 60)
QWEN_TICK_LOG_EVERY = 30.0  # log the wait tick at most once per ~30s

_qwen_episode_lock = threading.Lock()
_qwen_episode: dict[str, Any] = {"start": None, "failures": 0, "last_tick": None}


def _qwen_episode_reset() -> None:
    """End the outage episode — called ONLY on a real 200 completion."""
    with _qwen_episode_lock:
        _qwen_episode["start"] = None
        _qwen_episode["failures"] = 0
        _qwen_episode["last_tick"] = None


def _qwen_budget_error(elapsed: float) -> QwenUnavailableError:
    return QwenUnavailableError(
        f"qwen backend did not recover within "
        f"OVFX_QWEN_WAIT_MAX={QWEN_WAIT_MAX}s (episode elapsed {elapsed:.0f}s, "
        f"proxy={QWEN_PROXY_URL}). No paid fallback was attempted; bring the "
        "qwen slurm job back and re-run.")


def _qwen_log_tick_locked(now: float, elapsed: float, note: str) -> None:
    """Rate-limited episode tick log (call with _qwen_episode_lock held)."""
    last = _qwen_episode["last_tick"]
    if last is not None and now - last < QWEN_TICK_LOG_EVERY:
        return
    _qwen_episode["last_tick"] = now
    logger.warning(
        "qwen wait tick %d: %s (%.0fs elapsed%s)",
        _qwen_episode["failures"], note, elapsed,
        "" if not QWEN_WAIT_MAX else f"/{QWEN_WAIT_MAX}s max")


def _qwen_note_503(exc: Exception) -> float:
    """Record one 503 on the REAL request; return the mandatory backoff sleep.

    Starts the episode on the first 503. Raises QwenUnavailableError once the
    episode has lasted >= OVFX_QWEN_WAIT_MAX (0 = unbounded). Health-probe
    results never touch this state.
    """
    now = time.monotonic()
    with _qwen_episode_lock:
        if _qwen_episode["start"] is None:
            _qwen_episode["start"] = now
            _qwen_episode["failures"] = 0
            _qwen_episode["last_tick"] = None
            logger.warning(
                "qwen backend down (%s); pausing until it returns — backoff "
                "%.0f..%.0fs between retries; only a real 200 ends the episode",
                str(exc)[:120], QWEN_BACKOFF_MIN, QWEN_BACKOFF_CAP)
        _qwen_episode["failures"] += 1
        failures = _qwen_episode["failures"]
        elapsed = now - _qwen_episode["start"]
        if QWEN_WAIT_MAX and elapsed >= QWEN_WAIT_MAX:
            raise _qwen_budget_error(elapsed) from exc
        delay = min(QWEN_BACKOFF_CAP,
                    QWEN_BACKOFF_MIN * (2 ** min(failures - 1, 10)))
        _qwen_log_tick_locked(
            now, elapsed,
            f"503 on real request via {QWEN_PROXY_URL}; backing off "
            f"{delay:.0f}s before retry (healthy probe cannot bypass this)")
        return delay


def _qwen_probe_gate(exc: Exception) -> None:
    """After the backoff sleep, ALSO hold while the health probe is down
    (vLLM loading weights). Counts against the SAME episode budget; a healthy
    probe only exits this gate — it never resets the episode."""
    # is_qwen_healthy() is now a REAL serving signal (model list + a bounded
    # tiny completion), so a static 200 from GET /v1/models can no longer be
    # mistaken for recovery and let the gate through while POST still hangs.
    while not is_qwen_healthy():
        now = time.monotonic()
        with _qwen_episode_lock:
            start = _qwen_episode["start"]
            elapsed = (now - start) if start is not None else 0.0
            if QWEN_WAIT_MAX and elapsed >= QWEN_WAIT_MAX:
                raise _qwen_budget_error(elapsed) from exc
            _qwen_log_tick_locked(
                now, elapsed,
                f"proxy {QWEN_PROXY_URL} not completing; still pausing")
        time.sleep(QWEN_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Clients & backend resolution
# ---------------------------------------------------------------------------


def qwen_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=QWEN_PROXY_URL, api_key="local-qwen",
                               max_retries=0, timeout=600.0)


def resolve_opus_backend() -> tuple[str, str, anthropic.Anthropic | None]:
    """Resolve the opus billing path -> (path, model, client or None).

    - api_key:      official/headroom endpoint with ANTHROPIC_API_KEY.
    - oauth:        real opus via claude_agent_sdk on the logged-in Max/Pro
                    subscription (no API key); client is None -> query_tier
                    drives it through _opus_via_agent_sdk.
    - credit_pool:  Agent-SDK credit pool — not wired for the bare messages
                    API; raises with instructions rather than mis-billing.
    - qwen:         free local route (loses opus-grade ideation).
    """
    import os
    if OPUS_PATH == "api_key":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "opus_path=api_key but ANTHROPIC_API_KEY is not set. "
                "Set the key or use OVFX_OPUS_PATH=qwen.")
        client = anthropic.Anthropic(base_url=OPUS_URL, api_key=key,
                                     max_retries=2, timeout=600.0)
        return "api_key", OPUS_MODEL, client
    if OPUS_PATH == "oauth":
        # client is None: driven through claude_agent_sdk (subscription OAuth).
        return "oauth", OPUS_MODEL, None
    if OPUS_PATH == "credit_pool":
        raise RuntimeError(
            "opus_path=credit_pool: the Agent-SDK monthly credit pool cannot "
            "be billed through the bare anthropic messages API. Run the "
            "ideators through claude-agent-sdk with subscription auth, or "
            "switch OVFX_OPUS_PATH to api_key / qwen.")
    if OPUS_PATH == "qwen":
        return "qwen", QWEN_MODEL, None  # None -> use qwen_client()
    raise RuntimeError(
        f"Unknown OVFX_OPUS_PATH={OPUS_PATH!r}; expected api_key|credit_pool|qwen")


def log_active_backends() -> None:
    """Log the active model routing prominently at startup."""
    served = probe_served_model()
    logger.info("=" * 64)
    logger.info("MODEL ROUTING")
    logger.info("  qwen tier  : %s @ %s (served now: %s)",
                QWEN_MODEL, QWEN_PROXY_URL, served or "NOT SERVING")
    if OPUS_PATH == "qwen":
        logger.info("  opus tier  : OVFX_OPUS_PATH=qwen -> ideators run on the "
                    "FREE local qwen route (%s); opus-grade ideation OFF", QWEN_MODEL)
    elif OPUS_PATH == "oauth":
        logger.info("  opus tier  : OVFX_OPUS_PATH=oauth -> %s via claude_agent_sdk "
                    "on the logged-in Max/Pro subscription (no API key)", OPUS_MODEL)
        # The banner used to be a pure config echo and advertised this route
        # even when the package it is built on was not importable. Say so.
        if not opus_sdk_importable():
            logger.warning(
                "  opus tier  : NOT USABLE — claude_agent_sdk is not importable "
                "in this interpreter (%s). l2/analyst will run on qwen for the "
                "WHOLE cycle and the ideator (no fallback by design) will FAIL. "
                "Launch with `uv run python`, not bare python3.",
                sys.executable)
    else:
        logger.info("  opus tier  : OVFX_OPUS_PATH=%s -> %s @ %s",
                    OPUS_PATH, OPUS_MODEL, OPUS_URL)
    logger.info("=" * 64)
    if served and served != QWEN_MODEL:
        logger.warning("proxy serves %r but OVFX_QWEN_MODEL=%r — override the "
                       "env var to match", served, QWEN_MODEL)


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, anthropic.RateLimitError) or any(
        m in msg for m in _RATE_LIMIT_MARKERS)


def _is_backend_down_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, (anthropic.APIConnectionError,
                            anthropic.InternalServerError)) or any(
        m in msg for m in ("503", "connection refused", "connection error",
                           "service unavailable", "cannot connect"))


# A 401/403 from the LOCAL, self-hosted qwen relay is an OUTAGE signature.
#
# Live c11 incident (2026-08-13 18:05:55): SLURM preempted `qwen-serve-low-pri`;
# the relay on 127.0.0.1:12346 fell through to an upstream that rejected the
# local key, and `client.messages.create` raised
#   Error code: 401 - {'type': 'error', 'error':
#     {'type': 'authentication_error', 'message': 'invalid x-api-key'}}
# `_is_backend_down_error` matches only 503 / connection-refused shapes, so the
# exception escaped and killed three L3 lineages (fixed in agents/l3.py by
# 4515c72) — and, one layer down, escaped the qwen tier of `query_tier`, which
# is the ONLY backend path L2 uses (agents/l2.py -> llm.query_json ->
# query_role -> query_tier).
#
# SCOPE: consulted ONLY on the qwen tier. `_is_backend_down_error` above is
# SHARED with the opus tier (llm.py:_classify_sdk_error, agents/l3.py's opus
# branch), which talks to a REAL remote API over oauth/api-key where a 401 is a
# genuine credential fault that must surface immediately — so this classifier is
# deliberately kept OUT of it and wired in at the qwen call site instead.
#
# NOT a new policy and NOT an infinite wait: it feeds the EXISTING 503 outage
# machinery (`_qwen_note_503` backoff + `_qwen_probe_gate`), so a permanently
# wrong local key terminates with the same QwenUnavailableError at the same
# OVFX_QWEN_WAIT_MAX budget a connection error produces.
_LOCAL_AUTH_OUTAGE_MARKERS = (
    "error code: 401", "error code: 403", "authentication_error",
    "invalid x-api-key", "permission_error",
)


def _is_local_backend_auth_error(exc: Exception) -> bool:
    """True iff `exc` is an auth/permission rejection from the LOCAL relay.

    Typed check first (what the anthropic SDK actually raises); the string form
    is a FALLBACK for the case where the 401 only survives as a wrapped message.
    """
    if isinstance(exc, (anthropic.AuthenticationError,
                        anthropic.PermissionDeniedError)):
        return True
    if isinstance(exc, anthropic.APIStatusError) and \
            getattr(exc, "status_code", None) in (401, 403):
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _LOCAL_AUTH_OUTAGE_MARKERS)


def _extract_text(resp: Any) -> str:
    parts = []
    for block in resp.content:
        if getattr(block, "type", "") == "text" and getattr(block, "text", None):
            parts.append(block.text)
    return "".join(parts).strip()


def _sdk_usage_tokens(usage: Any) -> tuple[int | None, int | None]:
    """Extract (input_tokens, output_tokens) from a claude_agent_sdk usage dict.

    claude_agent_sdk surfaces the RAW Anthropic API usage object (a plain dict)
    on both ``AssistantMessage.usage`` and — cumulatively — ``ResultMessage.usage``.
    Standard keys: ``input_tokens``, ``output_tokens``, and the cache accounting
    fields ``cache_read_input_tokens`` / ``cache_creation_input_tokens``. Cache
    tokens are folded into the input count for billing fidelity (they are real
    input tokens the subscription was charged for). Returns ``(None, None)`` when
    usage is absent or malformed, so the caller can fall back to estimation.
    """
    if not isinstance(usage, dict):
        return None, None
    it = usage.get("input_tokens")
    ot = usage.get("output_tokens")
    inp: int | None = None
    if it is not None:
        inp = (int(it)
               + int(usage.get("cache_read_input_tokens") or 0)
               + int(usage.get("cache_creation_input_tokens") or 0))
    out: int | None = None if ot is None else int(ot)
    return inp, out


# claude_agent_sdk caps ONE stream-json message at 1 MiB by default; a research session reading a big file /
# tool result overflowed it (live 11-09..11-11: "JSON message exceeded maximum buffer size of 1048576 bytes",
# a fatal reader error that cost a retry). Every ClaudeAgentOptions here passes this instead.
SDK_MAX_BUFFER_SIZE = 16 * 1024 * 1024


def _claude_cli_path() -> str | None:
    """CLI binary for ClaudeAgentOptions.cli_path: $LOOP_CLAUDE_CLI, else the
    system ``claude`` on PATH (the SDK's bundled CLI breaks after an account
    change), else None (SDK default)."""
    return os.environ.get("LOOP_CLAUDE_CLI") or shutil.which("claude")


# Env vars that MUST be scrubbed before an opus-via-OAuth call. Any of these
# would re-point claude_agent_sdk at the paid local proxy (:12345 deepseek /
# OpenCode route) or a metered API key instead of the logged-in Max/Pro
# subscription — mis-billing the ideators. We pop them for the duration of the
# call AND pass them blanked in ClaudeAgentOptions.env (belt-and-suspenders for
# the spawned CLI subprocess).
_OAUTH_SCRUB_ENV = (
    "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_CUSTOM_HEADERS", "HEADROOM_BACKEND",
)

# Per-conversation opus turn cap — a runaway backstop ONLY, not a work budget.
# Single knob for both ideator paths (plain + web/literature). It must be high
# enough that a FINISHED response is never reported as a "Reached maximum number
# of turns" error: the SDK streams the assistant text FIRST and only surfaces
# that error afterward (from the CLI's non-zero exit), so a low cap makes the
# code discard an already-complete response and fail/retry. The old value of 1
# tripped this on ~half of all ideator calls (see git history / ledger). 200
# matches the L3-editor cap (agents/l3.py MAX_AGENT_TURNS).
OPUS_IDEATOR_MAX_TURNS = 200


def _opus_via_agent_sdk(prompt: str, system: str | None = None,
                        max_tokens: int = 16000,
                        temperature: float = 0.7) -> str:
    """One real-opus call through claude_agent_sdk on the logged-in OAuth
    (Max/Pro) subscription — NO API key, NO local proxy.

    CRITICAL SAFETY (load-bearing): the SDK normally runs with
    ANTHROPIC_BASE_URL / model overrides pointing at the PAID local proxy
    (:12345 OpenCode/deepseek) and possibly a metered ANTHROPIC_API_KEY. If any
    of those leaked into the claude-agent-sdk subprocess this call would be
    MIS-BILLED to that route instead of the subscription. We snapshot+pop every
    routing/override env var for the duration of the call (restored in
    ``finally``) AND pass them blanked in ClaudeAgentOptions.env so the spawned
    CLI subprocess cannot see them either. setting_sources=[] stops
    ~/.claude/settings.json from re-injecting ANTHROPIC_BASE_URL.

    temperature is accepted for signature parity but IGNORED — 0.2.110's
    ClaudeAgentOptions has no temperature field; the ideator uses a fixed 0.7,
    which is fine.
    """
    import os

    import anyio
    import claude_agent_sdk as cas

    saved = {k: os.environ.pop(k, None) for k in _OAUTH_SCRUB_ENV}
    try:
        opts = cas.ClaudeAgentOptions(
            model=OPUS_MODEL,
            cli_path=_claude_cli_path(),
            setting_sources=[],
            max_buffer_size=SDK_MAX_BUFFER_SIZE,
            # This generic opus-tier path stays TOOL-LESS: it serves plain text
            # calls (repair/escalation) AND the ideator's degraded fallback. The
            # ideator's PRIMARY, TOOL-ENABLED evidence-pull path lives in
            # agents/ideator.py (opus_sdk_agent_loop + a read-only MCP server);
            # do NOT add tools here or every opus-tier text call would inherit
            # them. With allowed_tools=[] a higher max_turns only lets a long
            # single-turn answer COMPLETE — it does not (and should not) make
            # this path "use tools". If a future change wants this path to
            # reason with tools/extended-thinking, wire it through the ideator
            # tool loop, not here.
            allowed_tools=[],
            # Runaway backstop only. The OLD value of 1 tripped a "Reached
            # maximum number of turns (1)" error on ~half of ideator calls: the
            # SDK streams the complete assistant answer FIRST, then surfaces
            # that error from the CLI's non-zero exit, so the collected response
            # was DISCARDED and the call failed/retried (throwing away good
            # ideation). A high cap means a finished single-turn answer is never
            # mis-reported as an error.
            max_turns=OPUS_IDEATOR_MAX_TURNS,
            system_prompt=system or "",
            env={k: "" for k in _OAUTH_SCRUB_ENV},
        )

        # Real token usage captured out of the SDK message stream (see
        # _sdk_usage_tokens). ResultMessage.usage is the final cumulative count;
        # AssistantMessage.usage is a per-turn fallback. Populated as a side
        # effect of _collect so the success-path ledger record can log real
        # tokens with estimated=False (mirrors the api_key / qwen paths).
        _usage_box: dict[str, int | None] = {"input": None, "output": None}

        _outage: dict[str, Any] = {}

        def _absorb(msg: Any) -> None:
            _sdk_outage(msg, _outage)
            if isinstance(msg, cas.AssistantMessage):
                for block in msg.content:
                    if isinstance(block, cas.TextBlock):
                        parts.append(block.text)
            u_in, u_out = _sdk_usage_tokens(getattr(msg, "usage", None))
            if u_in is not None:
                _usage_box["input"] = u_in
            if u_out is not None:
                _usage_box["output"] = u_out

        async def _collect() -> str:
            # Reset per attempt (the max-turns retry re-invokes _collect).
            parts.clear()
            _outage.clear()
            _usage_box["input"] = _usage_box["output"] = None
            # Hard wall-clock timeout: the bundled CLI subprocess can hang/die
            # silently. fail_after cancels the async query (which tears the
            # subprocess down) and raises TimeoutError, which we re-raise as a
            # typed OpusTimeoutError so the caller degrades instead of blocking
            # forever. OPUS_TIMEOUT=0 disables the wall (unbounded, old behavior).
            if OPUS_TIMEOUT and OPUS_TIMEOUT > 0:
                try:
                    with anyio.fail_after(OPUS_TIMEOUT):
                        async for msg in cas.query(prompt=prompt, options=opts):
                            _absorb(msg)
                except TimeoutError as e:
                    raise OpusTimeoutError(
                        f"opus (claude_agent_sdk oauth) call exceeded "
                        f"OVFX_OPUS_TIMEOUT={OPUS_TIMEOUT:.0f}s and was aborted "
                        f"(bundled Claude Code CLI hung/emitted nothing); "
                        f"{len(parts)} partial text block(s) received") from e
            else:
                async for msg in cas.query(prompt=prompt, options=opts):
                    _absorb(msg)
            return "".join(parts).strip()

        parts: list[str] = []
        first_attempt = [True]
        last_exc = None
        _sdk_attempt = 0
        _sdk_pc = len(prompt) + len(system or "")
        while True:
            _sdk_attempt += 1
            _sdk_t0 = time.monotonic()
            try:
                _sdk_result = anyio.run(_collect)
                _u_in, _u_out = _usage_box["input"], _usage_box["output"]
                _have_usage = _u_in is not None and _u_out is not None
                _ledger_record(
                    role="ideator", model=OPUS_MODEL,
                    backend="anthropic-oauth",
                    prompt_chars=_sdk_pc,
                    duration_ms=1000 * (time.monotonic() - _sdk_t0),
                    ok=True, attempt=_sdk_attempt,
                    input_tokens=_u_in if _have_usage else None,
                    output_tokens=_u_out if _have_usage else None,
                    estimated=not _have_usage,
                )
                return _sdk_result
            except Exception as e:
                last_exc = e
                _ledger_record(
                    role="ideator", model=OPUS_MODEL,
                    backend="anthropic-oauth",
                    prompt_chars=_sdk_pc,
                    duration_ms=1000 * (time.monotonic() - _sdk_t0),
                    ok=False, attempt=_sdk_attempt,
                    # Capture the exception TYPE + message so the ledger
                    # distinguishes rate-limit / CLI-hang / auth / decode
                    # failures instead of an opaque bare message.
                    error=f"{type(e).__name__}: {e}"[:300], estimated=True,
                )
                if _outage.get("kind"):
                    raise LLMOutageError(_outage["kind"], _outage.get("resets_at"),
                                         f"{_outage['kind']}: {''.join(parts)[:200] or e}") from e
                msg = str(e).lower()
                if "maximum number of turns" in msg and first_attempt[0]:
                    first_attempt[0] = False
                    logger.warning("opus (claude_agent_sdk) max-turns error — "
                                   "retrying once: %s", str(e))
                    continue
                raise

    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

# ---------------------------------------------------------------------------
# Web-enabled ideator (literature brief)
# ---------------------------------------------------------------------------
# Reuses the claude_agent_sdk plumbing from _opus_via_agent_sdk but adds
# WebSearch/WebFetch tools and runs a multi-turn session (_WEB_MAX_TURNS,
# The agent is instructed to search papers, arXiv, HuggingFace, GitHub, and
# return a structured JSON response. Returns the final text for downstream
# query_json-style parsing. Raises WebQueryError on any failure (including
# opus backend unavailable), which the caller catches to fall back to arXiv.
#
# Runtime caveat: WebSearch/WebFetch are claude_agent_sdk built-in server
# tools that run through the Claude Code CLI subprocess. They require the
# logged-in OAuth session (opus-tier). In headless/CI environments without
# a valid Claude Code auth, these tools will fail — the caller must degrade
# to the arXiv-API fallback.


_WEB_TOOLS: list[str] = ["web_search", "web_fetch"]
# The web/literature ideator is a genuine multi-turn tool session (WebSearch/
# WebFetch), so this is a real work budget — share the single ideator turn-cap
# knob (runaway backstop) with the plain ideator. Raised from 40 so a deep
# literature dig is never cut short mid-session.
_WEB_MAX_TURNS = OPUS_IDEATOR_MAX_TURNS


def query_ideator_with_web(prompt: str, system: str | None = None,
                           temperature: float = 0.7) -> str:
    """Run a multi-turn opus session with WebSearch/WebFetch enabled.

    Uses the existing resolve_opus_backend / OAuth plumbing to drive a
    claude_agent_sdk session that can search the web. Returns the combined
    assistant text (suitable for downstream JSON parsing). Raises
    WebQueryError if the opus backend is unavailable or the session errors.

    Parameters
    ----------
    prompt : str
        The user prompt (typically the literature brief prompt).
    system : str | None
        System prompt (ideator agent def body).
    temperature : float
        Ignored by claude_agent_sdk (no temperature field in 0.2.110), but
        kept for signature parity.

    Returns
    -------
    str
        Combined text from all assistant messages.

    Raises
    ------
    WebQueryError
        On any failure: opus backend unavailable, tool errors, timeout,
        claude_agent_sdk errors. Callers should catch this and fall back
        to the arXiv-API path.
    PromptTooLongError
        If the prompt exceeds the opus character budget (checked before
        the call).
    """
    # Prompt budget guard (same as query_tier, checked BEFORE the call)
    _check_prompt_budget(OPUS_MODEL, system,
                         [{"role": "user", "content": prompt}])

    # Verify opus backend is usable (oauth or api_key — qwen has no web tools)
    try:
        path, model, client = resolve_opus_backend()
    except Exception as e:
        raise WebQueryError(
            f"opus backend unavailable for web search ({str(e)[:120]}) — "
            "fall back to arXiv-API"
        ) from e

    if path == "qwen":
        raise WebQueryError(
            "OVFX_OPUS_PATH=qwen cannot use WebSearch/WebFetch — "
            "fall back to arXiv-API"
        )
    if path == "credit_pool":
        raise WebQueryError(
            "OVFX_OPUS_PATH=credit_pool cannot drive web tools — "
            "fall back to arXiv-API"
        )

    import os
    import anyio
    import claude_agent_sdk as cas

    web_tools = list(_WEB_TOOLS)
    saved = {k: os.environ.pop(k, None) for k in _OAUTH_SCRUB_ENV}
    try:
        opts = cas.ClaudeAgentOptions(
            model=OPUS_MODEL,
            cli_path=_claude_cli_path(),
            setting_sources=[],
            max_buffer_size=SDK_MAX_BUFFER_SIZE,
            allowed_tools=web_tools,
            max_turns=_WEB_MAX_TURNS,
            system_prompt=system or "",
            env={k: "" for k in _OAUTH_SCRUB_ENV},
        )

        # Real token usage from the SDK stream. This is a multi-turn session
        # (WebSearch/WebFetch), so ResultMessage.usage is the cumulative total
        # across all turns — exactly what we want to bill.
        _usage_box: dict[str, int | None] = {"input": None, "output": None}
        parts: list[str] = []

        def _absorb(msg: Any) -> None:
            if isinstance(msg, cas.AssistantMessage):
                for block in msg.content:
                    if isinstance(block, cas.TextBlock):
                        parts.append(block.text)
            u_in, u_out = _sdk_usage_tokens(getattr(msg, "usage", None))
            if u_in is not None:
                _usage_box["input"] = u_in
            if u_out is not None:
                _usage_box["output"] = u_out

        async def _collect() -> str:
            # Reset per attempt (the max-turns retry re-invokes _collect).
            parts.clear()
            _usage_box["input"] = _usage_box["output"] = None
            if OPUS_TIMEOUT and OPUS_TIMEOUT > 0:
                try:
                    with anyio.fail_after(OPUS_TIMEOUT):
                        async for msg in cas.query(prompt=prompt, options=opts):
                            _absorb(msg)
                except TimeoutError as e:
                    raise WebQueryError(
                        f"web query exceeded OVFX_OPUS_TIMEOUT={OPUS_TIMEOUT:.0f}s — "
                        f"session aborted; {len(parts)} partial text block(s) received",
                        attempted_tools=web_tools,
                    ) from e
            else:
                async for msg in cas.query(prompt=prompt, options=opts):
                    _absorb(msg)
            return "".join(parts).strip()

        # Retry-once for the known "maximum number of turns" flake
        first_attempt = [True]
        last_exc = None
        _sdk_attempt = 0
        _sdk_pc = len(prompt) + len(system or "")
        while True:
            _sdk_attempt += 1
            _sdk_t0 = time.monotonic()
            try:
                _sdk_result = anyio.run(_collect)
                _u_in, _u_out = _usage_box["input"], _usage_box["output"]
                _have_usage = _u_in is not None and _u_out is not None
                _ledger_record(
                    role="ideator", model=OPUS_MODEL,
                    backend="agent-sdk-web",
                    prompt_chars=_sdk_pc,
                    duration_ms=1000 * (time.monotonic() - _sdk_t0),
                    ok=True, attempt=_sdk_attempt,
                    input_tokens=_u_in if _have_usage else None,
                    output_tokens=_u_out if _have_usage else None,
                    estimated=not _have_usage,
                )
                return _sdk_result
            except WebQueryError as e:
                _ledger_record(
                    role="ideator", model=OPUS_MODEL,
                    backend="agent-sdk-web",
                    prompt_chars=_sdk_pc,
                    duration_ms=1000 * (time.monotonic() - _sdk_t0),
                    ok=False, attempt=_sdk_attempt,
                    # Record the REAL detail (timeout wall / unavailable-tools /
                    # partial-blocks) instead of a generic canned string.
                    error=f"{type(e).__name__}: {e}"[:300],
                    estimated=True,
                )
                raise  # Don't retry WebQueryError (our own type)
            except Exception as e:
                last_exc = e
                _ledger_record(
                    role="ideator", model=OPUS_MODEL,
                    backend="agent-sdk-web",
                    prompt_chars=_sdk_pc,
                    duration_ms=1000 * (time.monotonic() - _sdk_t0),
                    ok=False, attempt=_sdk_attempt,
                    # Exception TYPE + message so rate-limit vs CLI-hang vs
                    # decode failures are distinguishable in the ledger.
                    error=f"{type(e).__name__}: {e}"[:300], estimated=True,
                )
                msg_str = str(e).lower()
                if "maximum number of turns" in msg_str and first_attempt[0]:
                    first_attempt[0] = False
                    logger.warning(
                        "query_ideator_with_web: max-turns error — "
                        "retrying once: %s", str(e)
                    )
                    continue
                raise WebQueryError(
                    f"web ideator session failed: {e}",
                    attempted_tools=web_tools,
                ) from e

    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Opus-over-OAuth session: the knobs, the classifier, and the driver that reads
# them — one section, because they are one thing
# ---------------------------------------------------------------------------
# ``sdk_session`` below is the agent-SDK TRANSPORT of the one tool loop, not a
# loop of its own; ``agent.llm`` imports the name and drives it through
# ``AgentSdkTransport``. It preserves the OAuth env-scrub billing safety, the
# OPUS_TIMEOUT wall, and REAL-usage ledgering (estimated=False), and adds
# bounded retry-with-backoff + pause-not-degrade so the flaky OAuth CLI
# subprocess (measured ~61% of opus calls failed/retried) degrades cleanly to
# the caller (ok=False) instead of stranding it.
#
# These three constants and ``_classify_sdk_error`` stay HERE because they are
# read through this module at call time, which is what keeps a
# ``monkeypatch.setattr(llm, ...)`` measuring the thing it names. That is also
# why the driver is here and not in ``agent/llm.py``: a function that reads six
# names out of this module through an ``import llm`` while this module imports
# it back from inside a function body is a cycle carrying exactly one function.

OPUS_SDK_BACKOFF_MIN = 5.0   # first sleep after a transient SDK-subprocess fail
OPUS_SDK_BACKOFF_CAP = 60.0  # exponential backoff ceiling (5->10->20->40->60)
OPUS_SDK_DEFAULT_RETRIES = 3  # bounded transient-flake retries when opus is NOT
#   configured to pause (OVFX_PAUSE_BACKENDS) — still bounded so a flaky CLI can
#   never strand the caller; when opus IS paused we use config.opus_backend_retry_max().


def _classify_sdk_error(exc: Exception) -> str:
    """Classify a claude_agent_sdk failure for the retry policy.

    Returns one of: "timeout" | "cli_missing" | "max_turns" | "backend_down" |
    "quota" | "other" (``sdk_session`` reads quota / auth from the message stream first). Everything except cli_missing (permanent) and max_turns (retry-once)
    is treated as a bounded-retryable transient flake by opus_sdk_agent_loop.
    """
    if isinstance(exc, OpusTimeoutError):
        return "timeout"
    msg = str(exc).lower()
    try:
        import claude_agent_sdk as cas
        if isinstance(exc, cas.CLINotFoundError):
            return "cli_missing"
        if isinstance(exc, (cas.CLIConnectionError, cas.ProcessError,
                            cas.CLIJSONDecodeError)):
            return "backend_down"
    except Exception:  # noqa: BLE001 — SDK import/attr issues never break classify
        pass
    if "maximum number of turns" in msg:
        return "max_turns"
    if "usage limit" in msg or ("hit your" in msg and "limit" in msg):
        return "quota"
    if _is_backend_down_error(exc):
        return "backend_down"
    # OAuth CLI subprocess flakiness (spawn/die/JSON-decode/pipe) — retryable.
    if any(m in msg for m in ("cli", "subprocess", "process", "json",
                              "connection", "timeout", "timed out",
                              "broken pipe", "eof", "transport")):
        return "backend_down"
    return "other"


def sdk_session(prompt: str, *, system: str | None = None,
                mcp_servers: dict[str, Any] | None = None,
                allowed_tools: list[str] | None = None,
                max_turns: int = 40, stop_tool: str | None = None,
                cwd: str | None = None, role: str = "ideator",
                backend: str = "agent-sdk-editor",
                on_tool_use: Any = None, done: Any = None) -> dict[str, Any]:
    """One ``claude_agent_sdk`` session over the logged-in OAuth subscription.

    It is a TRANSPORT, not a loop: the SDK owns the turns, this streams the
    messages, captures usage and the ``stop_tool`` payload, applies the
    ``OVFX_OPUS_TIMEOUT`` wall and does bounded retry-with-backoff so a flaky
    OAuth CLI degrades to the caller (``ok=False``) instead of stranding it. The
    narration nudge and the turn budget live in ``agent.llm.Loop``, which drives
    this through ``agent.llm.AgentSdkTransport`` and re-exports the name.

    It lives HERE, with the constants and the classifier directly above, for the
    reason those constants give for staying: they are read through this module
    at call time, which is what keeps a ``monkeypatch.setattr(llm, ...)``
    measuring the thing it names. Reading them from ``agent/llm.py`` meant that
    package importing this one at module level while this one imported back into
    it from inside a function body — a loop whose only cargo was this function.

    BILLING SAFETY IS LOAD-BEARING. Every routing/override env var is popped for
    the duration (restored in ``finally``) AND passed blank in
    ``ClaudeAgentOptions.env``, and ``setting_sources=[]`` stops
    ``~/.claude/settings.json`` re-injecting ``ANTHROPIC_BASE_URL``. Without it
    the call is mis-billed to the paid local proxy instead of the subscription.
    ``permission_mode`` stays at the SDK default so ``allowed_tools`` is the
    allowlist and the built-in Bash/Edit/Write — which would bypass the caller's
    guards entirely — are not reachable.
    """
    import os

    import anyio
    import claude_agent_sdk as cas

    parts: list[str] = []
    transcript: list[dict] = []
    counters = {"tool_calls": 0, "turns": 0}
    usage_box: dict[str, int | None] = {"input": None, "output": None}
    stop_box: dict[str, Any] = {"input": None, "done": False}
    outage: dict[str, Any] = {}

    def _absorb(msg: Any) -> None:
        _sdk_outage(msg, outage)
        if isinstance(msg, cas.AssistantMessage):
            counters["turns"] += 1
            blocks: list[dict] = []
            for block in msg.content:
                if isinstance(block, cas.TextBlock):
                    parts.append(block.text)
                    blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, cas.ToolUseBlock):
                    counters["tool_calls"] += 1
                    blocks.append({"type": "tool_use", "id": block.id,
                                   "name": block.name, "input": block.input})
                    if on_tool_use is not None:
                        try:
                            on_tool_use(block.name, block.input)
                        except Exception:  # noqa: BLE001 — never break the loop
                            pass
                    if stop_tool and (block.name == stop_tool
                                      or block.name.endswith(f"__{stop_tool}")):
                        if stop_box["input"] is None:
                            stop_box["input"] = (dict(block.input)
                                                 if isinstance(block.input, dict)
                                                 else block.input)
                        stop_box["done"] = True
            if blocks:
                transcript.append({"role": "assistant", "content": blocks})
        elif isinstance(msg, cas.UserMessage):
            tr: list[dict] = []
            for block in getattr(msg, "content", None) or []:
                if isinstance(block, cas.ToolResultBlock):
                    tr.append({"type": "tool_result",
                               "tool_use_id": getattr(block, "tool_use_id", None),
                               "content": getattr(block, "content", None)})
            if tr:
                transcript.append({"role": "user", "content": tr})
        u_in, u_out = _sdk_usage_tokens(getattr(msg, "usage", None))
        if u_in is not None:
            usage_box["input"] = u_in
        if u_out is not None:
            usage_box["output"] = u_out

    async def _collect() -> None:
        parts.clear()
        transcript.clear()
        counters["tool_calls"] = counters["turns"] = 0
        usage_box["input"] = usage_box["output"] = None
        stop_box["input"] = None
        stop_box["done"] = False
        outage.clear()
        agen = cas.query(prompt=prompt, options=opts)
        wall = OPUS_TIMEOUT
        if wall and wall > 0:
            try:
                with anyio.fail_after(wall):
                    async for msg in agen:
                        _absorb(msg)
                        if stop_box["done"]:
                            break
            except TimeoutError as e:
                raise OpusTimeoutError(
                    f"opus SDK tool loop exceeded OVFX_OPUS_TIMEOUT={wall:.0f}s "
                    f"and was aborted; {counters['tool_calls']} tool call(s), "
                    f"{counters['turns']} turn(s) before the wall") from e
        else:
            async for msg in agen:
                _absorb(msg)
                if stop_box["done"]:
                    break

    def _session_fail(kind: str, exc: Exception, attempts: int) -> dict[str, Any]:
        return {"ok": False, "stop_input": None, "text": "".join(parts).strip(),
                "transcript": list(transcript),
                "tool_calls": counters["tool_calls"], "turns": counters["turns"],
                "usage": (None, None),
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "error_kind": kind, "attempts": attempts,
                "resets_at": outage.get("resets_at") if kind == "quota" else None}

    scrub = _OAUTH_SCRUB_ENV
    saved = {k: os.environ.pop(k, None) for k in scrub}
    try:
        opts = cas.ClaudeAgentOptions(
            model=OPUS_MODEL,
            cli_path=_claude_cli_path(),
            setting_sources=[],
            max_buffer_size=SDK_MAX_BUFFER_SIZE,
            allowed_tools=list(allowed_tools or []),
            mcp_servers=dict(mcp_servers or {}),
            max_turns=max_turns,
            system_prompt=system or "",
            env={k: "" for k in scrub},
            cwd=cwd,
        )
        pause = "opus" in config.pause_backends()
        retry_cap = (config.opus_backend_retry_max() if pause
                     else OPUS_SDK_DEFAULT_RETRIES)
        pc = len(prompt) + len(system or "")
        attempt = 0
        transient = 0
        maxturns_retried = False
        while True:
            attempt += 1
            t0 = time.monotonic()
            try:
                anyio.run(_collect)
            except Exception as e:  # noqa: BLE001 — classify + bounded retry
                _ledger_record(
                    role=role, model=OPUS_MODEL, backend=backend,
                    prompt_chars=pc,
                    duration_ms=1000 * (time.monotonic() - t0),
                    ok=False, attempt=attempt,
                    error=f"{type(e).__name__}: {e}"[:300], estimated=True)
                kind = outage.get("kind") or _classify_sdk_error(e)
                if done is not None and done():      # the caller already has its answer (an accepted finish): a
                    logger.warning("opus SDK session: attempt %d failed (%s) after the caller was done; "
                                   "not re-run", attempt, kind)     # re-run would start on a gate that says 'finished'
                    return {"ok": True, "stop_input": stop_box["input"], "text": "".join(parts).strip(),
                            "transcript": list(transcript), "tool_calls": counters["tool_calls"],
                            "turns": counters["turns"], "usage": (None, None), "error": None,
                            "error_kind": None, "attempts": attempt}
                if kind in ("quota", "auth"):        # the account refused: retrying only burns the caller
                    logger.warning("opus SDK session: account refused (kind=%s, resets_at=%s) — not "
                                   "retried; caller pauses", kind, outage.get("resets_at"))
                    return _session_fail(kind, e, attempt)
                if kind == "cli_missing":
                    logger.warning("opus SDK session: Claude Code CLI missing "
                                   "— cannot use opus; caller degrades")
                    return _session_fail(kind, e, attempt)
                if kind == "max_turns" and not maxturns_retried:
                    maxturns_retried = True
                    logger.warning("opus SDK session: max-turns flake — "
                                   "retrying once")
                    continue
                if kind == "max_turns":
                    return _session_fail(kind, e, attempt)
                transient += 1
                if transient > retry_cap:
                    logger.warning(
                        "opus SDK session: %d transient retries exhausted "
                        "(kind=%s, paused=%s) — degrading; caller falls back",
                        retry_cap, kind, pause)
                    return _session_fail(kind, e, attempt)
                delay = min(OPUS_SDK_BACKOFF_CAP,
                            OPUS_SDK_BACKOFF_MIN
                            * (2 ** min(transient - 1, 6)))
                logger.warning(
                    "opus SDK session: transient failure (%s: %s) — backoff-"
                    "and-retry (%d/%d, sleeping %.0fs)", kind, str(e)[:120],
                    transient, retry_cap, delay)
                time.sleep(delay)
                continue
            u_in, u_out = usage_box["input"], usage_box["output"]
            have = u_in is not None and u_out is not None
            _ledger_record(
                role=role, model=OPUS_MODEL, backend=backend,
                prompt_chars=pc, duration_ms=1000 * (time.monotonic() - t0),
                ok=True, attempt=attempt,
                input_tokens=u_in if have else None,
                output_tokens=u_out if have else None, estimated=not have)
            return {"ok": True, "stop_input": stop_box["input"],
                    "text": "".join(parts).strip(),
                    "transcript": list(transcript),
                    "tool_calls": counters["tool_calls"],
                    "turns": counters["turns"],
                    "usage": (u_in, u_out), "error": None,
                    "error_kind": None, "attempts": attempt}
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


#: The old spelling, kept for ``agents/ideator.py`` and the billing/ledger
#: assertions in ``tests/test_l3_opus_sdk.py``. It is the SAME function, not a
#: forwarder: the name was always wrong (the SDK owns the turns, so this was
#: never a loop) and a forwarding body only made that harder to see.
opus_sdk_agent_loop = sdk_session


def query_tier(tier: str, prompt: str, system: str | None = None,
               temperature: float = 0.7, max_tokens: int = 16000,
               json_mode: bool = False, role: str | None = None) -> str:
    """One LLM call at an explicit tier ('qwen' | 'opus'). Sync, blocking.

    qwen tier: infinite-patience pause-and-wait on backend-down errors; a
    rate-limit-shaped error raises QwenFallbackProxyError (qwen has no rate
    limits — that error means the old fallback proxy is in use).
    """
    if json_mode:
        prompt = prompt + "\n\nRespond with a single JSON object, nothing else."

    if tier == "opus":
        path, model, client = resolve_opus_backend()
        if path == "oauth":
            # The SDK path skipped the budget check entirely, so an over-budget
            # prompt (the analyst's ~130K-token payload) shipped unchecked and
            # failed downstream inside the subprocess. Same guard as every other
            # backend — the budget is a property of the MODEL, not the transport.
            _check_prompt_budget(model, system,
                                 [{"role": "user", "content": prompt}])
            return _opus_via_agent_sdk(prompt, system=system,
                                       max_tokens=max_tokens,
                                       temperature=temperature)
        if client is not None:
            _check_prompt_budget(model, system,
                                 [{"role": "user", "content": prompt}])
            _l_t0 = time.monotonic()
            _l_pc = _estimate_prompt_chars(system,
                                           [{"role": "user", "content": prompt}])
            try:
                resp = client.messages.create(
                    model=model, max_tokens=max_tokens, temperature=temperature,
                    system=system or anthropic.NOT_GIVEN,
                    messages=[{"role": "user", "content": prompt}])
                _l_usage = getattr(resp, "usage", None)
                _ledger_record(
                    role=role or "opus", model=model,
                    backend="anthropic-api_key",
                    prompt_chars=_l_pc,
                    duration_ms=1000 * (time.monotonic() - _l_t0),
                    ok=True, attempt=1,
                    input_tokens=getattr(_l_usage, "input_tokens", None),
                    output_tokens=getattr(_l_usage, "output_tokens", None),
                )
            except Exception as e:
                _ledger_record(
                    role=role or "opus", model=model,
                    backend="anthropic-api_key",
                    prompt_chars=_l_pc,
                    duration_ms=1000 * (time.monotonic() - _l_t0),
                    ok=False, attempt=1,
                    error=str(e)[:300],
                )
                raise
            return _extract_text(resp)
        # path == "qwen": fall through to the qwen tier
        tier = "qwen"

    if tier != "qwen":
        raise ValueError(f"Unknown tier {tier!r}; expected 'qwen' or 'opus'")

    _attempt = 0
    while True:
        _attempt += 1
        client = qwen_client()
        _t0 = time.monotonic()
        _prompt_chars = _estimate_prompt_chars(system,
                                               [{"role": "user", "content": prompt}])
        try:
            _check_prompt_budget(QWEN_MODEL, system,
                                 [{"role": "user", "content": prompt}])
            resp = client.messages.create(
                model=QWEN_MODEL, max_tokens=max_tokens, temperature=temperature,
                system=system or anthropic.NOT_GIVEN,
                messages=[{"role": "user", "content": prompt}])
            _usage = getattr(resp, "usage", None)
            _ledger_record(
                role=role, model=QWEN_MODEL, backend="qwen-proxy",
                prompt_chars=_prompt_chars,
                duration_ms=1000 * (time.monotonic() - _t0),
                ok=True, attempt=_attempt,
                input_tokens=getattr(_usage, "input_tokens", None),
                output_tokens=getattr(_usage, "output_tokens", None),
            )
        except Exception as e:
            _dur = 1000 * (time.monotonic() - _t0)
            if _is_rate_limit_error(e):
                _ledger_record(
                    role=role, model=QWEN_MODEL, backend="qwen-proxy",
                    prompt_chars=_prompt_chars, duration_ms=_dur,
                    ok=False, attempt=_attempt,
                    error="QwenFallbackProxyError: " + str(e)[:200],
                )
                raise QwenFallbackProxyError(
                    "RATE-LIMIT ERROR ON THE QWEN TIER. Local qwen has NO rate "
                    "limits — this means requests are going through the OLD "
                    f"fallback proxy (paid OpenCode route). Current proxy: "
                    f"{QWEN_PROXY_URL}. Point OVFX_QWEN_PROXY_URL at the "
                    "no-fallback proxy (running_qwen/hybrid_proxy_nofallback.py, "
                    f"port 12346). Original error: {e}") from e
            # A local-relay 401/403 is an OUTAGE (see _is_local_backend_auth_error):
            # qwen is a preemptible slurm job, so wait it out on the SAME bounded
            # backoff a connection error gets — never raise it out of L2.
            _local_auth = _is_local_backend_auth_error(e)
            if _is_backend_down_error(e) or _local_auth:
                _ledger_record(
                    role=role, model=QWEN_MODEL, backend="qwen-proxy",
                    prompt_chars=_prompt_chars, duration_ms=_dur,
                    ok=False, attempt=_attempt,
                    error=("backend_down(local_auth): " if _local_auth
                           else "backend_down: ") + str(e)[:200],
                )
                time.sleep(_qwen_note_503(e))
                _qwen_probe_gate(e)
                continue
            _ledger_record(
                role=role, model=QWEN_MODEL, backend="qwen-proxy",
                prompt_chars=_prompt_chars, duration_ms=_dur,
                ok=False, attempt=_attempt,
                error=str(e)[:300],
            )
            raise
        _qwen_episode_reset()  # real 200 on the completion call ends the episode
        return _extract_text(resp)


def query_role(role: str, prompt: str, system: str | None = None,
               temperature: float = 0.7, max_tokens: int = 16000,
               json_mode: bool = False, tier: str | None = None) -> str:
    """Query at the tier configured for `role` in config.ROLE_TIERS.

    `tier` (optional) forces a specific tier for THIS call, overriding
    ROLE_TIERS — used by query_json's opus->qwen fallback for the L2/analyst
    judgment roles so a flaky opus call can degrade to qwen without changing the
    role's configured tier. Defaults to `tier_for_role(role)`.
    """
    return query_tier(tier or tier_for_role(role), prompt, system=system,
                      temperature=temperature, max_tokens=max_tokens,
                      json_mode=json_mode, role=role)


# ---------------------------------------------------------------------------
# qwen -> opus quality escalation ("if qwen is doing too bad use opus")
# ---------------------------------------------------------------------------
#
# The SDK's runtime judgment roles (analyst / L2 / L3) run on the FREE local
# qwen tier. When a qwen judgment call EXHAUSTS its bounded retries (JSON never
# parses; L3 narration-nudges / tool loop never converges) we make ONE final
# attempt on opus — the FAILING CALL ONLY, not the whole role. This is a NO-OP
# (stay on qwen, log a warning) when no REAL opus route is configured, so the
# research loop never crashes because opus is unreachable, and the credit_pool
# path (which would mis-bill through the bare messages API) counts as
# unavailable. A per-cycle cap (config.escalation_cap) bounds opus spend.

_escalation_lock = threading.Lock()
_escalation_count = 0
_escalation_sinks: list = []

# Roles whose opus tier degraded to qwen this cycle, role -> first reason.
# A PERMANENT 100% failure across the ideation/selection layer used to look
# exactly like a transient hiccup: one WARNING per call and nothing on the
# record. This is that record; run_cycle surfaces it in the cycle summary.
# The fallback POLICY is unchanged — only its visibility.
_degraded_roles: dict[str, str] = {}


def record_role_degradation(role: str, reason: str) -> None:
    """Note that *role* ran off its intended tier. First reason wins."""
    with _escalation_lock:
        _degraded_roles.setdefault(str(role), str(reason))


def degraded_roles() -> list[str]:
    """Sorted roles that degraded off opus this cycle (empty = none)."""
    with _escalation_lock:
        return sorted(_degraded_roles)


def degraded_role_reasons() -> dict[str, str]:
    """role -> the first recorded degradation reason."""
    with _escalation_lock:
        return dict(_degraded_roles)


def opus_sdk_importable() -> bool:
    """True iff ``claude_agent_sdk`` can be imported in THIS interpreter.

    The oauth route is driven entirely through that package, and the import at
    ``_opus_via_agent_sdk`` is function-local and LAZY. Without this probe
    ``opus_route_available()`` answered True unconditionally on the oauth
    branch, so the clean single-line degrade in ``query_json`` never fired and
    every call took the mid-flight exception path instead — one WARNING, then
    silence, forever (live cycle 2026-08-01, launched under an interpreter
    without the package). find_spec, not import: a probe must not pay the
    import cost or run module-level side effects.
    """
    import importlib.util
    try:
        return importlib.util.find_spec("claude_agent_sdk") is not None
    except (ImportError, ValueError):  # namespace/broken package
        return False


def opus_route_available() -> bool:
    """True iff a REAL opus route is usable right now.

    Usable when OVFX_OPUS_PATH=oauth (subscription route via claude_agent_sdk,
    client is None) AND that package is actually importable, OR api_key with
    ANTHROPIC_API_KEY (bare messages client). credit_pool (blocked/costly —
    raises) and qwen (not opus; client is None) both count as UNAVAILABLE, so
    escalation degrades to a logged no-op.
    """
    try:
        path, _model, client = resolve_opus_backend()
    except Exception:
        return False
    if path == "oauth":
        return opus_sdk_importable()
    return client is not None


def opus_bare_client_available() -> bool:
    """True iff opus is reachable as a BARE anthropic messages client.

    Distinct from ``opus_route_available()``. The L3 editor drives a hand-rolled
    tool loop (``client.messages.create(..., tools=TOOLS)``) and therefore needs
    a REAL bare client — which only the api_key path yields. Under
    OVFX_OPUS_PATH=oauth the ideators reach opus through claude_agent_sdk
    (client is None); that agent-SDK path CANNOT drive the editor's tool loop, so
    it counts as UNAVAILABLE here even though ``opus_route_available()`` is True
    for the ideators. This is the precise reason L3 editors silently degraded to
    qwen under oauth (resolve_opus_backend returns a None client).
    """
    try:
        _path, _model, client = resolve_opus_backend()
    except Exception:
        return False
    return client is not None


# One-time (per process) first-class notice that the L3 editor asked for opus
# but no BARE-client opus route is configured, so it runs on qwen. The editor
# requests opus ~35x/run (prefer_opus big directions + qwen-failure escalations);
# a per-call warning is pure noise, so the actionable remediation is logged ONCE
# and thereafter this is a known, bounded condition (not a silent degrade).
_l3_opus_notice_lock = threading.Lock()
_l3_opus_notice_emitted = False


def note_l3_opus_unavailable(context: str = "") -> None:
    """Log — ONCE per process — that L3 requested opus with no bare-client route.

    Makes 'opus-unavailable-for-L3' an EXPLICIT, BOUNDED, first-class condition
    with actionable remediation, instead of a silent per-call degrade. First call
    logs a WARNING with the fix; later calls log at DEBUG. Reset via
    ``reset_l3_opus_notice()`` (cycle start / test hook)."""
    global _l3_opus_notice_emitted
    with _l3_opus_notice_lock:
        first = not _l3_opus_notice_emitted
        _l3_opus_notice_emitted = True
    tag = f"[{context}] " if context else ""
    if first:
        logger.warning(
            "L3 EDITOR OPUS UNAVAILABLE (OVFX_OPUS_PATH=%s): %sthe editor's tool "
            "loop needs a BARE anthropic client, which only the api_key path "
            "provides; L3 edits run on qwen this run. To route L3 onto opus set "
            "OVFX_OPUS_PATH=api_key + ANTHROPIC_API_KEY=<key> + "
            "OVFX_OPUS_URL=<endpoint>. Ideators are unaffected (oauth agent-SDK "
            "path). This notice is logged once per run.", OPUS_PATH, tag)
    else:
        logger.debug("L3 opus still unavailable %s(staying on qwen)", tag)


def reset_l3_opus_notice() -> None:
    """Reset the one-time L3-opus-unavailable notice (cycle start / test hook)."""
    global _l3_opus_notice_emitted
    with _l3_opus_notice_lock:
        _l3_opus_notice_emitted = False


def register_escalation_sink(fn) -> None:
    """Register fn(event: dict) called on every escalation (orchestrator wires
    its Journal here so the event is journaled as well as observer-logged)."""
    _escalation_sinks.append(fn)


def reset_escalation_state() -> None:
    """Reset the per-cycle escalation counter and sinks (call at cycle start;
    also a test hook)."""
    global _escalation_count
    with _escalation_lock:
        _escalation_count = 0
        _escalation_sinks.clear()
        _degraded_roles.clear()


def escalation_count() -> int:
    """Number of qwen->opus escalations recorded this cycle (dashboard/status)."""
    return _escalation_count


def _emit_escalation_event(**fields: Any) -> None:
    """Log + observer + journal-sink an escalation event, all best-effort."""
    logger.warning("MODEL ESCALATION: role=%s reason=%s qwen->opus "
                   "(call=%s, #%s this cycle)",
                   fields.get("role"), fields.get("reason"),
                   fields.get("call_id"), fields.get("escalation_n"))
    try:  # observer is a process-global singleton; never break on absence
        from observer import get_observer
        get_observer().log("model_escalation", **fields)
    except Exception:
        pass
    event = {"event": "model_escalation", **fields}
    for sink in list(_escalation_sinks):
        try:
            sink(dict(event))
        except Exception:  # a broken sink must never break the research loop
            pass


def maybe_escalate_to_opus(role: str, reason: str,
                           call_id: str | None = None) -> bool:
    """Gate a single qwen->opus escalation for a retry-exhausted judgment call.

    Returns True (and records + emits the escalation event) iff a real opus
    route exists AND the per-cycle cap is not yet reached; otherwise logs and
    returns False (caller stays on qwen). Escalates the FAILING CALL ONLY.
    """
    global _escalation_count
    call_id = call_id or uuid.uuid4().hex[:8]
    if not opus_route_available():
        logger.warning("escalation NO-OP (no opus route: OVFX_OPUS_PATH=%s): "
                       "role=%s reason=%s stays on qwen",
                       OPUS_PATH, role, reason)
        return False
    cap = config.escalation_cap()
    with _escalation_lock:
        if _escalation_count >= cap:
            logger.warning("escalation CAP reached (%d/%d): role=%s reason=%s "
                           "stays on qwen", _escalation_count, cap, role, reason)
            return False
        _escalation_count += 1
        n = _escalation_count
    _emit_escalation_event(role=role, reason=reason, call_id=call_id,
                           escalation_n=n, **{"from": "qwen", "to": "opus"})
    return True

JSON_REASK_ATTEMPTS = 3  # total attempts per judgment call (1 + 2 re-asks)

# Structured-JSON judgment roles that default to opus (ROLE_TIERS) but MUST fall
# back to qwen rather than strand a lineage when the flaky OAuth opus subprocess
# is unavailable or a call fails/times out. The L2 lineage-owner
# (derive_spec / should_stop / render_verdict) and the analyst (select_survivors)
# — low-volume, high-judgment. The ideator is NOT here: it defaults to opus but
# owns its own degrade path (agents/ideator.py), so query_json must keep raising
# for it (no silent qwen fallback).
_QWEN_FALLBACK_ROLES = frozenset({"l2", "analyst"})


def _is_shell_leakage(text: str) -> bool:
    """Detect qwen bash/dispatch leakage (shell commands or agent-dispatch prose)."""
    first_line = text.split("\n")[0].strip().lower()
    if first_line in ("bash", "sh") or text.startswith("#!"):
        return True
    if "$(" in text or "python3 " in text:
        return True
    stripped = text.strip()
    if stripped.startswith("Agent:") or stripped.startswith("I need to"):
        return True
    return False


def query_json(role: str, prompt: str, system: str | None = None,
               temperature: float = 0.7, max_tokens: int = 16000,
               attempts: int = JSON_REASK_ATTEMPTS) -> dict[str, Any]:
    """Query and parse a JSON object response, with BOUNDED re-ask on non-JSON.

    Known qwen failure mode: prose/markdown narration instead of the JSON
    object. On a parse failure we re-ask (up to `attempts` total calls) with
    an explicit repair instruction quoting the malformed reply, at reduced
    temperature. Exhaustion raises NonJSONResponseError.
    """
    # Effective tier with a robust qwen fallback for the low-volume, high-judgment
    # structured roles (L2 lineage reasoning + analyst survivor selection). They
    # default to opus but MUST NOT strand a lineage when the flaky OAuth opus
    # subprocess is down: an unavailable route degrades to qwen up front, and a
    # per-call opus EXCEPTION (timeout / subprocess / backend) degrades to qwen
    # mid-flight. (A pure non-JSON reply from opus is handled by the normal re-ask
    # loop; opus rarely mis-formats.)
    cur_tier = tier_for_role(role)
    opus_fallback = role in _QWEN_FALLBACK_ROLES and cur_tier == "opus"
    if opus_fallback and not opus_route_available():
        logger.info("query_json[%s]: opus route unavailable -> running on qwen "
                    "(no-strand fallback)", role)
        record_role_degradation(role, "opus route unavailable at dispatch")
        cur_tier = "qwen"
        opus_fallback = False

    last_err: Exception | None = None
    bad_text = ""
    for attempt in range(1, attempts + 1):
        if attempt == 1:
            p = prompt + "\n\nRespond with exactly one JSON object, no fences or prose."
            temp = temperature
        else:
            p = (prompt
                 + "\n\nIMPORTANT — your previous reply was NOT parseable JSON."
                 + f"\nIt began: {bad_text[:400]!r}"
                 + "\nRespond with ONLY one valid JSON object. No prose, no "
                   "markdown fences, no commentary before or after it.")
            if "<tool_call" in bad_text:
                p += ("\nDo NOT emit <tool_call>/<tool_response> tags or any "
                      "function-calling syntax — there are no tools here; "
                      "write the JSON object as plain text.")
            if _is_shell_leakage(bad_text):
                p += ("\nYou cannot run commands or dispatch agents — "
                      "there is no shell. Return ONLY the JSON object, "
                      "no code fences, no prose.")
            temp = min(temperature, 0.2)
            logger.warning("query_json[%s]: re-ask %d/%d after non-JSON reply",
                           role, attempt, attempts)
        try:
            text = query_role(role, p, system=system, temperature=temp,
                              max_tokens=max_tokens, json_mode=True, tier=cur_tier)
        except Exception as e:  # noqa: BLE001
            # A fallback role's opus call FAILED (timeout / CLI hang / backend).
            # Degrade to qwen for this attempt and all remaining ones rather than
            # stranding the lineage; anything else propagates unchanged.
            if opus_fallback and cur_tier == "opus":
                logger.warning("query_json[%s]: opus call failed (%s) -> qwen "
                               "fallback (no-strand)", role, str(e)[:160])
                record_role_degradation(
                    role, f"opus call failed: {str(e)[:160]}")
                cur_tier = "qwen"
                opus_fallback = False
                text = query_role(role, p, system=system, temperature=temp,
                                  max_tokens=max_tokens, json_mode=True,
                                  tier="qwen")
            else:
                raise
        try:
            return parse_json_text(text)
        except NonJSONResponseError as e:
            last_err, bad_text = e, text

    # qwen exhausted its bounded JSON re-asks. If this role runs on qwen (i.e.
    # is not already an opus role / manually forced to opus) and a real opus
    # route exists under the per-cycle cap, make ONE final opus attempt.
    if tier_for_role(role) == "qwen" and maybe_escalate_to_opus(
            role, reason="json_parse_exhausted"):
        repair = (prompt
                  + "\n\nIMPORTANT — a prior model produced non-parseable JSON."
                  + f"\nIt began: {bad_text[:400]!r}"
                  + "\nRespond with ONLY one valid JSON object. No prose, no "
                    "markdown fences, no commentary before or after it.")
        try:
            otext = query_tier("opus", repair, system=system,
                               temperature=min(temperature, 0.2),
                               max_tokens=max_tokens, json_mode=True)
            return parse_json_text(otext)
        except NonJSONResponseError as e:
            last_err, bad_text = e, otext
        except Exception as e:  # opus route hiccup — fall through to the raise
            logger.warning("query_json[%s]: opus escalation attempt failed: %s",
                           role, str(e)[:160])

    raise NonJSONResponseError(
        f"non-JSON reply from role={role!r} after {attempts} attempts: "
        f"{bad_text[:200]}") from last_err


def _unwrap_native_tool_call(obj: dict[str, Any]) -> dict[str, Any]:
    """Unwrap qwen-native tool-call leakage.

    Observed live failure mode: instead of the requested JSON, qwen emits its
    NATIVE function-calling format — <tool_call>{"name": ..., "arguments":
    {...}}</tool_call> — with the intended payload inside "arguments".
    """
    if set(obj) <= {"name", "arguments"} and isinstance(
            obj.get("arguments"), dict):
        logger.info("parse_json_text: unwrapped qwen-native tool_call "
                    "envelope (name=%r)", obj.get("name"))
        return obj["arguments"]
    return obj


def parse_json_text(text: str) -> dict[str, Any]:
    """Strip markdown fences / native tool_call tags and parse JSON."""
    if "<tool_call" in text:
        # qwen-native tool-call leakage: strip the tags, keep the payload
        text = (text.replace("<tool_call>", " ")
                    .replace("</tool_call>", " ").strip())
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0].strip()
    elif "```bash" in text:
        text = text.split("```bash")[-1].split("```")[0].strip()
    elif "```sh" in text:
        text = text.split("```sh")[-1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return _unwrap_native_tool_call(json.loads(text))
    except json.JSONDecodeError as e:
        # Local repair: prose around a JSON object (no fences) — decode the
        # first balanced object found in the text before re-asking the model.
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(text[idx:])
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    logger.info("parse_json_text: repaired prose-wrapped JSON "
                                "(offset %d)", idx)
                    return _unwrap_native_tool_call(obj)
        logger.error("JSON parse failed: %s\nRaw: %s", str(e)[:100], text[:300])
        raise NonJSONResponseError(
            f"Agent returned non-JSON: {text[:200]}") from e
