"""agent.llm — ONE tool loop, two transports. Role routing, hold, escalation, usage.

WHY THIS EXISTS. The tree carried **two complete agent-loop implementations** for
one editor: ``agents/l3.py:_agent_loop`` (270 lines, bare ``anthropic`` messages)
and ``agents/l3.py:_agent_loop_sdk`` (91 lines, ``claude_agent_sdk`` over OAuth).
They were not two policies — they were one policy written twice, and the second
copy existed for exactly one reason: ``resolve_opus_backend`` returns a ``None``
client on the oauth path, so the hand-rolled ``client.messages.create`` loop had
nothing to call. Everything *around* that difference was duplicated: the
finish()-contract, the bounded narration nudge, the "an outage must not burn the
deliberation budget" rule, the shutdown check, the transcript logging. And they
had already drifted — the bare loop's exhaustion note says ``without finish()``
while the SDK twin says ``no finish() after N nudges``; the bare loop counts
turns against ``max_turns`` while the SDK twin counted only nudges; the bare loop
logged its transcript **twice** (``l3.py:453-468`` at ``rebuild/phase3`` is a
byte-identical copy of ``:437-452``) and streamed the narration nudge twice
(``:409-414`` again at ``:415-420``). Both duplications die here.

So the transport becomes a PARAMETER (plan §5: "the second loop dies with
``llm.py``'s transport parameter"). :class:`Loop` owns the policy; a transport
owns exactly one question — *how is one advance of the conversation executed* —
and answers it in ~40 lines each:

  * :class:`MessagesTransport`  — one ``client.messages.create``; the caller's
    :class:`~agents.l3_tools.ToolExecutor` runs each ``tool_use`` block and the
    results go back as the next user message. One advance = one model turn.
  * :class:`AgentSdkTransport`  — one ``claude_agent_sdk`` session whose
    in-process MCP tool server routes **back into the same executor**, so every
    guard (worktree scoping, locked paths, banned commands, ``validate_config``,
    the ``finish()`` contract) is the same code. One advance = one session.

That asymmetry is real and is not hidden: the SDK owns its own inner turn loop
and we do not get to interleave with it. One consequence is deliberate and is
the single behaviour change in this move — the turn budget now bounds the SDK
route's TOTAL turns, where the twin bounded only its nudges. In practice the two
agree, because a session that hits the SDK's own cap raises ``max_turns`` and
degrades rather than returning; where they differ, this is the bounded one. It
is also the only asymmetry — every policy decision below is made once, for
both.

WHAT THE LOOP REFUSES TO CONFLATE. A backend outage is not a failed edit. Both
old loops knew this and both said so in their own words; here it is one rule:
a transport error is classified before it is counted, an outage retries the SAME
advance without consuming the turn budget, and a PERMANENT outage terminates as
``backend_unavailable`` — an infra void that re-opens the lineage — never as
``apply_failed``, which would read as "this mechanism was tested and failed". It
is the same distinction ``agent.world`` turns on and the same one 93.6% of dead
lineages were on the wrong side of.

THE HOLD IS BOUNDED BY A DECLARATION, NOT BY AN ENV VAR. ``hold_for_backend`` is
deliberately unbounded in production — a down backend must never void a lineage.
But "unbounded" was a *constant* (``OVFX_QWEN_WAIT_MAX`` defaults to 0, and the
probe gate then polls every 30s forever), which means the only way to bound it
was to set an environment variable, which is what ``tests/behavioural/conftest.py``
has to do today to stop ``consolidate`` hanging past its 300s harness timeout.
:class:`Hold` makes the bound a **declared field** instead, and its default reads
that same env-derived constant — so nothing changes silently and the behaviour
that ships is byte-for-byte today's.

LAYERING, HONESTLY. This module imports the root ``llm`` module and reads its
attributes AT CALL TIME. That is the wrong direction for the finished tree and it
is temporary: ``llm.py`` still owns the backend primitives (clients, the serving
probe, the 503 episode, the error classifiers) and still has eight production
callers outside this commit's lane (``briefs``, ``ideator``, ``analyst``, ``l2``,
``orchestrator``, ``run_cycle``, ``main``, ``watchdog``). Re-homing those is
commits 11 and 13. Reading through the module rather than importing the names
also keeps the ~10 test files that ``monkeypatch.setattr(llm, ...)`` honest:
there is exactly ONE definition of each primitive and patching it patches what
this loop actually calls.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol, Sequence

from pydantic import BaseModel, ValidationError

import config
import llm
from llm import sdk_session
from llm_usage import aggregate as usage_rollup

logger = logging.getLogger(__name__)

__all__ = [
    "Role", "role_for", "query", "SchemaRefusal",
    "Loop", "LoopPolicy", "LoopHooks", "tools", "Advance",
    "MessagesTransport", "AgentSdkTransport", "Transport",
    "BackendExhausted", "sdk_tool_server", "sdk_session",
    "Hold", "hold_for_backend", "escalate", "usage_rollup",
]


# ---------------------------------------------------------------------------
# Roles — where one role's judgment is bought, and over which transport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Role:
    """One resolved routing decision: who is asking, at what tier, over what.

    ``transport`` is empty when the tier has no usable route at all (the
    ``credit_pool`` path, which would mis-bill through the bare messages API,
    and an unresolvable ``OVFX_OPUS_PATH``). An unroutable Role is a first-class
    answer — callers degrade rather than raise.
    """

    name: str
    tier: str
    transport: str = ""
    model: str = ""

    @property
    def routable(self) -> bool:
        return bool(self.transport)


def role_for(name: str, *, tier: str | None = None) -> Role:
    """Resolve ``name`` (or an explicit ``tier``) to a Role, transport included.

    The transport follows from the billing path and nothing else, which is the
    fact ``_resolve_opus_runner`` used to re-derive by hand at two call sites:
    ``api_key`` yields a real bare client (messages), ``oauth`` yields ``None``
    and is reachable only through ``claude_agent_sdk`` (agent_sdk), ``qwen``
    means the free local route, and ``credit_pool`` means no route.
    """
    effective = tier or llm.tier_for_role(name)
    if effective != "opus":
        return Role(name, "qwen", "messages", llm.QWEN_MODEL)
    try:
        path, model, client = llm.resolve_opus_backend()
    except Exception as e:  # noqa: BLE001 — an unroutable tier is an answer
        logger.debug("role_for(%s): opus unresolvable (%s)", name, str(e)[:120])
        return Role(name, "opus", "", "")
    if path == "qwen":
        return Role(name, "qwen", "messages", model)
    if path == "oauth":
        return Role(name, "opus", "agent_sdk", model)
    return Role(name, "opus", "messages" if client is not None else "", model)


# ---------------------------------------------------------------------------
# 1. query — one structured answer, validated against a pydantic schema
# ---------------------------------------------------------------------------


class SchemaRefusal(RuntimeError):
    """The reply parsed as JSON but did not satisfy the requested schema.

    Carries ``locs`` — the ``ValidationError.errors()[*].loc`` paths of the
    offending keys. That list is the whole reason the schema is a pydantic model
    and not a hand-rolled check: it names the exact key rather than the whole
    payload, which is the refusal behaviour ``l2-schema-grounding`` hand-rolls
    today and a documented contributor to 174 ``spec_invalid`` kills.
    """

    def __init__(self, msg: str, locs: Sequence[tuple] = ()) -> None:
        self.locs = list(locs)
        super().__init__(msg)


def _loc_str(loc: Sequence[Any]) -> str:
    return ".".join(str(p) for p in loc) or "<root>"


def query(role: str, prompt: str, *, schema: type[BaseModel],
          system: str | None = None, temperature: float = 0.7,
          max_tokens: int = 16000, attempts: int = 2) -> BaseModel:
    """Ask ``role`` for one object of type ``schema``. Returns a validated model.

    The schema is shown to the model as JSON Schema generated FROM the same
    class that validates the answer, so the prompt and the validator cannot
    drift apart. JSON-parse repair (fences, native ``<tool_call>`` leakage,
    prose-wrapped objects) and the qwen->opus escalation on exhaustion are
    already the single implementation in ``llm.query_json``; this adds the
    schema layer on top and re-asks ``attempts`` times naming the offending
    keys. Raises :class:`SchemaRefusal` when the shape never arrives, and lets
    ``llm.NonJSONResponseError`` through when the JSON never does.
    """
    rendered = json.dumps(schema.model_json_schema(), sort_keys=True)
    ask = f"{prompt}\n\nReturn ONE JSON object matching this schema:\n{rendered}"
    last: ValidationError | None = None
    for attempt in range(1, max(1, attempts) + 1):
        raw = llm.query_json(role, ask, system=system, temperature=temperature,
                             max_tokens=max_tokens)
        try:
            return schema.model_validate(raw)
        except ValidationError as e:
            last = e
            bad = "; ".join(f"{_loc_str(err['loc'])}: {err['msg']}"
                            for err in e.errors()[:8])
            logger.warning("query[%s]: schema re-ask %d/%d — %s", role,
                           attempt, attempts, bad[:300])
            ask = (f"{prompt}\n\nYour previous reply was valid JSON but did NOT "
                   f"satisfy the schema. These keys were wrong:\n  {bad}\n"
                   f"Return ONE JSON object matching this schema exactly:\n"
                   f"{rendered}")
    locs = [tuple(err["loc"]) for err in (last.errors() if last else [])]
    raise SchemaRefusal(
        f"role={role!r} never produced a valid {schema.__name__} in {attempts} "
        f"attempts; offending keys: {[_loc_str(loc) for loc in locs]}", locs)


# ---------------------------------------------------------------------------
# 2. THE tool loop — policy here, transport below
# ---------------------------------------------------------------------------


class BackendExhausted(RuntimeError):
    """A transport gave up after ITS OWN bounded retries. The caller degrades.

    Distinct from ``backend_unavailable``: that is the loop's terminal verdict
    for a permanent outage on the ONLY route, and it is a result. This is a
    signal that one route is unusable right now and another should be tried —
    the flaky OAuth CLI subprocess, measured failing/retrying on ~61% of opus
    calls, must never strand an edit that qwen could finish.
    """

    def __init__(self, msg: str, *, kind: str | None = None) -> None:
        self.kind = kind
        super().__init__(msg)


@dataclass
class Advance:
    """What one advance of a transport produced. The loop reads only this."""

    turns: int = 0
    tool_calls: int = 0
    #: the stop-tool payload, the instant the model called it. None = not yet.
    finish: dict | None = None
    #: (block_role, content) pairs to stream to the observer, in order.
    emitted: list[tuple[str, Any]] = field(default_factory=list)


class Transport(Protocol):
    """How one advance is executed. The ONLY thing the two routes disagree on."""

    kind: str
    #: True when an advance is one model turn worth streaming turn-by-turn.
    streams: bool

    def start(self, prompt: str) -> None: ...
    def advance(self, executor: Any) -> Advance: ...
    def push(self, text: str) -> None: ...
    def transcript(self) -> list[dict]: ...


@dataclass(frozen=True)
class LoopPolicy:
    """Every bound the loop enforces. Defaults reproduce ``agents/l3_core``."""

    max_turns: int = 200
    max_nudges: int = 3
    nudge: str = ""
    #: Bounded backend-down retries inside ONE session. A transient outage
    #: retries WITHOUT consuming the turn budget; only this many waits later
    #: does a PERMANENT outage terminate as the infra void.
    max_backend_retries: int = 40
    #: Opus has no serving probe to poll (it is a remote API), so a paused opus
    #: backs off exponentially and eventually degrades instead of holding.
    backoff_min: float = 5.0
    backoff_cap: float = 60.0
    temperature: float | None = 0.3
    max_tokens: int = 16000
    stop_tool: str = "finish"
    abort_status: str = "void_shutdown"


def _never() -> bool:
    return False


@dataclass(frozen=True)
class LoopHooks:
    """What the loop must ask someone else.

    Injected rather than imported so the CALLER's module-level names stay the
    single source of truth for backend classification and waiting. ``l3.py``
    binds these from its own globals, which is also what keeps the six test
    files that ``monkeypatch.setattr(l3, "wait_for_qwen"|"MAX_BACKEND_RETRIES"|
    "_is_backend_down_error", ...)`` measuring the thing they name.
    """

    is_rate_limit: Callable[[Exception], bool] = staticmethod(
        lambda e: False)
    is_backend_down: Callable[[Exception], bool] = staticmethod(
        lambda e: False)
    is_local_auth_outage: Callable[[Exception], bool] = staticmethod(
        lambda e: False)
    wait_for_backend: Callable[..., bool] | None = None
    should_abort: Callable[[], bool] = staticmethod(_never)
    paused_backends: Callable[[], frozenset] = staticmethod(
        lambda: frozenset({"qwen"}))
    opus_retry_max: Callable[[], int] = staticmethod(lambda: 6)


class Loop:
    """One multi-turn tool loop. Built by :func:`tools`; run once per phase.

    The turn budget is a DELIBERATION budget and nothing else spends it. That is
    the c7 autopsy in one sentence: a 503 window drained a 200-turn budget with
    the message history frozen and killed ~35 code-change trials that were fine.
    """

    def __init__(self, role: Role, tool_defs: Sequence[dict], system: str, *,
                 transport: Transport, policy: LoopPolicy = LoopPolicy(),
                 hooks: LoopHooks = LoopHooks()) -> None:
        self.role = role
        self.tool_defs = list(tool_defs)
        self.system = system
        self.transport = transport
        self.policy = policy
        self.hooks = hooks

    # -- the loop ----------------------------------------------------------

    def run(self, prompt: str, executor: Any, *, track: str = "",
            lineage_id: str = "", phase: str = "") -> dict:
        """Drive the loop to a ``finish()`` payload. Always returns a dict."""
        pol, hooks = self.policy, self.hooks
        self.transport.start(prompt)
        observer = _observer()
        t0 = time.time()
        label = "-".join(p for p in (self.role.name, track, phase) if p)
        if self.transport.kind != "messages":
            label = f"{label}-opus-sdk"
            phase_tag = f"{phase}-opus-sdk"
        else:
            phase_tag = phase

        stream = None
        if self.transport.streams:
            stream = _best_effort(observer, "stream_begin", agent=label,
                                  track=track, lineage_id=lineage_id,
                                  role=self.role.name, model=self.role.model,
                                  prompt=prompt, phase=phase)

        finish: dict | None = None
        turns = nudges = backend_retries = opus_retries = 0
        while turns < pol.max_turns and finish is None:
            # INTERRUPTIBLE at the TOP of every advance, so a shutdown lands at
            # the next boundary rather than after the whole turn budget. The
            # live incident burned ~58 minutes inside one phase.
            if hooks.should_abort():
                logger.warning(
                    "%s[%s/%s]: %s INTERRUPTED by shutdown at turn %d/%d — "
                    "abandoning the remaining turn budget", self.role.name,
                    track, lineage_id, phase, turns, pol.max_turns)
                return {"status": pol.abort_status,
                        "notes": (f"editor {phase} phase interrupted by "
                                  f"shutdown at turn {turns}/{pol.max_turns}")}
            try:
                step = self.transport.advance(executor)
            except Exception as e:  # noqa: BLE001 — classify, then decide
                verdict, backend_retries, opus_retries = self._on_error(
                    e, track, lineage_id, phase, turns, backend_retries,
                    opus_retries)
                if verdict is not None:
                    finish = verdict
                    break
                continue  # retry the SAME advance — turn budget NOT consumed
            turns += max(0, step.turns)
            for block_role, content in step.emitted:
                if stream is not None:
                    _best_effort(observer, "stream_turn", stream,
                                 {"role": block_role, "content": content},
                                 block_role=block_role)
            if step.finish is not None:
                finish = _coerce_finish(step.finish)
                break
            if step.tool_calls:
                nudges = 0
                continue
            # Narration instead of tool calls (the known qwen failure mode, and
            # the SDK's early-stop): bounded nudge, then fail. ONE rule now —
            # the two loops phrased this refusal differently ("narrated text
            # without tool calls" vs "produced no finish()"), so a caller
            # reading `notes` could not tell which one had answered. The single
            # note names both, and the transport that produced it.
            nudges += 1
            if nudges > pol.max_nudges:
                finish = {
                    "status": "apply_failed",
                    "notes": (f"agent narrated text without tool calls and "
                              f"produced no finish() for {nudges} consecutive "
                              f"turns (nudged {pol.max_nudges}x, "
                              f"transport={self.transport.kind})")}
                break
            logger.warning("%s[%s/%s]: %s turn %d produced NO tool call — "
                           "nudging (%d/%d)", self.role.name, track, lineage_id,
                           phase, turns, nudges, pol.max_nudges)
            self.transport.push(pol.nudge)
            if stream is not None:
                _best_effort(observer, "stream_turn", stream,
                             {"role": "user", "content": pol.nudge},
                             block_role="user")

        outcome = (finish or {}).get("status", "turns_exhausted")
        _best_effort(observer, "log", "l3_phase", track=track,
                     lineage_id=lineage_id, role=self.role.name, phase=phase_tag,
                     turns=turns, max_turns=pol.max_turns, nudges=nudges,
                     backend_retries=backend_retries, outcome=outcome)
        if stream is not None:
            _best_effort(observer, "stream_end", stream,
                         success=finish is not None)
        # The full turn-by-turn transcript, ONCE. The bare loop logged this
        # block TWICE — `l3.py:453-468` at `rebuild/phase3` is a byte-identical
        # copy of `:437-452` — so every phase wrote two agent_details records
        # and the dashboard replayed the loop twice.
        _best_effort(observer, "log_agent_call", agent=label, track=track,
                     lineage_id=lineage_id, role=self.role.name,
                     model=self.role.model, duration=time.time() - t0,
                     prompt_preview=prompt[:200],
                     response_preview=json.dumps(finish or {})[:200],
                     full_prompt=prompt,
                     full_response=json.dumps(
                         finish or {"status": "turns_exhausted"}, indent=2),
                     turns=self.transport.transcript())
        logger.info("%s[%s/%s]: %s used %d/%d turns (backend_retries=%d, "
                    "transport=%s) -> %s", self.role.name, track, lineage_id,
                    phase, turns, pol.max_turns, backend_retries,
                    self.transport.kind, outcome)
        if finish is None:
            # GENUINE deliberation-exhaustion: the backend was up and the editor
            # failed. Distinct from the backend drain, which returned earlier as
            # backend_unavailable. Name the retry count so attribution is
            # possible from the note alone.
            return {"status": "apply_failed",
                    "notes": (f"agent exhausted {pol.max_turns} turns without "
                              f"finish() (genuine deliberation-exhaustion; "
                              f"backend_retries={backend_retries} — no backend "
                              f"outage involved)")}
        return finish

    # -- the one place a transport error is turned into a decision ---------

    def _on_error(self, exc: Exception, track: str, lineage_id: str, phase: str,
                  turns: int, backend_retries: int, opus_retries: int
                  ) -> tuple[dict | None, int, int]:
        """Classify ``exc``. Return (terminal_verdict | None, retries, retries).

        ``None`` means "retry the same advance"; the turn budget is untouched
        either way. Anything not classified here is re-raised so the caller can
        degrade — a route that cannot answer is not the same as an edit that
        cannot be made.
        """
        pol, hooks = self.policy, self.hooks
        tier = self.role.tier
        if tier == "qwen" and hooks.is_rate_limit(exc):
            # Local qwen has NO rate limits: a rate-limit shape means the old
            # PAID fallback proxy is answering. That must never be waited out.
            raise llm.QwenFallbackProxyError(
                f"rate-limit-shaped error in the {self.role.name} loop — old "
                f"fallback proxy in use? proxy={llm.QWEN_PROXY_URL}: {exc}"
            ) from exc
        local_auth = tier == "qwen" and hooks.is_local_auth_outage(exc)
        if tier == "qwen" and (hooks.is_backend_down(exc) or local_auth):
            backend_retries += 1
            if backend_retries > pol.max_backend_retries:
                logger.error(
                    "%s[%s/%s]: qwen backend down — %d waits exhausted, giving "
                    "up as backend_unavailable (turn %d/%d budget intact)",
                    self.role.name, track, lineage_id, backend_retries - 1,
                    turns, pol.max_turns)
                return ({"status": "backend_unavailable",
                         "notes": (f"qwen backend unavailable: "
                                   f"{pol.max_backend_retries} backend-down "
                                   f"waits exhausted without recovery; last "
                                   f"error: {exc}")},
                        backend_retries, opus_retries)
            logger.warning(
                "%s[%s/%s]: qwen down mid-loop — pause-and-wait (backend retry "
                "%d/%d; turn budget %d/%d NOT consumed)", self.role.name, track,
                lineage_id, backend_retries, pol.max_backend_retries, turns,
                pol.max_turns)
            wait = hooks.wait_for_backend or llm.wait_for_qwen
            # The wait is unbounded by design; hand it the shutdown predicate so
            # a SIGTERM lands within a poll slice instead of being slept through.
            if not wait(should_abort=hooks.should_abort):
                if hooks.should_abort():
                    return ({"status": pol.abort_status,
                             "notes": (f"editor {phase} phase interrupted by "
                                       f"shutdown during a qwen backend-down "
                                       f"pause-and-wait (turn {turns}/"
                                       f"{pol.max_turns})")},
                            backend_retries, opus_retries)
                return ({"status": "backend_unavailable",
                         "notes": (f"qwen backend unavailable: the wait gave up "
                                   f"on backend retry {backend_retries}; last "
                                   f"error: {exc}")},
                        backend_retries, opus_retries)
            return None, backend_retries, opus_retries
        if (tier == "opus" and hooks.is_backend_down(exc)
                and "opus" in hooks.paused_backends()):
            # PAUSE-NOT-DEGRADE. Raising on a transient 503 would degrade a
            # deliberately-chosen opus route to qwen; a PERMANENT outage still
            # terminates after the configured cap so the caller falls back.
            opus_retries += 1
            cap = hooks.opus_retry_max()
            if opus_retries > cap:
                logger.warning(
                    "%s[%s/%s]: opus backend down — %d backoff retries "
                    "exhausted; raising so the phase degrades (turn %d/%d "
                    "budget intact)", self.role.name, track, lineage_id, cap,
                    turns, pol.max_turns)
                raise exc
            delay = min(pol.backoff_cap,
                        pol.backoff_min * (2 ** min(opus_retries - 1, 6)))
            logger.warning(
                "%s[%s/%s]: opus backend down mid-loop (%s) — pause-and-retry "
                "(backoff %d/%d, sleeping %.0fs; turn budget %d/%d NOT "
                "consumed)", self.role.name, track, lineage_id, str(exc)[:120],
                opus_retries, cap, delay, turns, pol.max_turns)
            _interruptible_sleep(delay, hooks.should_abort)
            return None, backend_retries, opus_retries
        raise exc


def _interruptible_sleep(delay: float, should_abort: Callable[[], bool]) -> None:
    """Sleep in 1s slices so a shutdown lands INSIDE a backoff, not after it."""
    slept = 0.0
    while slept < delay and not should_abort():
        step = min(1.0, delay - slept)
        time.sleep(step)
        slept += step


def _coerce_finish(payload: Any) -> dict:
    """Normalise a stop-tool payload into the ``{status, notes}`` contract."""
    if isinstance(payload, dict) and "status" in payload:
        return payload
    return {"status": "apply_failed",
            "notes": f"unparseable finish(): {payload!r}"[:300]}


def _observer() -> Any:
    """The process observer, resolved UNGUARDED — as both old loops did.

    Deliberately not best-effort. ``get_observer()`` raises when nothing called
    ``init_observer``, and that is a wiring fault in the caller, not a telemetry
    hiccup: a loop that ran with no observer would leave no transcript and no
    ``l3_phase`` row, which is the write-only failure this project keeps
    re-learning. Individual CALLS below are best-effort (a broken sink must not
    break a running edit); acquiring the observer at all is not.
    """
    from observer import get_observer
    return get_observer()


def _best_effort(observer: Any, method: str, *args: Any, **kw: Any) -> Any:
    """Call one observer method. Telemetry must never break the research loop."""
    if observer is None:
        return None
    try:
        return getattr(observer, method)(*args, **kw)
    except Exception:  # noqa: BLE001
        logger.debug("observer.%s skipped", method, exc_info=True)
        return None


def tools(role: Role | str, tools: Sequence[dict], system: str, *,
          client: Any = None, model: str | None = None,
          transport: str | None = None,
          cwd: str | None = None, backend: str = "agent-sdk-editor",
          policy: LoopPolicy = LoopPolicy(),
          hooks: LoopHooks = LoopHooks()) -> Loop:
    """Build the loop for ``role`` over the transport its route implies.

    ``role`` may be a role NAME (resolved through :func:`role_for`) or an
    already-resolved :class:`Role`, which is what ``agents/l3.py`` hands in: it
    has the client and model in scope and must not re-resolve them.
    """
    resolved = role if isinstance(role, Role) else role_for(role)
    if model:
        resolved = replace(resolved, model=model)
    kind = transport or resolved.transport or "messages"
    resolved = replace(resolved, transport=kind)
    if kind == "agent_sdk":
        tp: Transport = AgentSdkTransport(
            resolved.model, list(tools), system,
            cwd=cwd, role=resolved.name, backend=backend,
            stop_tool=policy.stop_tool, max_turns=policy.max_turns)
    else:
        tp = MessagesTransport(
            client, resolved.model, list(tools), system,
            max_tokens=policy.max_tokens, temperature=policy.temperature,
            stop_tool=policy.stop_tool)
    return Loop(resolved, tools, system, transport=tp, policy=policy,
                hooks=hooks)


# ---------------------------------------------------------------------------
# Transport A — the bare `anthropic` messages API. One advance = one turn.
# ---------------------------------------------------------------------------


class MessagesTransport:
    """``client.messages.create`` with ``tools=``; the executor runs the calls.

    Owns the conversation: the assistant blocks and the tool results are the
    next two messages, which is why a backend-down retry can re-issue the SAME
    advance with the history intact.
    """

    kind = "messages"
    streams = True

    def __init__(self, client: Any, model: str, tool_defs: Sequence[dict],
                 system: str, prompt: str = "", *, max_tokens: int = 16000,
                 temperature: float | None = 0.3,
                 stop_tool: str = "finish") -> None:
        self.client = client
        self.model = model
        self.tool_defs = list(tool_defs)
        self.system = system
        self.max_tokens = max_tokens
        self.temperature = 0.3 if temperature is None else temperature
        self.stop_tool = stop_tool
        self.messages: list[dict] = []
        if prompt:
            self.messages.append({"role": "user", "content": prompt})

    def start(self, prompt: str) -> None:
        self.messages = [{"role": "user", "content": prompt}]

    def push(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def transcript(self) -> list[dict]:
        return self.messages

    def advance(self, executor: Any) -> Advance:
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature, system=self.system,
            tools=self.tool_defs, messages=self.messages)
        assistant: list[dict] = []
        results: list[dict] = []
        finish: dict | None = None
        calls = 0
        for block in resp.content:
            if block.type == "thinking":
                continue
            if block.type == "text":
                assistant.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant.append({"type": "tool_use", "id": block.id,
                                  "name": block.name, "input": block.input})
                calls += 1
                out = executor.execute(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": out})
                if block.name == self.stop_tool:
                    try:
                        finish = json.loads(out)
                    except json.JSONDecodeError:
                        finish = {"status": "apply_failed",
                                  "notes": "unparseable finish()"}
        if not assistant:
            assistant.append({"type": "text", "text": "(empty)"})
        self.messages.append({"role": "assistant", "content": assistant})
        emitted = [("assistant", assistant)]
        if results:
            self.messages.append({"role": "user", "content": results})
            emitted.append(("user", results))
        return Advance(turns=1, tool_calls=calls, finish=finish, emitted=emitted)


# ---------------------------------------------------------------------------
# Transport B — claude_agent_sdk over the logged-in OAuth subscription.
# One advance = one session; the SDK owns the turns inside it.
# ---------------------------------------------------------------------------


_SDK_SERVER_NAME = "l3editor"  # tools land as mcp__l3editor__<tool>


def sdk_tool_server(tool_defs: Sequence[dict], executor: Any, *,
                    name: str = _SDK_SERVER_NAME):
    """Expose ``executor`` as an in-process SDK MCP tool server.

    Returns ``(mcp_servers, allowed_tools, tool_objs)``. Every call routes back
    through ``executor.execute(name, args)``, so the worktree scoping, the
    locked-path and banned-command guards, ``validate_config`` and the
    ``finish()`` contract are the SAME code the bare transport runs — not a
    re-implementation that can drift. The schemas are the caller's own
    ``tool_defs``, so this is tool-set agnostic: a different executor and a
    different stop tool is all an L2 reasoning call would need.
    """
    import claude_agent_sdk as cas

    def _make(tool_def: dict):
        tname = tool_def["name"]

        async def _handler(args, _name=tname):
            result = executor.execute(_name, dict(args or {}))
            return {"content": [{"type": "text", "text": result}]}

        return cas.tool(tname, tool_def.get("description", ""),
                        tool_def["input_schema"])(_handler)

    tool_objs = [_make(td) for td in tool_defs]
    server = cas.create_sdk_mcp_server(name, "1.0.0", tool_objs)
    allowed = [f"mcp__{name}__{td['name']}" for td in tool_defs]
    return {name: server}, allowed, tool_objs


# ``sdk_session`` — the OAuth session driver — is imported at module scope from
# ``llm``. It reads NOTHING from this module and EVERYTHING from that one (the
# opus model id, the timeout wall, the backoff constants, the error classifier,
# the OAuth env scrub, the ledger writer), and ``llm.py`` states outright that
# those constants cannot move up here because a ``monkeypatch.setattr(llm, ...)``
# has to keep measuring the thing it names. A driver two modules away from every
# knob it reads is what forced ``llm`` to import back into this package from
# inside a function body. It now lives beside them; this module keeps the NAME,
# and :class:`AgentSdkTransport` below is still the transport that drives it.


class AgentSdkTransport:
    """One SDK session per advance. The SDK owns the turns inside it.

    ``streams`` is False because there is no per-turn boundary to stream: the
    session is opaque until it ends. Its transcript is reconstructed from the
    message stream and handed to the observer once, like the bare transport's.
    """

    kind = "agent_sdk"
    streams = False

    def __init__(self, model: str, tool_defs: Sequence[dict], system: str,
                 prompt: str = "", *, cwd: str | None = None,
                 role: str = "l3", backend: str = "agent-sdk-editor",
                 stop_tool: str = "finish", max_turns: int = 200) -> None:
        self.model = model
        self.tool_defs = list(tool_defs)
        self.system = system
        self.cwd = cwd
        self.role = role
        self.backend = backend
        self.stop_tool = stop_tool
        self.max_turns = max_turns
        self._base = prompt
        self._prompt = prompt
        self._servers: Any = None
        self._allowed: list[str] = []
        self._transcript: list[dict] = []

    def start(self, prompt: str) -> None:
        self._base = self._prompt = prompt

    def push(self, text: str) -> None:
        # A session cannot be resumed, so the nudge is appended to the prompt
        # and the next advance re-asks. That IS the SDK twin's nudge, unchanged.
        self._prompt = f"{self._base}\n\n{text}"

    def transcript(self) -> list[dict]:
        return self._transcript

    def advance(self, executor: Any) -> Advance:
        if self._servers is None:
            self._servers, self._allowed, _ = sdk_tool_server(
                self.tool_defs, executor)
        res = sdk_session(
            self._prompt, system=self.system, mcp_servers=self._servers,
            allowed_tools=self._allowed, max_turns=self.max_turns,
            stop_tool=self.stop_tool, cwd=self.cwd, role=self.role,
            backend=self.backend)
        self._transcript = res.get("transcript") or []
        if not res.get("ok"):
            raise BackendExhausted(
                f"opus SDK session exhausted (kind={res.get('error_kind')}): "
                f"{res.get('error')}", kind=res.get("error_kind"))
        stop = res.get("stop_input")
        return Advance(turns=int(res.get("turns") or 0),
                       tool_calls=int(res.get("tool_calls") or 0),
                       finish=None if stop is None else _coerce_finish(stop))


# ---------------------------------------------------------------------------
# 3. hold_for_backend — the outage hold, with a bound that is DECLARED
# ---------------------------------------------------------------------------


class Hold(BaseModel):
    """How long dispatch may pause for a down backend. ``0`` = forever.

    THE DEFAULT IS TODAY'S BEHAVIOUR AND THAT IS DELIBERATE. Production holds
    without limit because a down backend must never void a lineage — c8 lost
    6,766 backend-down events to voids because the guard gated on a health probe
    that answered 200 while POSTs hung. What changes is only WHERE the bound
    lives: it was ``OVFX_QWEN_WAIT_MAX``, an environment variable that
    ``tests/behavioural/conftest.py`` has to set to stop ``consolidate`` hanging
    past a 300s harness timeout; it is now a field a task spec can declare, whose
    default still reads that constant. Nothing changes silently.
    """

    max_s: float = 0.0
    probe_every_s: float = 60.0

    @classmethod
    def from_spec(cls, spec: Any = None) -> "Hold":
        """Read the declared bound off ``spec``; fall back to today's constant.

        ``kernel.model.TaskSpec`` does not carry a ``hold`` field yet — adding
        one is a change to another commit's module, so it is requested rather
        than made. Until it lands, any object exposing ``.hold`` is honoured and
        everything else falls back to ``config.QWEN_WAIT_MAX`` (0 = unbounded).
        """
        if isinstance(spec, cls):
            return spec
        declared = getattr(spec, "hold", None) if spec is not None else None
        if isinstance(declared, cls):
            return declared
        if isinstance(declared, dict):
            return cls.model_validate(declared)
        return cls(max_s=float(llm.QWEN_WAIT_MAX or 0.0))


def hold_for_backend(spec: Any = None, *,
                     should_abort: Callable[[], bool] | None = None) -> str:
    """HOLD until the backend serves again. Returns the reason it stopped.

    ``"serving"`` (the backend answered a real completion), ``"aborted"`` (a
    shutdown was requested — the only way out of an unbounded hold), or
    ``"expired"`` (the declared bound was reached).

    The gate is a REAL serving signal, not a health probe: ``GET /v1/models``
    answers a static 200 on this proxy while ``POST /v1/messages`` hangs, which
    is why the old guard never engaged for a whole cycle. Each inner wait is
    bounded by ``probe_every_s`` so the hold stays observable — one tick per
    interval — and a hung POST can never park us inside a single call.
    """
    hold = Hold.from_spec(spec)
    abort = should_abort or _never
    if abort():
        logger.warning("HOLD: skipped — shutdown already requested")
        return "aborted"
    if llm.is_qwen_healthy():
        return "serving"
    logger.warning(
        "HOLD: qwen backend not serving (completion probe failed) — pausing "
        "dispatch, re-probing every %.0fs, bound %s (never voiding lineages on "
        "a down backend)", hold.probe_every_s,
        f"{hold.max_s:.0f}s" if hold.max_s else "NONE (wait forever)")
    probe_every = max(0.001, float(hold.probe_every_s))
    t0 = time.monotonic()
    ticks = 0
    while True:
        if llm.wait_for_qwen(max_wait=probe_every,
                             poll_interval=min(float(llm.QWEN_POLL_INTERVAL),
                                               probe_every),
                             should_abort=abort):
            logger.warning("HOLD: released after %.0fs (%d re-probes) — "
                           "resuming dispatch", time.monotonic() - t0, ticks)
            return "serving"
        if abort():
            logger.warning("HOLD: RELEASED by shutdown request after %.0fs "
                           "(%d re-probes) — the backend is still down",
                           time.monotonic() - t0, ticks)
            return "aborted"
        ticks += 1
        elapsed = time.monotonic() - t0
        if hold.max_s and elapsed >= hold.max_s:
            logger.error("HOLD: declared bound of %.0fs reached after %d "
                         "re-probes — the backend is still down", hold.max_s,
                         ticks)
            return "expired"
        logger.warning("HOLD: qwen still down after %.0fs (%d re-probes) — "
                       "continuing to hold; no lineage is voided while held",
                       elapsed, ticks)


# ---------------------------------------------------------------------------
# 4. escalate — one failing call moves tier, and says so on the record
# ---------------------------------------------------------------------------


def escalate(reason: str, *, role: str = "l3",
             call_id: str | None = None) -> Role | None:
    """Ask for ONE qwen->opus escalation of a retry-exhausted call.

    Returns the opus :class:`Role` to re-issue the call on, or ``None`` when
    there is no real opus route or the per-cycle cap is spent — in which case
    the caller stays on qwen. It escalates the FAILING CALL, never the role: a
    permanent tier change is a routing decision and belongs in the spec.
    """
    if not llm.maybe_escalate_to_opus(role, reason=reason, call_id=call_id):
        return None
    escalated = role_for(role, tier="opus")
    return escalated if escalated.routable else None


# ---------------------------------------------------------------------------
# 5. usage_rollup — what the cycle spent, by model, role and backend
# ---------------------------------------------------------------------------
# The body lives in ``llm_usage.aggregate``, beside the ledger it reads, and is
# imported at module scope above. This module still NAMES it, because "what the
# cycle spent" is part of the LLM layer's contract; what it no longer does is
# own arithmetic over rows it does not store.
