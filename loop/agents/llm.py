"""loop.agents.llm — the only door to a model (DESIGN §9). :class:`FakeLLMClient` (scripted, every
test) and :class:`SdkLLMClient` (Claude via the root ``llm`` module, imported lazily; cheap roles on
qwen, never falling back to Opus). A dead backend raises :class:`BackendUnavailable`: the driver waits
instead of voiding work, and a CIRCUIT BREAKER (:func:`note_failure`) turns a known SDK outage error or the
same failure repeated into an outage the driver pauses every new LLM job on."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)
FINISH_TOOL = "finish"


@dataclass(frozen=True)
class AgentRun:
    ok: bool
    finish: Mapping[str, Any] | None          # the validated finish payload
    turns: int
    wall_s: float
    transcript_path: str = ""
    error: str = ""
    web_searches: int = 0                     # built-in WebSearch / WebFetch calls (web sessions only)
    web_fetches: int = 0


WEB_TOOLS = ("WebSearch", "WebFetch")


def web_counts(names: Sequence[str]) -> tuple[int, int]:
    """(WebSearch calls, WebFetch calls) among tool names."""
    return sum(n == "WebSearch" for n in names), sum(n == "WebFetch" for n in names)


class SchemaError(ValueError):
    """A structured answer never satisfied its schema."""


class BackendUnavailable(RuntimeError):
    """The model backend is down or unreachable: wait and retry, never void."""


#: account refusals (``llm.sdk_session`` error_kind / ``llm.LLMOutageError.kind``): the driver pauses ALL new
#: LLM work until the reset (:meth:`loop.driver.Driver._llm_gate`), it never retries into them
OUTAGE_KINDS = ("quota", "auth")
#: sdk_session endings that are the session's OWN budget (turns / wall), never a backend outage (v3-c7i I7)
SESSION_LIMITS = ("max_turns", "timeout")
#: CIRCUIT BREAKER: an SDK failure carrying one of these is an outage at once (live 2026-10-03: 'Claude Code returned
#: an error result: success' failed 809 research sessions, 2292 postmortems and 122 reviews in one storm, none paused);
#: any other identical failure signature seen BREAKER_N times within BREAKER_WINDOW_S, with no success between, is too
SDK_OUTAGE_PATTERNS = ("error result: success",)
BREAKER_N, BREAKER_WINDOW_S = 3, 900.0
SDK_KIND = "sdk"
_OUTAGE_LOCK = threading.Lock()
_OUTAGE: dict[str, Any] = {}
_FAILS: dict[str, Any] = {"sig": "", "n": 0, "first": 0.0, "ok_ts": 0.0}


def note_outage(kind: str, resets_at: float | None) -> None:
    """A Claude call was refused by the account (any pool thread); read once by :func:`take_outage`."""
    with _OUTAGE_LOCK:
        _OUTAGE.update(kind=kind, resets_at=resets_at or _OUTAGE.get("resets_at"))


def take_outage() -> dict[str, Any] | None:
    """The refusal noted since the last call ({kind, resets_at}), or None."""
    with _OUTAGE_LOCK:
        out = dict(_OUTAGE) or None
        _OUTAGE.clear()
    return out


def _signature(err: str) -> str:
    return re.sub(r"\d+", "N", " ".join(str(err).split()))[:200]


def note_failure(err: str, now: float | None = None) -> bool:
    """A failed Claude call (any pool thread): True (and an outage of kind 'sdk' noted) when it carries a
    :data:`SDK_OUTAGE_PATTERNS` text, or its signature is the BREAKER_N-th identical one within BREAKER_WINDOW_S."""
    now = time.time() if now is None else now
    sig = _signature(err)
    with _OUTAGE_LOCK:
        if sig != _FAILS["sig"] or now - _FAILS["first"] > BREAKER_WINDOW_S:
            _FAILS.update(sig=sig, n=0, first=now)
        _FAILS["n"] += 1
        hit = any(p in str(err) for p in SDK_OUTAGE_PATTERNS) or _FAILS["n"] >= BREAKER_N
    if hit:
        note_outage(SDK_KIND, None)
    return hit


def note_success(now: float | None = None) -> None:
    """A Claude call that worked: the breaker's count restarts (and :func:`last_success` moves)."""
    with _OUTAGE_LOCK:
        _FAILS.update(sig="", n=0, first=0.0, ok_ts=time.time() if now is None else now)


def last_success() -> float:
    """When a Claude call last worked (0.0: not since this process started)."""
    with _OUTAGE_LOCK:
        return float(_FAILS["ok_ts"])


_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, Mapping), "array": lambda v: isinstance(v, (list, tuple)),
    "string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "null": lambda v: v is None}


def validate(value: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Violations (empty = ok) of: type, enum, required, properties, additionalProperties,
    items, minItems, minLength, minimum, exclusiveMinimum."""
    t = schema.get("type")
    types = [t] if isinstance(t, str) else list(t or [])
    if types and not any(_TYPES[x](value) for x in types):
        return [f"{path}: expected {'/'.join(types)}, got {type(value).__name__}"]
    errs = [f"{path}: {value!r} not in {list(schema['enum'])}"] \
        if "enum" in schema and value not in schema["enum"] else []
    if isinstance(value, Mapping):
        props, extra = schema.get("properties", {}), schema.get("additionalProperties", True)
        errs += [f"{path}.{k}: required" for k in schema.get("required", ()) if k not in value]
        for k, sub in value.items():
            if k in props or isinstance(extra, Mapping):
                errs += validate(sub, props.get(k, extra), f"{path}.{k}")
            elif extra is False:
                errs.append(f"{path}.{k}: not allowed")
    if isinstance(value, (list, tuple)):
        if len(value) < schema.get("minItems", 0):
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        for i, sub in enumerate(value if "items" in schema else ()):
            errs += validate(sub, schema["items"], f"{path}[{i}]")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        errs.append(f"{path}: shorter than {schema['minLength']}")
    if _TYPES["number"](value):
        if "minimum" in schema and value < schema["minimum"] or \
                "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errs.append(f"{path}: {value} is below the minimum")
    return errs


class ToolGate:
    """Wraps the caller's ``executor(name, args) -> str``: a valid ``finish`` is recorded
    and ends the session, an invalid one returns its violations so the model can retry;
    after ``deadline`` every tool call is refused. The executor's ``off_clock`` tools (v3-c4i: the smoke, the
    diagnose job) do not count against the deadline: their wall time moves it, at most :data:`OFF_CLOCK_MAX_S` in
    all (10-13 transcripts: a median 13.5 min smoke after ~12.5 min of work timed out 34% of 20-min sessions)."""

    def __init__(self, executor: Callable[[str, Mapping[str, Any]], str],
                 finish_schema: Mapping[str, Any], deadline: float) -> None:
        self._call, self.finish_schema, self.deadline = executor, finish_schema, deadline
        self.finish: dict[str, Any] | None = None
        self.log: list[tuple[str, dict, str]] = []
        self.off_clock_s = 0.0

    def __call__(self, name: str, args: Mapping[str, Any] | None) -> str:
        args, name = dict(args or {}), name.rsplit("__", 1)[-1]          # mcp__server__tool -> tool
        if self.finish is not None:
            out = "session already finished; stop now"
        elif name == FINISH_TOOL:
            errs = validate(args, self.finish_schema)
            hook = getattr(self._call, "check_finish", None)     # e.g. the intake rules on the diff
            bounce = "" if errs or hook is None else hook(args)
            out = ("finish REJECTED, fix and call finish again:\n  " + "\n  ".join(errs[:12]) if errs
                   else f"finish BOUNCED: {bounce}" if bounce else "finish accepted; end the session now.")
            self.finish = None if errs or bounce else args
        elif time.monotonic() > self.deadline:
            out = "TIME LIMIT REACHED: no more tools; call finish now."
        else:
            t = time.monotonic()
            try:
                out = str(self._call(name, args))
            except Exception as e:  # noqa: BLE001 — a tool error is a tool result: one line, no trace
                msg = (str(e).strip().splitlines() or [""])[0][:300]
                out = f"ERROR ({type(e).__name__}): {msg}. Fix the arguments and retry."
            if name in tuple(getattr(self._call, "off_clock", ()) or ()):
                dt = min(time.monotonic() - t, max(0.0, OFF_CLOCK_MAX_S - self.off_clock_s))
                self.off_clock_s += dt
                self.deadline += dt
        self.log.append((name, args, out))
        return out

    def execute(self, name: str, args: Mapping[str, Any] | None = None) -> str:
        """The ``agent.llm.sdk_tool_server`` executor protocol (it calls ``executor.execute``)."""
        return self(name, args)

    def result(self, *, turns: int, t0: float, transcript_path: str = "", error: str = "") -> AgentRun:
        if self.finish is None and not error:
            error = ("timeout: wall limit reached without finish" if time.monotonic() > self.deadline
                     else "no_finish: session ended without a valid finish")
        return AgentRun(self.finish is not None, self.finish, turns, time.monotonic() - t0,
                        transcript_path, "" if self.finish is not None else error)


class FakeLLMClient:
    """Scripted client for tests. ``scripts[role]`` is a list consumed front to back or a
    callable receiving the call's keyword arguments. ``complete_json`` responses are dicts
    (schema-validated) or exceptions (raised). ``run_agent`` responses are
    ``{"tool_calls": [(name, args), ...], "finish": {...}}`` (run through the real
    executor behind a :class:`ToolGate`), an :class:`AgentRun`, or an exception.
    Every call is recorded in ``self.calls``."""

    def __init__(self, scripts: Mapping[str, Any]) -> None:
        self.scripts = {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in scripts.items()}
        self.calls: list[dict[str, Any]] = []

    def _next(self, role: str, call: dict[str, Any]) -> Any:
        src = self.scripts[role]
        resp = src(**call) if callable(src) else src.pop(0)
        if isinstance(resp, BaseException):
            raise resp
        return resp

    def complete_json(self, role: str, *, system: str, prompt: str,
                      schema: Mapping[str, Any]) -> Mapping[str, Any]:
        call = {"role": role, "kind": "json", "system": system, "prompt": prompt, "schema": schema}
        self.calls.append(call)
        resp = self._next(role, dict(call))
        if errs := validate(resp, schema):
            raise SchemaError(f"role={role!r}: {errs}")
        return resp

    def run_agent(self, role: str, *, system: str, prompt: str, cwd: str,
                  tools: Sequence[Mapping[str, Any]], executor: Any,
                  finish_schema: Mapping[str, Any], max_turns: int, timeout_s: float,
                  web: bool = False) -> AgentRun:
        t0 = time.monotonic()
        gate = ToolGate(executor, finish_schema, t0 + timeout_s)
        call = {"role": role, "kind": "agent", "system": system, "prompt": prompt, "cwd": cwd,
                "tools": [t["name"] for t in tools], "max_turns": max_turns, "timeout_s": timeout_s,
                "web": web}
        self.calls.append({**call, "tool_log": gate.log})
        resp = self._next(role, call)
        if isinstance(resp, AgentRun):
            return resp
        steps = list(resp.get("tool_calls", ()))
        web_names: list[str] = []
        for name, args in steps[:max_turns]:
            if name in WEB_TOOLS:            # the CLI's built-ins: never the executor; refused without web
                if web:
                    web_names.append(name)
                    getattr(executor, "on_web", lambda *_: None)(name, args)
                continue
            gate(name, args)
        ws, wf = web_counts(web_names)
        if len(steps) > max_turns:
            return dataclasses.replace(gate.result(turns=max_turns, t0=t0, error=f"max_turns: {max_turns} turns "
                                                   "without finish"), web_searches=ws, web_fetches=wf)
        turns = len(steps)
        if resp.get("finish") is not None:
            # a script cannot react to a bounced finish, so it insists until the bounce cap lets it through
            for _ in range(4):
                turns += 1
                if not gate(FINISH_TOOL, resp["finish"]).startswith("finish BOUNCED"):
                    break
        return dataclasses.replace(gate.result(turns=turns, t0=t0), web_searches=ws, web_fetches=wf)


#: role -> default tier when the spec's ``llm_roles`` is silent
DEFAULT_TIERS = {"researcher": "opus", "review": "opus", "scout": "opus"}
#: after qwen raises BackendUnavailable, qwen-routed calls are skipped (BackendUnavailable, never Opus) this long
QWEN_COOLDOWN_S = 600.0
#: per-role qwen request timeout (s; default 600): v3-c6r -- the pre-screen is one short ranking call, and a hung one
#: must not hold its worker for 10 min. A timeout of a role listed here does NOT start the qwen cooldown (the backend
#: is slow, not down: the other roles keep going)
ROLE_TIMEOUT_S = {"prescreen": 45.0, "mech_judge": 30.0, "bold_knob_judge": 30.0, "slate_judge": 45.0}
#: one tool call may outlive the gate's deadline by this much (a queued SLURM smoke blocks up to 1 h)
AGENT_TOOL_GRACE_S = 3600.0
#: the most an executor's off-clock tools (ToolGate) may move a session's deadline in all
OFF_CLOCK_MAX_S = 2400.0


def opus_chat(system: str, prompt: str, temperature: float, role: str) -> str:
    import llm as _llm   # D0.4: the sanctioned lazy import (pulls config)
    return _llm.query_tier("opus", prompt, system=system, temperature=temperature,
                           json_mode=True, role=role)


def qwen_url() -> str:
    """OpenAI-compatible base URL of the cheap-tier model, from ``OVFX_LOOP_QWEN_URL``."""
    url = os.environ.get("OVFX_LOOP_QWEN_URL", "").strip()
    if not url:
        raise BackendUnavailable("no qwen endpoint (set OVFX_LOOP_QWEN_URL)")
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def qwen_chat(system: str, prompt: str, temperature: float, timeout: float = 600.0) -> str:
    body = {"model": os.environ.get("OVFX_QWEN_MODEL", "Qwen/Qwen3.6-27B"),
            "temperature": temperature, "max_tokens": 8000, "response_format": {"type": "json_object"},
            # Qwen3.6 thinks by default and the reasoning burns the token cap; switch it off
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request(qwen_url() + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        if not text:      # thinking ate the cap / empty reply: retryable -> _ask falls back to opus
            raise BackendUnavailable("qwen returned empty content")
        return text
    except urllib.error.HTTPError as e:
        if e.code < 500 and e.code != 429:
            raise
        raise BackendUnavailable(f"qwen HTTP {e.code}") from e
    except (OSError, ValueError, KeyError) as e:      # URLError, timeouts, resets, bad JSON
        raise BackendUnavailable(f"qwen: {type(e).__name__}: {e}") from e


def parse_json(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


class SdkLLMClient:
    """Production client routed by the ACTIVE spec's ``llm_roles``: tier/backend in
    {opus, strong} -> Claude, else qwen. Tool-using agents always run on Claude
    (``llm.sdk_session``) with an in-process tool server behind :class:`ToolGate`."""

    def __init__(self, spec: Any, *, attempts: int = 3, transcript_dir: str = "",
                 qwen_cooldown_s: float = QWEN_COOLDOWN_S) -> None:
        self.spec, self.attempts, self.transcript_dir = spec, max(1, attempts), transcript_dir
        self.qwen_cooldown_s = qwen_cooldown_s
        self._qwen_down_until = 0.0          # monotonic; qwen-routed calls are skipped until then

    def default_backend(self) -> str:
        """The spec's ``llm_default_backend`` ('qwen' unless a spec says 'opus'): where undeclared roles run."""
        return str(getattr(self.spec, "llm_default_backend", "qwen") or "qwen").lower()

    def tier(self, role: str) -> str:
        decl = (getattr(self.spec, "llm_roles", None) or {}).get(role)
        if decl is None:
            return DEFAULT_TIERS.get(role, "opus" if self.default_backend() == "opus" else "qwen")
        return "opus" if {str(decl.tier).lower(), str(decl.backend).lower()} & {"opus", "strong"} else "qwen"

    def _ask(self, role: str, system: str, prompt: str, temperature: float) -> str:
        """A qwen-routed role runs on qwen only: while qwen is down (and ``qwen_cooldown_s`` after a failure) it
        raises BackendUnavailable -- the job is skipped and asked again later, never sent to Opus. A Claude
        failure feeds the circuit breaker (:func:`note_failure`); a success resets it."""
        if self.tier(role) == "qwen":
            if time.monotonic() < self._qwen_down_until:
                raise BackendUnavailable(f"qwen in cooldown (role {role}: skipped, no Opus fallback)")
            short = ROLE_TIMEOUT_S.get(role)
            try:
                out = qwen_chat(system, prompt, temperature, timeout=short) if short else \
                    qwen_chat(system, prompt, temperature)
            except Exception as e:  # noqa: BLE001 — any qwen failure: skip the job
                if type(e).__name__ == "PromptTooLongError":
                    raise
                if short and "timed out" in str(e).lower():     # a short-timeout role ran out of time: no cooldown
                    raise BackendUnavailable(f"qwen: role {role} timed out after {short:.0f}s") from e
                self._qwen_down_until = time.monotonic() + self.qwen_cooldown_s
                logger.warning("llm_skip role=%s qwen down for %.0fs (no Opus fallback): %s", role,
                               self.qwen_cooldown_s, str(e)[:200])
                raise BackendUnavailable(str(e)[:300] if isinstance(e, BackendUnavailable) else f"qwen: {e}"[:300]) from e
            self._qwen_down_until = 0.0
            return out
        try:
            out = opus_chat(system, prompt, temperature, role)
        except BackendUnavailable as e:
            note_failure(str(e))
            raise BackendUnavailable(f"opus: {e}"[:300]) from e
        except Exception as e:  # noqa: BLE001 — opaque SDK/oauth/network failure
            if type(e).__name__ == "PromptTooLongError":
                raise
            if getattr(e, "kind", None) in OUTAGE_KINDS:             # llm.LLMOutageError
                note_outage(e.kind, getattr(e, "resets_at", None))
            else:
                note_failure(f"{type(e).__name__}: {e}")
            raise BackendUnavailable(f"opus: {type(e).__name__}: {e}"[:300]) from e
        note_success()
        return out

    def complete_json(self, role: str, *, system: str, prompt: str,
                      schema: Mapping[str, Any]) -> Mapping[str, Any]:
        rendered = json.dumps(schema, sort_keys=True)
        ask, problem = f"{prompt}\n\nReturn ONE JSON object matching this JSON schema:\n{rendered}", ""
        for attempt in range(1, self.attempts + 1):
            text = self._ask(role, system, ask, 0.7 if attempt == 1 else 0.2)
            try:
                obj = parse_json(text)
            except ValueError as e:
                problem = f"reply was not parseable JSON ({e}); it began {text[:300]!r}"
            else:
                if not (errs := validate(obj, schema)):
                    return obj
                problem = "reply violated the schema: " + "; ".join(errs[:10])
            logger.warning("complete_json[%s]: re-ask %d/%d: %s", role, attempt, self.attempts,
                           problem[:300])
            ask = (f"{prompt}\n\nIMPORTANT: your previous {problem}\nReturn ONLY one JSON object "
                   f"matching this JSON schema exactly:\n{rendered}")
        raise SchemaError(f"role={role!r} never produced a valid answer: {problem}")

    def run_agent(self, role: str, *, system: str, prompt: str, cwd: str,
                  tools: Sequence[Mapping[str, Any]], executor: Any,
                  finish_schema: Mapping[str, Any], max_turns: int, timeout_s: float,
                  web: bool = False) -> AgentRun:
        t0 = time.monotonic()
        gate = ToolGate(executor, finish_schema, t0 + timeout_s)
        defs = [dict(t) for t in tools] + [{
            "name": FINISH_TOOL, "input_schema": dict(finish_schema),
            "description": "End the session with your final structured answer. Call exactly "
                           "once, after your last edit and smoke test."}]
        try:
            import llm as _llm
            from agent.llm import sdk_tool_server   # lazy: agent.* pulls config
            servers, allowed, _ = sdk_tool_server(defs, gate, name="loopagent")
            # the CLI's built-in web tools are reachable only when allow-listed (read-only, no worktree access)
            # (verified live 2026-09-30: with web the model's WebSearch returns results; without, the CLI
            # answers "Claude requested permissions to use WebSearch" -- the default permission mode is
            # what keeps every non-allow-listed built-in, incl. Bash/Edit/Write, unreachable)
            allowed = [*allowed, *(WEB_TOOLS if web else ())]
            # the CLI defers MCP tools behind ToolSearch by default (2 wasted turns per session);
            # "false" loads every tool up front. Process-wide on purpose: concurrent sessions agree.
            os.environ["ENABLE_TOOL_SEARCH"] = "false"
            # sdk_session's wall is the global OVFX_OPUS_TIMEOUT (600 s by default) and a wall hit
            # RE-RUNS the whole session on the same worktree; the agent's own limit is the gate
            # deadline, so the wall must cover it plus one in-flight tool call (the smoke).
            wall = timeout_s + AGENT_TOOL_GRACE_S + (OFF_CLOCK_MAX_S if getattr(executor, "off_clock", ()) else 0.0)
            if 0 < float(getattr(_llm, "OPUS_TIMEOUT", 0) or 0) < wall:
                _llm.OPUS_TIMEOUT = wall
            seen = getattr(executor, "on_web", None)       # the live web log (loop.agents.notes research stage)
            res = _llm.sdk_session(prompt, system=system, mcp_servers=servers, allowed_tools=allowed,
                                   max_turns=max_turns + 2, stop_tool=None, cwd=cwd, role=role,
                                   backend="loop-agent-sdk", done=lambda: gate.finish is not None,
                                   on_tool_use=(lambda n, i: seen(n, i) if n in WEB_TOOLS else None) if seen else None)
            if not res.get("ok") and res.get("error_kind") in OUTAGE_KINDS:
                note_outage(res["error_kind"], res.get("resets_at"))
            elif not res.get("ok") and gate.finish is None and res.get("error_kind") not in ("max_turns", "timeout"):
                note_failure(f"sdk_session {res.get('error_kind')}: {res.get('error')}")
            elif res.get("ok") or gate.finish is not None:
                note_success()
            if not res.get("ok") and gate.finish is None and res.get("error_kind") in SESSION_LIMITS:
                # v3-c7i I7: the session ran out of turns / time -- the model's budget, not a backend outage (827
                # max_turns endings were labelled backend_unavailable and re-issued as "outages")
                limit = f"{res.get('error_kind')}: sdk_session {res.get('error_kind')}: {res.get('error')}"[:400]
            elif not res.get("ok") and gate.finish is None:
                raise RuntimeError(f"sdk_session {res.get('error_kind')}: {res.get('error')}")
            else:
                limit = ""
        except Exception as e:  # noqa: BLE001 — a backend outage is not a failed edit
            return gate.result(turns=0, t0=t0, error=f"backend_unavailable: {type(e).__name__}: {e}"[:400])
        path = ""
        if self.transcript_dir:
            path = str(Path(self.transcript_dir) / f"{role}-{int(time.time() * 1000)}.json")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(res.get("transcript") or [], default=str, indent=1))
        ws, wf = web_counts([b.get("name", "") for m in res.get("transcript") or [] for b in m.get("content") or []
                             if isinstance(b, Mapping) and b.get("type") == "tool_use"]) if web else (0, 0)
        return dataclasses.replace(gate.result(turns=int(res.get("turns") or 0), t0=t0, transcript_path=path,
                                               error=limit),
                                   web_searches=ws, web_fetches=wf)
