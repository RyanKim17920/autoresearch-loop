"""overfitX SDK — Structured observability logging (spec §11).

One Observer per process, initialized exactly once in main (init_observer is
idempotent — repeat calls return the existing instance, they never stomp it).
The session_id is persisted to <batch>/sdk_session.json and reused on resume,
so one logical run keeps one session across crashes.

Every event is a single JSON line appended to <batch>/sdk_events.jsonl
(one write+flush per event, guarded by a lock — thread-safe, append-only).
Events are tagged with track/lineage/trial/role/parent_session where known.

Full agent transcripts go to <batch>/agent_details/ as ONE FILE PER CALL
(timestamped name) — no read-modify-write.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any
import uuid

from atomicio import atomic_write_text

logger = logging.getLogger(__name__)


def generate_session_id() -> str:
    return str(uuid.uuid4())[:8]


class Observer:
    """Append-only structured event logger. Thread-safe."""

    def __init__(self, batch_dir: Path, session_id: str | None = None):
        self.batch_dir = Path(batch_dir)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.session_id = session_id or self._load_or_create_session_id()
        self._detail_seq = 0

    # -- session persistence -------------------------------------------------

    @property
    def _session_file(self) -> Path:
        return self.batch_dir / "sdk_session.json"

    def _load_or_create_session_id(self) -> str:
        """Reuse the persisted session id (stable across resume) or mint one."""
        try:
            data = json.loads(self._session_file.read_text())
            sid = data.get("session_id")
            if sid:
                return sid
        except (OSError, json.JSONDecodeError):
            pass
        sid = generate_session_id()
        atomic_write_text(self._session_file, json.dumps(
            {"session_id": sid, "created_at": time.time()}, indent=2))
        return sid

    # -- event log -------------------------------------------------------------

    @property
    def log_path(self) -> Path:
        return self.batch_dir / "sdk_events.jsonl"

    def log(self, event_type: str, *, track: str | None = None,
            lineage_id: str | None = None, trial_n: int | None = None,
            role: str | None = None, parent_session: str | None = None,
            **fields: Any) -> None:
        """Append one event as a single JSON line (write + flush)."""
        event = {
            "ts": time.time(),
            "session": self.session_id,
            "event": event_type,
            "track": track,
            "lineage_id": lineage_id,
            "trial_n": trial_n,
            "role": role,
            "parent_session": parent_session,
            **fields,
        }
        line = json.dumps(event, default=str)
        with self._lock:
            try:
                with open(self.log_path, "a") as f:
                    f.write(line + "\n")
                    f.flush()
            except OSError as e:
                logger.warning("observer write failed: %s", e)

    # -- typed helpers -----------------------------------------------------------

    def log_agent_call(self, agent: str, track: str | None,
                       lineage_id: str | None,
                       prompt_preview: str, response_preview: str,
                       duration: float, model: str,
                       trial_n: int | None = None,
                       role: str | None = None,
                       full_prompt: str = "", full_response: str = "",
                       turns: list | None = None) -> None:
        """Log an LLM agent call; full transcript goes to its OWN detail file.

        `turns` (optional) is the raw multi-turn message list of an agent
        loop (L3 edit phases): [{role, content}, ...] where content is either
        a string or Anthropic-style blocks (text / tool_use / tool_result).
        Persisted verbatim under "turns" so the dashboard can replay the
        actual back-and-forth including tool calls and nudges.
        """
        detail_file = None
        if full_prompt or full_response or turns:
            payload = {
                "ts": time.time(),
                "agent": agent,
                "track": track,
                "lineage_id": lineage_id,
                "trial_n": trial_n,
                "model": model,
                "prompt": full_prompt,
                "response": full_response,
            }
            if turns:
                payload["turns"] = turns
            detail_file = self._write_detail(agent, lineage_id, payload)
        self.log(
            "agent_call",
            track=track, lineage_id=lineage_id, trial_n=trial_n,
            role=role or agent,
            agent=agent, model=model, duration_s=round(duration, 2),
            prompt_preview=prompt_preview[:200],
            response_preview=response_preview[:200],
            detail_file=str(detail_file) if detail_file else None,
        )

    def _write_detail(self, agent: str, lineage_id: str | None,
                      payload: dict) -> Path:
        """One file per agent call — never read-modify-write."""
        with self._lock:
            self._detail_seq += 1
            seq = self._detail_seq
        detail_dir = self.batch_dir / "agent_details"
        detail_dir.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}_{seq:04d}_{agent}_{lineage_id or 'none'}.json"
        path = detail_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def log_tool_call(self, tool: str, cmd: list[str] | str,
                      exit_code: int, duration: float,
                      parsed_status: str | None = None,
                      track: str | None = None,
                      lineage_id: str | None = None,
                      trial_n: int | None = None,
                      fake: bool = False) -> None:
        """Log a subprocess tool call (cmd, duration, rc, parsed status)."""
        self.log(
            "tool_call",
            track=track, lineage_id=lineage_id, trial_n=trial_n,
            tool=tool,
            cmd=cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd),
            exit_code=exit_code,
            duration_s=round(duration, 3),
            parsed_status=parsed_status,
            fake=fake,
        )

    def log_state_transition(self, state: str, old: str, new: str,
                             track: str | None = None,
                             lineage_id: str | None = None) -> None:
        self.log("state_transition", track=track, lineage_id=lineage_id,
                 state=state, old=old, new=new)

    # -- incremental turn streaming ------------------------------------------------

    @staticmethod
    def _sanitize(name: str) -> str:
        """Replace every char not in [A-Za-z0-9._-] with hyphen."""
        return re.sub(r'[^A-Za-z0-9._-]', '-', name)

    def stream_begin(self, *, agent: str, track: str, lineage_id: str | None,
                     trial_n: int | None = None, role: str | None = None,
                     model: str | None = None, prompt: str = "",
                     phase: str | None = None) -> dict:
        """Open an incremental turn-stream (.part) for a multi-turn agent call.

        Returns a handle dict.  If the initial write fails the handle's ``path``
        is ``None`` and subsequent ``stream_turn``/``stream_end`` no-op safely.
        """
        try:
            detail_dir = self.batch_dir / "agent_details"
            detail_dir.mkdir(parents=True, exist_ok=True)
            safe_agent = self._sanitize(agent)
            safe_lineage = self._sanitize(lineage_id or "none")
            part_path = detail_dir / f"inprogress__{safe_agent}__{safe_lineage}.jsonl.part"

            meta = {
                "_meta": {
                    "agent": agent,
                    "model": model,
                    "track": track,
                    "lineage_id": lineage_id,
                    "trial_n": trial_n,
                    "role": role,
                    "prompt": prompt[:20000],
                    "phase": phase,
                    "started": time.time(),
                }
            }
            with self._lock:
                with open(part_path, "a") as f:
                    f.write(json.dumps(meta, default=str) + "\n")
                    f.flush()
            return {
                "path": part_path,
                "n": 0,
                "agent": agent,
                "track": track,
                "lineage_id": lineage_id,
                "trial_n": trial_n,
                "role": role,
                "phase": phase,
            }
        except Exception as e:
            logger.warning("stream_begin failed for %s: %s", agent, e)
            return {
                "path": None,
                "n": 0,
                "agent": agent,
                "track": track,
                "lineage_id": lineage_id,
                "trial_n": trial_n,
                "role": role,
                "phase": phase,
            }

    def stream_turn(self, handle: dict, turn: dict, *, block_role: str) -> None:
        """Append one turn to the incremental .part file and emit an l3_turn event.

        Never raises.  Silently no-ops when *handle* is ``None`` or has no path.
        """
        if handle is None or handle.get("path") is None:
            return
        try:
            line = json.dumps(turn, default=str) + "\n"
            with self._lock:
                with open(handle["path"], "a") as f:
                    f.write(line)
                    f.flush()

            # Build preview
            role = turn.get("role", "")
            content = turn.get("content", "")
            preview = ""
            tool_name = None

            if role == "assistant":
                blocks = content if isinstance(content, list) else [content]
                text_blocks = [b for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
                tool_blocks = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
                if text_blocks:
                    preview = text_blocks[0].get("text", "")[:200]
                elif tool_blocks:
                    tb = tool_blocks[0]
                    tn = tb.get("name", "?")
                    tool_name = tn
                    inp = tb.get("input", {})
                    inp_str = json.dumps(inp, default=str)[:120]
                    preview = f"→ {tn}({inp_str})"
                else:
                    preview = str(content)[:200]
            else:
                # tool_result / user turn
                if isinstance(content, list):
                    combined = " ".join(
                        str(b.get("content", ""))
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    )
                else:
                    combined = str(content)
                preview = "result: " + combined[:180]

            if not preview and tool_name is None:
                preview = str(content)[:200]

            self.log(
                "l3_turn",
                track=handle.get("track"),
                lineage_id=handle.get("lineage_id"),
                trial_n=handle.get("trial_n"),
                role=role,
                agent=handle.get("agent"),
                phase=handle.get("phase"),
                turn_index=handle["n"],
                block_role=block_role,
                tool=tool_name,
                preview=preview[:200],
                part_file=str(handle["path"]),
            )
            handle["n"] += 1
        except Exception as e:
            logger.warning("stream_turn failed for %s: %s", handle.get("agent"), e)

    def stream_end(self, handle: dict, *, success: bool = True) -> None:
        """Clean up the .part file when the agent loop finishes normally.

        Always removes the .part — a crash/kill never reaches this method so
        the .part persists automatically for forensic inspection.
        """
        if handle is None or handle.get("path") is None:
            return
        try:
            Path(handle["path"]).unlink(missing_ok=True)
        except Exception as e:
            logger.warning("stream_end cleanup failed for %s: %s",
                           handle.get("agent"), e)


# ---------------------------------------------------------------------------
# Global instance — initialized exactly once (idempotent)
# ---------------------------------------------------------------------------

_observer: Observer | None = None
_init_lock = threading.Lock()


def init_observer(batch_dir: Path | str | None = None,
                  session_id: str | None = None) -> Observer:
    """Initialize the process-wide observer ONCE.

    Idempotent: if already initialized, returns the existing instance (the
    global is never stomped; the session id never churns mid-run).
    """
    global _observer
    with _init_lock:
        if _observer is not None:
            return _observer
        if batch_dir is None:
            from config import get_batch_dir
            batch_dir = get_batch_dir()
        _observer = Observer(Path(batch_dir), session_id=session_id)
        return _observer


def get_observer() -> Observer:
    if _observer is None:
        raise RuntimeError("Observer not initialized — call init_observer() "
                           "from main first.")
    return _observer


def reset_observer_for_tests() -> None:
    """Test hook: drop the global instance (tests only)."""
    global _observer
    with _init_lock:
        _observer = None
