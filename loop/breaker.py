"""The research circuit breaker (11-08): an LLM-spend stop for sessions that keep producing nothing.

11-08 nanobrain: 1207 researcher sessions in 10 h, every one "produced nothing (failed)" (an unsatisfiable slate rule
locked the edits; every smoke of the native root died on a forced plot period) and nothing stopped issuing briefs.

The breaker counts harvested sessions (loop.tick ``_intake``):

- ``researcher.breaker_consecutive`` (6) sessions IN A ROW that produced nothing (no node: failed / timeout /
  malformed ...; a backend outage and an infra re-issue are no session result and count nothing), or
- on a FRESH history (no node besides the root ever passed intake: every node rejected), ``researcher
  .breaker_fresh_sessions`` (30) sessions without one passing (a node queued = its smoke passed);

then it TRIPS: one ``ALERT_research_breaker`` event + stderr line naming the dominant failure reason, and no research
session starts (``Research.capacity`` -> 0) until a redeploy (the loop's git revision differs from the one that
tripped it) or an operator clear (drop ``<state_dir>/research_breaker.clear``; it is consumed). Any session that
yields a node resets the consecutive count; the first admitted node ends the fresh-history rule for good. 0 disables
a rule. While sessions succeed the counters never reach the limits, so the defaults leave a working loop as it was.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

K_BREAKER = "research.breaker"
CLEAR_FILE = "research_breaker.clear"
ROOT_ID = "root"
#: statuses of a harvested session that are no verdict on the session (an outage, a restart): never counted
NEUTRAL_STATUSES = frozenset({"backend_unavailable", "restart"})


def _reason(res: Any, outcome: str) -> str:
    """A short failure key: the session's status plus the first word of its error (``failed: empty_diff``), or the
    intake's reject reason (``rejected: smoke_failed``)."""
    if m := re.match(r"rejected \S+ from \S+: (\S+)", outcome or ""):
        return f"rejected: {m.group(1).rstrip(';,')}"
    status = str(getattr(res, "status", "") or "?")
    err = str(getattr(res, "error", "") or "").strip()
    head = re.split(r"[:\s(]", err, maxsplit=1)[0] if err else ""
    return f"{status}: {head}" if head else status


class Breaker:
    def __init__(self, ctx: Any, store: Any, *, code_rev: Callable[[], str] = lambda: "",
                 clock: Callable[[], float] | None = None) -> None:
        import time
        self.ctx, self.store, self.clock = ctx, store, clock or time.time
        self._rev_fn, self._rev = code_rev, None
        self.rev()          # review fix: the revision at PROCESS START -- a trip during a redeploy window (the new code
        #                     checked out, this process still the old one) must not pin the new revision as "tripped"
        rc = getattr(getattr(ctx, "cfg", None), "researcher", None)
        self.k = int(getattr(rc, "breaker_consecutive", 6) or 0)
        self.n_fresh = int(getattr(rc, "breaker_fresh_sessions", 30) or 0)

    # ------------------------------------------------------------------ state
    def rev(self) -> str:
        if self._rev is None:
            try:
                self._rev = str(self._rev_fn() or "")
            except Exception:  # noqa: BLE001 - no revision: no auto-release on redeploy
                self._rev = ""
        return self._rev

    def state(self) -> dict[str, Any]:
        st = self.store.get_state(K_BREAKER)
        if not isinstance(st, dict):
            st = {}
        if "fresh" not in st:
            st["fresh"] = self._fresh_history()
        st.setdefault("consecutive", 0)
        st.setdefault("fresh_sessions", 0)
        st.setdefault("reasons", {})
        return st

    def _fresh_history(self) -> bool:
        try:
            return not any(n.id != ROOT_ID and getattr(getattr(n, "status", None), "value", str(n.status)).upper()
                           != "REJECTED" for n in self.store.nodes())
        except Exception:  # noqa: BLE001 - no node table: treat as not fresh (the rule stays off)
            return False

    def _save(self, st: dict[str, Any]) -> None:
        self.store.set_state(K_BREAKER, st)

    def tripped(self) -> str:
        """Why new research sessions are held ('' = not)."""
        st = self.store.get_state(K_BREAKER)
        return str(st.get("tripped") or "") if isinstance(st, dict) else ""

    # ------------------------------------------------------------------ the tick's calls
    def check(self, acts: list[str]) -> str:
        """Release a tripped breaker on a redeploy (another code revision) or the operator's clear file; return why
        sessions are held ('' = not)."""
        st = self.store.get_state(K_BREAKER)
        if not isinstance(st, dict) or not st.get("tripped"):
            return ""
        why = ""
        clear = Path(getattr(self.ctx, "state_dir", "") or ".") / CLEAR_FILE
        if getattr(self.ctx, "state_dir", "") and clear.exists():
            try:
                clear.unlink()
            except OSError:
                pass
            why = "operator clear"
        elif (rev := self.rev()) and st.get("rev") and rev != st.get("rev"):
            why = f"redeploy ({st.get('rev')} -> {rev})"
        if not why:
            return str(st["tripped"])
        self._reset(st, why, acts)
        return ""

    def _reset(self, st: dict[str, Any], why: str, acts: list[str]) -> None:
        old = st.get("tripped")
        st.update(tripped="", rev="", consecutive=0, fresh_sessions=0, reasons={})
        self._save(st)
        self.store.log("research_breaker_cleared", why=why, was=str(old)[:300])
        acts.append(f"research breaker cleared ({why}): research sessions may start")

    def record(self, res: Any, outcome: str, acts: list[str]) -> str:
        """Count one harvested session (``outcome``: the intake's line). Returns the trip reason when THIS session
        tripped the breaker, else ''."""
        if str(getattr(res, "status", "")) in NEUTRAL_STATUSES:
            return ""
        out = str(outcome or "")
        if out.startswith("research ") and ("re-issued" in out or "reissued" in out or "asked again" in out):
            return ""                                    # an infra / starved re-issue: no verdict
        st = self.state()
        produced = out.startswith(("queued ", "rejected "))
        admitted = out.startswith("queued ")
        reasons = Counter(st.get("reasons") or {})
        if produced and st.get("tripped"):              # review fix: a node after a trip releases the breaker
            old = st.get("tripped")
            st.update(tripped="", rev="")
            self.store.log("research_breaker_cleared", why="a session produced a node", was=str(old)[:300])
            acts.append("research breaker cleared (a session produced a node): research sessions may start")
        if admitted:                                     # a node passed intake (its smoke passed): all clear
            st.update(fresh=False, fresh_sessions=0, consecutive=0)
            reasons = Counter()
        else:
            if produced:                                 # a node, rejected: no "produced nothing" streak
                st["consecutive"] = 0
                if not st["fresh"]:
                    reasons = Counter()
            else:
                st["consecutive"] = int(st["consecutive"]) + 1
            if not produced or st["fresh"]:
                reasons[_reason(res, out)] += 1
            if st["fresh"]:
                st["fresh_sessions"] = int(st["fresh_sessions"]) + 1
        st["reasons"] = dict(reasons.most_common(12))
        why = ""
        if not st.get("tripped"):
            if self.k > 0 and st["consecutive"] >= self.k:
                why = f"{st['consecutive']} consecutive research sessions produced nothing"
            elif self.n_fresh > 0 and st["fresh"] and st["fresh_sessions"] >= self.n_fresh:
                why = f"{st['fresh_sessions']} research sessions on a fresh history and no smoke passed"
        if why:
            top = Counter(st["reasons"]).most_common(3)
            dom = ", ".join(f"{r} x{n}" for r, n in top) or "?"
            st.update(tripped=f"{why}; dominant failure: {dom}", rev=self.rev(), ts=self.clock())
            self.store.log("ALERT_research_breaker", reason=why, dominant=top[0][0] if top else "",
                           reasons=dict(top), rev=st["rev"])
            line = (f"ALERT research breaker tripped: {why}; dominant failure: {dom}. No new research session until "
                    f"a redeploy or `touch {Path(getattr(self.ctx, 'state_dir', '') or '.') / CLEAR_FILE}`")
            acts.append(line[:600])
            print(line[:800], file=sys.stderr, flush=True)
        self._save(st)
        return why
