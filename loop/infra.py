"""loop.infra -- a smoke that died of the ENVIRONMENT is no verdict on the idea.

Live 2026-10-21: the torch hub cache (``~/.cache/torch`` -> ``/data/.../torch``) was deleted; every smoke died with
``FileNotFoundError: .../.cache/torch/hub`` and the loop REJECTED four ideas ``smoke_failed`` (and a program session
produced nothing). This module holds the pieces that keep that from recurring:

* :func:`classify` -- a smoke log's INFRA signature (a cache / home path that is gone or unwritable, a missing
  TORCH_HOME / HF cache, a download error on the init weights, a corrupt weights file, ENOSPC) or ``''``;
  :func:`signature` -- the normalized traceback tail two sessions' failures are compared on.
* :class:`SmokeGate` -- one small JSON file in the state dir. An infra smoke failure TRIPS it: new smokes are held
  (returned as ``unavailable``, the work is kept) for a short, doubling backoff; then ONE canary smoke goes through,
  and the first smoke that gets past the environment clears it. ``>= CORRELATED`` sessions failing smoke with the same
  normalized tail inside an hour count as infra too (the 3rd is flagged; the gate trips).
* :func:`preflight` -- cheap checks before a smoke is submitted: the torch hub dir resolves (no dangling link), and
  the configured pretrained weights file is in ``<hub>/checkpoints`` (or its host is reachable).
* :func:`reissue_brief` -- the brief of a re-issued request (previous attempt's hypothesis and diff appended).
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

CORRELATED = 3                  # sessions failing smoke with one normalized tail within CORRELATED_WINDOW_S: infra
CORRELATED_WINDOW_S = 3600.0
BACKOFF_S = 300.0               # first hold after an infra smoke failure; doubles per consecutive trip
BACKOFF_MAX_S = 3600.0
CANARY_POLL_S = 120.0           # a held smoke re-asks this often while another session's canary is in flight
CANARY_STALE_S = 1800.0         # a canary that never reported (its thread died) is replaced after this
ALERT_EVERY_S = 3600.0          # the same signature alerts at most this often
MAX_INFRA_REISSUES = 3          # per request id
REISSUE_MARK = "\n\n=== RE-ISSUED: the previous attempt's smoke died of the ENVIRONMENT, not of your change ==="
MAX_STARVE_REISSUES = 6          # per request id: re-issues after a smoke queue timeout (10-22; the work is kept)
K_STARVE_REISSUES = "infra.starve_reissues"
K_INFRA_REISSUES = "infra.reissues"         # {request id: re-issues so far}
K_INFRA_ALERTED = "infra.alerted_trip"      # the gate's trip id the tick last surfaced
REISSUE_FILE = "reissue_smoke.json"         # state-dir drop file: {"nodes": ["a node", ...]}
REISSUED_REASON = "smoke_infra_reissued"    # reject_reason of a node whose request was re-issued from its stored diff

_CACHE_PATH = (r"(?:/\.cache\b|/\.config\b|/torch/hub\b|torch/hub/checkpoints|/huggingface\b|/hub/(?:models|datasets)|"
               r"\bTORCH_HOME\b|\bHF_HOME\b|\bHF_HUB_CACHE\b|\bTRANSFORMERS_CACHE\b|\bXDG_CACHE_HOME\b|"
               r"/\.local/share\b|/home/[^\s'\"]*/\.[a-z])")
#: (kind, regex) over the smoke's log: the first match names the infra kind. Order: most specific first.
RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple((k, re.compile(rx, re.I | re.M)) for k, rx in (
    ("disk full", r"No space left on device|\bENOSPC\b|Disk quota exceeded|\[Errno 28\]"),
    ("cache path", r"(?:FileNotFoundError|PermissionError|NotADirectoryError|IsADirectoryError|OSError)\b[^\n]*"
                   + _CACHE_PATH),
    ("cache env", r"\b(?:TORCH_HOME|HF_HOME|HF_HUB_CACHE|TRANSFORMERS_CACHE)\b[^\n]*(?:not set|not found|does not exist|"
                  r"no such|permission denied|not writable|read-only)"),
    ("corrupt weights", r"PytorchStreamReader failed reading zip archive|failed finding central directory|"
                        r"invalid load key|unexpected EOF|file is too small|Unexpected end of (?:file|archive)"),
    ("weights download", r"(?:urllib\.error\.(?:URLError|HTTPError)|HTTP Error \d{3}|Temporary failure in name "
                         r"resolution|Name or service not known|Connection (?:refused|reset|timed out)|"
                         r"ConnectionError|ReadTimeout|SSLError|Max retries exceeded|Network is unreachable|"
                         r"Could not resolve host|ssl\.SSLError)"),
))
#: a network error counts as a WEIGHTS download failure only when the log also shows a hub / download frame
_DOWNLOAD_CTX = re.compile(r"load_state_dict_from_url|download_url_to_file|torch[/.]hub|hf_hub|from_pretrained|"
                           r"urlretrieve|snapshot_download|fbaipublicfiles|huggingface\.co", re.I)
_EXC_LINE = re.compile(r"^\s*(?:[\w.]+\.)?(?:\w*Error|\w*Exception|\w*Exit|KeyboardInterrupt)\b.*$", re.M)


def classify(text: str) -> str:
    """The infra kind a smoke log shows ('' = none): ``'<kind>: <the matching line>'``."""
    text = text or ""
    for kind, rx in RULES:
        m = rx.search(text)
        if not m:
            continue
        if kind == "weights download" and not _DOWNLOAD_CTX.search(text):
            continue
        a = text.rfind("\n", 0, m.start()) + 1
        b = text.find("\n", m.end())
        line = text[a:b if b >= 0 else len(text)].strip()
        return f"{kind}: {line[:240]}"
    return ""


def signature(text: str) -> str:
    """The normalized traceback tail of a smoke log (its last exception line; '' when the log shows none): numbers,
    hex ids, request / worktree names and temp dirs collapsed, so the same failure of different sessions compares equal."""
    lines = _EXC_LINE.findall(text or "")
    if not lines:
        return ""
    s = lines[-1].strip()
    s = re.sub(r"loop-req-[\w-]+|lp[0-9a-f]{6}-smoke-\S+", "<wt>", s)
    s = re.sub(r"0x[0-9a-f]+|\b[0-9a-f]{7,40}\b", "<h>", s, flags=re.I)
    s = re.sub(r"\d+(?:\.\d+)?", "N", s)
    return re.sub(r"\s+", " ", s)[:200]


class SmokeGate:
    """The smoke hold, one JSON file. Thread-safe within a process (researcher threads); the loop is the only writer."""

    def __init__(self, path: Path | str, clock: Callable[[], float] = time.time) -> None:
        self.path, self.clock, self._lock = Path(path), clock, threading.Lock()

    def _load(self) -> dict[str, Any]:
        try:
            d = json.loads(self.path.read_text())
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, d: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, sort_keys=True))
            os.replace(tmp, self.path)
        except OSError:
            pass                                     # the gate is advisory: never fatal

    def status(self) -> dict[str, Any]:
        return self._load()

    def blocked(self) -> str:
        """'' = this smoke may go (maybe as THE canary); else why it is held."""
        with self._lock:
            d, now = self._load(), self.clock()
            if not d.get("tripped"):
                return ""
            if now < float(d.get("hold_until") or 0):
                return (f"smokes held after an infra failure ({d.get('reason', '')[:160]}); "
                        f"{float(d['hold_until']) - now:.0f}s of backoff left")
            if now - float(d.get("canary") or 0) < CANARY_STALE_S and d.get("canary"):
                return "a canary smoke is in flight after an infra failure"
            d["canary"] = now
            self._save(d)
            return ""

    def wait_s(self) -> float:
        """Seconds until a held smoke may retry (0 = now / not held)."""
        d = self._load()
        if not d.get("tripped"):
            return 0.0
        now = self.clock()
        if now < float(d.get("hold_until") or 0):
            return float(d["hold_until"]) - now
        if d.get("canary") and now - float(d["canary"]) < CANARY_STALE_S:
            return CANARY_POLL_S                     # another session's canary is in flight: poll, don't burn the waits
        return 0.0

    def trip(self, reason: str, sig: str, wt: str = "") -> bool:
        """Hold new smokes (backoff doubles per consecutive trip); True = print the ALERT now (once per signature/hour)."""
        with self._lock:
            d, now = self._load(), self.clock()
            back = min(BACKOFF_MAX_S, float(d.get("backoff") or 0) * 2 if d.get("tripped") else BACKOFF_S)
            alert = not (d.get("alert_sig") == (sig or reason[:80]) and now - float(d.get("alert_ts") or 0) < ALERT_EVERY_S)
            d.update(tripped=True, backoff=back, hold_until=now + back, canary=0, reason=reason[:400], sig=sig,
                     trip_id=int(d.get("trip_id") or 0) + (1 if not d.get("tripped") or alert else 0),
                     last_trip=now, last_wt=wt)
            if alert:
                d.update(alert_sig=sig or reason[:80], alert_ts=now)
            self._save(d)
            return alert

    def cleared(self) -> bool:
        """A smoke got past the environment (it passed, or failed in the idea's own code): the hold is over."""
        with self._lock:
            d = self._load()
            if not d.get("tripped"):
                return False
            d.update(tripped=False, backoff=0, hold_until=0, canary=0, cleared_ts=self.clock())
            self._save(d)
            return True

    def release_canary(self) -> None:
        """The canary ended without a verdict (queue timeout): let the next smoke be the canary."""
        with self._lock:
            d = self._load()
            if d.get("tripped") and d.get("canary"):
                d["canary"] = 0
                self._save(d)

    def correlated(self, wt: str, sig: str) -> int:
        """Record a non-infra smoke failure ``sig`` of worktree ``wt``; the number of DISTINCT sessions that failed with
        this signature within the window (this one included)."""
        if not sig:
            return 0
        with self._lock:
            d, now = self._load(), self.clock()
            fails = [f for f in d.get("fails", []) if now - float(f[0]) < CORRELATED_WINDOW_S]
            if not any(f[1] == wt and f[2] == sig for f in fails):
                fails.append([now, wt, sig])
            d["fails"] = fails[-200:]
            self._save(d)
            return len({f[1] for f in fails if f[2] == sig})

    def flag_correlated(self, wt: str, sig: str) -> bool:
        """Record that ``wt`` got a CORRELATED infra verdict for ``sig``; True when it already had one -- the same
        session failing with the same tail AGAIN after the hold is the idea's own bug, not the environment (a shape
        mismatch shared by three ideas must not hold the loop and re-issue them forever)."""
        with self._lock:
            d = self._load()
            seen = [f for f in d.get("flagged", []) if [f[0], f[1]] == [wt, sig]]
            if not seen:
                d["flagged"] = [*d.get("flagged", [])[-99:], [wt, sig]]
                self._save(d)
            return bool(seen)

    def sessions(self, sig: str) -> list[str]:
        return sorted({f[1] for f in self._load().get("fails", []) if f[2] == sig})


# ------------------------------------------------------------------------------------------------ preflight
def hub_dir(env: Mapping[str, str] | None = None) -> Path:
    """Where ``torch.hub`` caches (its own resolution order: TORCH_HOME, XDG_CACHE_HOME/torch, ~/.cache/torch)."""
    env = os.environ if env is None else env
    home = env.get("TORCH_HOME") or os.path.join(env.get("XDG_CACHE_HOME") or os.path.join(
        env.get("HOME") or os.path.expanduser("~"), ".cache"), "torch")
    return Path(home) / "hub"


def weights_files(model_py: str, model_type: str) -> list[tuple[str, str]]:
    """``[(url, file name)]`` of the pretrained weights ``model_type`` names in a model.py's ``VIT_VARIANTS`` table
    (an alias assignment is followed); [] when unknown -- the preflight then checks nothing about weights."""
    seen: set[str] = set()
    typ = model_type
    while typ and typ not in seen:
        seen.add(typ)
        m = re.search(rf"""["']{re.escape(typ)}["']\s*:\s*\([^\n]*?(https?://[^\s"')]+)""", model_py)
        if m:
            url = m.group(1)
            return [(url, os.path.basename(url.split("?")[0]))]
        a = re.search(rf"""\[["']{re.escape(typ)}["']\]\s*=\s*\w+\[["']([\w.-]+)["']\]""", model_py)
        typ = a.group(1) if a else ""
    return []


def _reachable(url: str, timeout: float = 3.0) -> bool:
    u = urlparse(url)
    try:
        with socket.create_connection((u.hostname, u.port or (443 if u.scheme == "https" else 80)), timeout):
            return True
    except (OSError, TypeError):
        return False


def preflight(model_py: str, model_type: str, *, env: Mapping[str, str] | None = None,
              reachable: Callable[[str], bool] = _reachable) -> str:
    """'' = fine; else a clear ALERT sentence naming the path. Checks: the hub dir resolves (a dangling link anywhere
    on its path is the failure of 2026-10-21), and the pretrained weights file is in ``<hub>/checkpoints`` or its host
    is reachable (torch downloads it)."""
    hub = hub_dir(env)
    probe = hub
    for p in [hub, *hub.parents]:                    # the first existing-or-dangling ancestor decides
        if os.path.lexists(p):
            probe = p
            break
    if os.path.islink(probe) and not os.path.exists(probe):
        return f"torch hub cache path {hub} is under a dangling symlink {probe} -> {os.readlink(probe)} (target missing)"
    if os.path.lexists(probe) and probe != hub and not os.path.isdir(probe):
        return f"torch hub cache path {hub}: {probe} exists but is not a directory"
    if probe == hub and not os.path.isdir(hub):
        return f"torch hub cache {hub} is not a directory"
    if not os.access(probe, os.W_OK | os.X_OK):
        return f"torch hub cache path {probe} is not writable by {os.getuid()}"
    for url, name in weights_files(model_py, model_type):
        f = hub / "checkpoints" / name
        if not f.is_file() or f.stat().st_size == 0:
            if not reachable(url):
                return f"pretrained weights {f} are missing and {urlparse(url).hostname} is unreachable (model.type {model_type})"
    return ""


# ------------------------------------------------------------------------------------------------ re-issue
def reissue_brief(req: Any, cand: Any, why: str) -> Any:
    """``req`` with its brief carrying the previous attempt's hypothesis and diff: the smoke died of the environment, so
    the work is re-applied and smoked again, not redone."""
    brief = (req.brief or "").split(REISSUE_MARK)[0] + REISSUE_MARK + (
        f"\n({why[:200]}) It says NOTHING about the idea. Re-apply the previous attempt's change below (fix only what "
        "your own smoke shows broken), smoke it, and finish with the same hypothesis unless the smoke changes your mind.")
    if cand is not None:
        h = getattr(cand, "hypothesis", None)
        if h is not None:
            brief += f"\nPrevious hypothesis ({h.family}/{h.mechanism_tag}): {h.claim[:1500]}"
        brief += f"\nPrevious diff:\n{(getattr(cand, 'diff_text', '') or '')[:20000]}"
    return replace(req, brief=brief)


def is_infra_unavailable(res: Any) -> bool:
    """A finished session whose smoke was held / killed by the environment (``backend_unavailable`` + infra marker)."""
    return getattr(res, "status", "") == "backend_unavailable" and str(getattr(res, "error", "")).startswith(
        "smoke_unavailable") and "smoke_unavailable: infra" in str(res.error)


def is_smoke_starved(res: Any) -> bool:
    """A finished session whose smoke never got a slot in time (queue timeout; not an infra failure): no verdict on the idea."""
    return getattr(res, "status", "") == "backend_unavailable" and str(getattr(res, "error", "")).startswith("smoke_unavailable")


# ------------------------------------------------------------------------------------------------ the loop's own repair
def reissue_from_file(ctx: Any, store: Any, research: Any, now: float) -> list[str]:
    """The operator's repair for ideas an infra outage already REJECTED ``smoke_failed``: drop ``<state>/reissue_smoke.json``
    (``{"nodes": ["a node", ...]}``). On its next live tick the loop relabels each listed node (REJECTED ``smoke_failed``,
    no commit) ``smoke_infra_reissued`` (no verdict, its bandit credit dropped) and re-issues its request with the stored
    diff and hypothesis; the file is renamed ``.done``. A node in any other state is left alone and reported."""
    from types import SimpleNamespace
    from loop.contracts import CandidateRequest, NodeStatus, Operator

    f = Path(ctx.state_dir) / REISSUE_FILE
    if not f.is_file():
        return []
    try:
        raw = json.loads(f.read_text())
        ids = [str(i) for i in (raw.get("nodes") if isinstance(raw, dict) else raw)]
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        f.rename(f.with_suffix(f".bad-{int(now)}"))
        return [f"ERROR {REISSUE_FILE} unreadable ({exc!r}); renamed .bad, nothing re-issued"]
    started = {e.get("request_id"): e for e in store.events("research_started")}
    live = {r.id for r in [*research.inflight.values(), *research.requeue]}
    out = []
    for nid in ids:
        n = store.get_node(nid)
        m = re.search(r"\brequest=(\S+)", (n.notes or "") if n else "")
        if n is None or n.status != NodeStatus.REJECTED or n.reject_reason != "smoke_failed" or n.commit or not m:
            out.append(f"infra re-issue: {nid} skipped (not a REJECTED smoke_failed node without a commit)")
            continue
        rid = m.group(1)
        if rid in live:
            out.append(f"infra re-issue: {nid} skipped ({rid} already in flight)")
            continue
        note = lambda k: (mm.group(1) if (mm := re.search(rf"\b{k}=(\S+)", n.notes or "")) else "")  # noqa: E731
        h = store.get_hypothesis(n.hypothesis_id or "")
        diff = Path(n.diff_path).read_text() if n.diff_path and Path(n.diff_path).is_file() else ""
        req = CandidateRequest(
            id=rid, task_id=ctx.task_id, strategy=n.strategy, operator=n.operator or Operator.DRAFT, island=n.island,
            parent_ids=tuple((started.get(rid) or {}).get("parents") or [n.parent_id]), brief=h.claim if h else "",
            brief_kind=note("brief"), family_hint=n.family or "", persona=note("persona"), category=note("lens_cat"))
        store.update_node(n.id, reject_reason=REISSUED_REASON, diff_hash="")   # else the re-applied diff is its exact_dup
        store.drop_pulls(n.id)
        out.append(f"infra re-issue: {nid} relabelled {REISSUED_REASON}; " + research.reissue_infra(
            req, SimpleNamespace(error="smoke_unavailable: infra (operator re-issue of an outage-rejected node)",
                                 candidate=SimpleNamespace(hypothesis=h, diff_text=diff) if h else None)))
    f.rename(f.with_suffix(f".done-{int(now)}"))
    return out
