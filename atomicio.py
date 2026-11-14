"""Small filesystem-write helpers shared across the SDK.

Both primitives here were independently reimplemented in half a dozen modules
and scripts during the burst of merges on 2026-07-14. They are consolidated
here verbatim so every caller uses the same tmp-name + rename + backup
semantics. Behaviour is intentionally identical to the inline versions they
replace -- do not "improve" (e.g. add fsync or tmp cleanup) without auditing
every caller, since some rely on the exact current semantics.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write ``text`` to ``path`` atomically via a sibling temp file followed by
    ``os.replace``. The temp file lands next to the target so the rename stays
    on one filesystem. Any exception propagates to the caller (callers that need
    tmp cleanup on failure handle it themselves).

    The temp name is UNIQUE PER CALL (pid + uuid4). It used to be a fixed
    ``<name>.tmp``, which is only atomic against a crash -- NOT against a second
    concurrent writer. A cycle runs ~26 lineages at once and several targets are
    genuinely shared: ``track_state/<track>.json`` (which carries
    ``committed_toward_budget``), the reopen ledger, and the global
    ``stable_reseed_evidence.json`` that feeds the graduation decision. With one
    shared temp name, writer A's half-written bytes could be ``os.replace``d
    into place by writer B, silently installing a torn or stale document over
    budget/graduation state. Unique names make the two writers independent; the
    final ``os.replace`` still decides the winner, which is the intended
    last-writer-wins semantics.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except BaseException:
        # Clean up the temp file on ANY failure (os.replace raises on
        # permission errors, disk full, or if the target vanished).
        tmp.unlink(missing_ok=True)
        raise


def backup_once(path: Path | str, bak_suffix: str, *, verbose: bool = True) -> Path:
    """Copy ``path`` to ``<name><bak_suffix>`` exactly once (skips if the
    backup already exists), preserving metadata via ``shutil.copy2``. Returns
    the backup path. When ``verbose`` (default), prints ``backup: <bak>`` on a
    fresh copy -- matching the one-off scripts' existing output.
    """
    path = Path(path)
    bak = path.with_name(path.name + bak_suffix)
    if not bak.exists():
        shutil.copy2(path, bak)
        if verbose:
            print(f"backup: {bak}")
    return bak
