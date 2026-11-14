"""loop.knowledge -- the project's curated KNOWLEDGE FILE, briefed to every research session (11-08).

The operator's per-task picture (which axis a known change moves, where the gap to the reference model is, what is
decodable from the embedding) lived in hand-written notes the sessions never saw, and the brief told them nothing about
how big a change must be to be seen. This module feeds both into every v3 brief the way the board digest and the root
audit do (``loop.agents.briefs.research_blocks``), from markdown files instead of prompt literals:

- :func:`block` -- ``## Project knowledge`` (the files, minus the ``## Tested mechanisms`` section) and ``## The bar``:
  each axis must plausibly move by >= ``knowledge.bar_sd`` paired sd, computed from ``loop.axis_sd`` / ``panel_sd``.
- :func:`descriptions` -- the ``- local:<run>: <what it changed>`` lines of the ``## Tested mechanisms`` section: what
  each imported manual run (loop.localruns) actually tested, for the mechanism memory (loop.core.tested).

``loop.knowledge.files`` names the files (repo-relative or absolute); empty = ``<domain prompt_dir>/../knowledge.md``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loop import project as P

HEADING = "## Project knowledge"
BAR_HEADING = "## The bar: how big a change must be to be seen"
TESTED_SECTION = "## Tested mechanisms"
_DESC = re.compile(r"^\s*[-*]\s*local:([A-Za-z0-9_.\-]+)\s*[:=-]\s*(.+?)\s*$")


def cfg_of(ctx: Any) -> Any:
    return getattr(getattr(ctx, "cfg", None), "knowledge", None)


def enabled(ctx: Any) -> bool:
    return bool(getattr(cfg_of(ctx), "enabled", False))


def paths(ctx: Any) -> list[Path]:
    """The knowledge files that exist (configured, else the domain overlay's sibling ``knowledge.md``)."""
    kc = cfg_of(ctx)
    files = tuple(getattr(kc, "files", ()) or ())
    out: list[Path] = []
    if files:
        for f in files:
            p = Path(f)
            out.append(p if p.is_absolute() else P.REPO / p)
    elif (d := P.domain_dir()) is not None:
        out.append(Path(d).parent / "knowledge.md")
    return [p for p in out if p.is_file()]


def _text(ctx: Any) -> str:
    parts = []
    for p in paths(ctx):
        try:
            parts.append(p.read_text())
        except OSError:
            continue
    return "\n\n".join(x.strip() for x in parts if x.strip())


def _split(text: str) -> tuple[str, str]:
    """(the text without the ``## Tested mechanisms`` section, that section)."""
    i = text.find(TESTED_SECTION)
    if i < 0:
        return text, ""
    j = text.find("\n## ", i + len(TESTED_SECTION))
    sec = text[i:] if j < 0 else text[i:j + 1]
    rest = text[:i] + ("" if j < 0 else text[j + 1:])
    return rest.strip(), sec


def descriptions(ctx: Any) -> dict[str, str]:
    """``run name / name prefix -> what it tested``, from the ``## Tested mechanisms`` lines ``- local:<name>: text``
    (a prefix ``fb`` covers ``a root replicate run`` / ``a root replicate run``: :func:`describe`)."""
    if not enabled(ctx):
        return {}
    _, sec = _split(_text(ctx))
    out = {}
    for line in sec.splitlines():
        if m := _DESC.match(line):
            out[m.group(1)] = m.group(2)
    return out


def describe(names: Any, desc: dict[str, str]) -> str:
    """The description of a run group (its names): the exact name first, else the longest ``prefix-`` match."""
    best, size = "", -1
    for n in names or ():
        n = str(n)
        if n in desc:
            return desc[n]
        for k, v in desc.items():
            if n.startswith(k + "-") and len(k) > size:
                best, size = v, len(k)
    return best


def _short(key: str) -> str:
    return key.replace("nb_", "").replace("_mean", "").replace("final_probe_", "")


def bar_line(ctx: Any) -> str:
    """The noise and the bar, computed: per axis >= ``bar_sd`` x the measured paired sd."""
    cfg = getattr(ctx, "cfg", None)
    k = float(getattr(cfg_of(ctx), "bar_sd", 2.0) or 2.0)
    sds = {str(a): float(v) for a, v in (getattr(cfg, "axis_sd", {}) or {}).items()}
    panel = float(getattr(cfg, "panel_sd", 0.0) or 0.0)
    if not sds and not panel:
        return ""
    ax = "; ".join(f"{_short(a)} sd {v:.4f} -> needs >= {k * v:.4f}" for a, v in sds.items() if v > 0)
    return (f"{BAR_HEADING}\nThe loop's measured PAIRED per-run noise (one seed vs the same-seed control): "
            f"{ax}{f'; panel sd {panel:.4f}' if panel else ''}. Past sessions predicted gains of +-.004 on the panel: "
            f"that is pure noise and was never measurable. A change is worth a GPU run only if it can plausibly move "
            f"ONE SPECIFIC AXIS by >= {k:g} paired sd (the numbers above) through a mechanism you can name -- say which "
            f"axis, by how much and why in the slate (median / p_gap) and in hypothesis.predictions. A change whose "
            f"honest effect is smaller than that on every axis is not a candidate: drop it or combine it into a "
            f"bigger change.")


def block(ctx: Any, store: Any = None) -> str:
    """The brief section ('' when off): the knowledge files (without the tested-mechanisms list) + the bar."""
    if not enabled(ctx):
        return ""
    body, _ = _split(_text(ctx))
    cap = int(getattr(cfg_of(ctx), "max_chars", 6000) or 6000)
    if len(body) > cap:
        body = body[:cap] + "\n... (truncated: loop.knowledge.max_chars)"
    out = []
    if body:
        out.append(f"{HEADING} (operator-curated; {', '.join(p.name for p in paths(ctx))})\n{body}")
    if paths(ctx) and (bar := bar_line(ctx)):        # a project without a knowledge file keeps its briefs as they were
        out.append(bar)
    return "\n\n".join(out)
