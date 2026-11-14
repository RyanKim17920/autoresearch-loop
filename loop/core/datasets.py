"""loop.core.datasets — v3-c8a B2 DEAD DATASET LANES. Evidence (10-14): 14 of 14 data-bank / curated datasets lost to
the incumbent, and mpp050_448 alone ran 3x (a node, a node, a node at -.0237) -- each a full session and ~5 GPU-h.

Each tick (:func:`record`, beside the anti-repeat set) the loop writes, per DATASET (``data.dataset_dir`` content
identity, :func:`loop.select.intake.dataset_identity`: a data-bank dataset by its spec hash, else the path), every
node that set it in its own config diff with that node's own-increment posterior (mean gain over its parent, seeds).
A dataset is DEAD (v3 review M4) once some CLEAN single-lever node on it (config diff = ``data.dataset_dir`` plus keys no
lever group names, no code diff) of the current epoch, measured against the current incumbent, has >=
``ideation.dead_dataset_min_seeds`` paired seeds with a mean gain below -1.28 x its posterior sd (a -.0004 on 4 seeds
is noise, not dead); an incumbent change re-allows every dataset. At intake (:func:`rule`) a new node pointing ``data.dataset_dir`` (or a ``dataset_spec``) at a dead dataset is
rejected ``dead_dataset`` unless its claim's ``[differs from: ...]`` names AND explains each such node (the
mechanisms dead_mechanism check, :func:`loop.core.mechanisms.explained`). The briefs' data-bank section lists the
table (:func:`lines`). Nothing is named."""

from __future__ import annotations

from typing import Any, Mapping

K_DATASETS = "intake.datasets"     # {identity: {"name": str, "nodes": {node: [mean gain, seeds]}}}
DATA_KEY = "data.dataset_dir"


def _name(ctx: Any, path: Any) -> str:
    from loop import databank as DB
    if not isinstance(path, str):
        return str(path)
    try:
        return DB.dataset_of(ctx, path) or path
    except Exception:  # noqa: BLE001 - no databank config: the path itself
        return path


_CFG_SUFFIX = (".yaml", ".yml", ".json", ".toml", ".cfg", ".ini")
Z_DEAD = 1.28     # dead = the mean gain is below -Z_DEAD x its posterior sd (one-sided ~90%), not merely below 0


def _code_free(store: Any, node: Any) -> bool:
    """The node's diff touches no file but configs (a node without a readable diff is not known to be code-free)."""
    import re
    from pathlib import Path
    dp = getattr(node, "diff_path", "") or ""
    try:
        text = Path(dp).read_text() if dp else ""
    except OSError:
        return False
    paths = re.findall(r"^\+\+\+ (?:b/)?(\S+)", text, re.M)
    return bool(text.strip()) and all(p.endswith(_CFG_SUFFIX) for p in paths)


def _clean(ctx: Any, store: Any, node: Any, diff: Mapping[str, Any]) -> bool:
    """A CLEAN single-lever dataset node: its config diff is ``data.dataset_dir`` plus keys no lever group names
    (``intake.lever_groups``: neutral), and no code changed (a confounded node says nothing about the dataset)."""
    from loop.core import mixed as MX
    if not MX.lever_map(ctx):
        return set(diff) == {DATA_KEY} and _code_free(store, node)
    return all(k == DATA_KEY or MX.neutral(ctx, str(k)) for k in diff) and _code_free(store, node)


def table(ctx: Any, store: Any, beliefs: Mapping[str, Any], led: Any = None) -> dict[str, dict[str, Any]]:
    """{identity: {name, nodes: {node: [m, n, v, parent, clean]}}} over every node of the CURRENT epoch/base whose OWN
    config diff sets ``data.dataset_dir`` (``beliefs``: node -> an object with .m / .v / .n, the ideation view's
    own-increment posterior; unmeasured: [0, 0, 0]); ``clean``: a single-lever, code-free node (:func:`_clean`)."""
    from loop.select.intake import dataset_identity
    out: dict[str, dict[str, Any]] = {}
    epoch = getattr(led, "epoch", "") if led is not None else ""
    lnodes = getattr(led, "nodes", None) or {}
    for n in store.nodes():
        diff = store.get_state(f"config_diff:{n.id}") or {}
        path = diff.get(DATA_KEY)
        if not isinstance(path, str) or not path:
            continue
        if epoch and (ln := lnodes.get(n.id)) is not None and getattr(ln, "base", epoch) != epoch:
            continue                                    # an older base: its pairing is not this epoch's
        ident = dataset_identity(ctx, store, path)
        b = beliefs.get(n.id)
        row = out.setdefault(ident, {"name": _name(ctx, path), "nodes": {}})
        row["nodes"][n.id] = ([round(float(b.m), 6), int(b.n), round(float(getattr(b, "v", 0.0)), 8)]
                              if b is not None else [0.0, 0, 0.0]) + [n.parent_id or "", _clean(ctx, store, n, diff)]
    return out


K_DS_INC = "intake.datasets_incumbent"


def record(ctx: Any, store: Any, beliefs: Mapping[str, Any], led: Any = None, incumbent: str = "") -> None:
    """Write the per-dataset table (state ``intake.datasets``) and the incumbent it was read at when they changed."""
    got = table(ctx, store, beliefs, led)
    if store.get_state(K_DATASETS) != got:
        store.set_state(K_DATASETS, got)
    if incumbent and store.get_state(K_DS_INC) != incumbent:
        store.set_state(K_DS_INC, incumbent)


def min_seeds(ctx: Any) -> int:
    from loop.core.levers import opt
    return int(opt(ctx, "dead_dataset_min_seeds") or 0)


def is_dead(rec: Any, k: int, incumbent: str = "") -> bool:
    """One node's record is a dead-dataset measurement: clean single-lever, >= ``k`` paired seeds, mean below
    ``-Z_DEAD`` x its posterior sd, and measured against the CURRENT incumbent (an incumbent change re-allows the
    dataset: it was not measured on the new base of ideas)."""
    try:
        m, n, v, parent, clean = rec[0], rec[1], rec[2], rec[3], rec[4]
    except (IndexError, TypeError):
        return False                                    # a pre-v3-review record: no variance / cleanness known
    return bool(clean) and int(n) >= k and float(m) < -Z_DEAD * (max(0.0, float(v)) ** 0.5) \
        and (not incumbent or parent == incumbent)


def dead_nodes(ctx: Any, store: Any, ident: str) -> list[tuple[str, float, int]]:
    """(node, mean gain, seeds) of the nodes that measured dataset ``ident`` clearly below its parent (:func:`is_dead`)."""
    k = min_seeds(ctx)
    if k <= 0:
        return []
    row = (store.get_state(K_DATASETS) or {}).get(ident) or {}
    inc = store.get_state(K_DS_INC) or ""
    return sorted(((nid, float(r[0]), int(r[1])) for nid, r in (row.get("nodes") or {}).items() if is_dead(r, k, inc)),
                  key=lambda r: r[1])


def rule(ctx: Any, store: Any, claim: str, ident: str, exclude: str = "") -> tuple[str, str]:
    """('dead_dataset', fix) when dataset ``ident`` is dead and ``claim``'s ``[differs from: ...]`` does not name and
    explain each dead node (``exclude``: the node being judged itself); else ('', '')."""
    from loop.core.mechanisms import DIFFERS, explained
    dead = [r for r in dead_nodes(ctx, store, ident) if r[0] != exclude]
    if not dead:
        return "", ""
    differs = claim.split(DIFFERS, 1)[1] if DIFFERS in claim else ""
    miss = [r for r in dead if not explained(differs, r[0])]
    if not miss:
        return "", ""
    name = ((store.get_state(K_DATASETS) or {}).get(ident) or {}).get("name") or ident
    return "dead_dataset", (f"dataset {name} was already measured below its parent on >= {min_seeds(ctx)} paired "
                            "seeds: " + ", ".join(f"{nid} ({m:+.4f}, {n} seeds)" for nid, m, n in miss)
                            + ". Pick another dataset (or none), or state a NEW mechanism that differs from those "
                            "nodes with [differs from: <id> <what is materially different> ...] naming each")


def candidate_rule(ctx: Any, store: Any, candidate: Any) -> tuple[str, str]:
    """:func:`rule` for a candidate whose config diff sets ``data.dataset_dir``."""
    from loop.select.intake import dataset_identity
    path = (getattr(candidate, "config_diff", None) or {}).get(DATA_KEY)
    if not isinstance(path, str) or not path:
        return "", ""
    return rule(ctx, store, str(getattr(candidate.hypothesis, "claim", "") or ""), dataset_identity(ctx, store, path))


def lines(ctx: Any, store: Any) -> list[str]:
    """The briefs' per-dataset table: measured nodes, seeds, seed-weighted mean gain, DEAD flags (worst first)."""
    k, rows = min_seeds(ctx), []
    inc = store.get_state(K_DS_INC) or ""
    for ident, row in (store.get_state(K_DATASETS) or {}).items():
        nodes = row.get("nodes") or {}
        n = sum(int(r[1]) for r in nodes.values())
        mean = sum(float(r[0]) * int(r[1]) for r in nodes.values()) / n if n else 0.0
        dead_ids = sorted(nid for nid, r in nodes.items() if k > 0 and is_dead(r, k, inc))
        rows.append((not dead_ids, mean if n else 1.0, f"- {row.get('name') or ident}: {len(nodes)} node(s), {n} paired "
                     f"seed(s), mean own gain {mean:+.4f}" + (" -- DEAD (a new node on it needs [differs from: ...] "
                                                               "naming " + ", ".join(dead_ids) + ")" if dead_ids else "")
                     + " [" + ", ".join(f"{nid} {float(r[0]):+.4f}/{int(r[1])}" + ("" if len(r) < 5 or r[4] else "*")
                                        for nid, r in sorted(nodes.items())) + "]"))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in rows]
