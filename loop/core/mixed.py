"""loop.core.mixed — v3-c8a B3 MIXED CHANGES. Evidence (10-14): a node changed the dataset (mpp050_448) AND the train
resolution (global / local 336 / 168, while the probes resize to 224) in one node, so its -.0237 measured neither.

The spec's LEVER GROUPS (``intake.lever_groups``: group -> config-key globs; a key no group names is NEUTRAL --
num_workers, activation checkpointing, logging -- never a lever) classify a config diff. One whose keys span >= 2
groups is MIXED (:func:`rule`, ``mixed_change``) unless it is a crossover / a debug fix / a design point, a STACK
(a 2-family compound of already-positive knobs, :func:`loop.select.intake.compound_split`), or its claim carries a
``[joint: ...]`` justification (>= JOINT_MIN characters on why the parts only work together).

``intake.mixed_change``: ``split`` (default) -- the node is admitted with ``mixed=<groups>`` in its notes and the tick
(:func:`held`) builds one QUEUED SIBLING per group (the node's commit with every OTHER group's keys reverted to the
parent's values, ``split_of=<node> group=<g>``; a part on a dead dataset or with a key the parent lacks is not built)
and HOLDS the combined node until every part has screened (>= ``intake.split_screen_seeds`` seeds, or closed); fewer
than 2 buildable parts: the combined node runs as it is. ``reject``: rejected ``mixed_change`` with the groups.
``off``: no rule. Nothing is named."""

from __future__ import annotations

import hashlib
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Mapping, Sequence

from loop.contracts import NodeStatus

K_SPLIT = "intake.split"      # {combined node: {"groups": {g: [keys]}, "parts": {g: node | "-: why"}, "ts", "released"}}
TAG = "mixed="
JOINT = "[joint:"
JOINT_MIN = 40
RELEASE_H = 36.0       # v3 review S4: a combined node is released this long after its split whatever its parts did
_CONFIG_SUFFIXES = (".yaml", ".yml", ".json", ".toml", ".cfg", ".ini")


def lever_map(ctx: Any) -> Mapping[str, Sequence[str]]:
    return getattr(getattr(getattr(ctx, "cfg", None), "intake", None), "lever_groups", None) or {}


def mode(ctx: Any) -> str:
    return str(getattr(getattr(getattr(ctx, "cfg", None), "intake", None), "mixed_change", "split") or "off")


def group_of(ctx: Any, key: str) -> str:
    """The first lever group whose globs match ``key`` ('' = neutral / no map)."""
    for g, pats in lever_map(ctx).items():
        if any(fnmatch(key, p) for p in pats):
            return str(g)
    return ""


def neutral(ctx: Any, key: str) -> bool:
    """A config key no lever group names (num_workers, logging ...): never a lever. False without a lever map."""
    return bool(lever_map(ctx)) and not group_of(ctx, key)


def groups(ctx: Any, keys: Any) -> dict[str, list[str]]:
    """group -> the (sorted) keys of ``keys`` in it; neutral keys left out."""
    out: dict[str, list[str]] = {}
    for k in sorted(keys or ()):
        if g := group_of(ctx, str(k)):
            out.setdefault(g, []).append(str(k))
    return out


def joint(claim: str) -> bool:
    """The claim justifies the coupled change: ``[joint: ...]`` with >= JOINT_MIN characters of words."""
    if JOINT not in (claim or ""):
        return False
    text = claim.split(JOINT, 1)[1].split("]", 1)[0]
    return len(re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ]+", " ", text)).strip()) >= JOINT_MIN


def rule(ctx: Any, store: Any, snapshot: Any, candidate: Any, strategy: Any, operator: Any) -> tuple[str, str]:
    """('mixed_change', fix) for a mixed config diff (see the module docstring); else ('', '')."""
    from loop.contracts import Operator, Strategy
    if mode(ctx) == "off" or not lever_map(ctx):
        return "", ""
    g = groups(ctx, getattr(candidate, "config_diff", None) or {})
    if len(g) < 2 or operator in (Operator.CROSSOVER, Operator.DEBUG) or strategy is Strategy.DESIGN:
        return "", ""
    if joint(str(getattr(candidate.hypothesis, "claim", "") or "")):
        return "", ""
    from loop.select import intake as IN
    try:
        if IN.compound_split(ctx, store, snapshot, candidate) is not None:
            return "", ""                       # a stack of already-positive knobs on one new change
    except Exception:  # noqa: BLE001 - no family map: judged as mixed
        pass
    what = "; ".join(f"{k}: {', '.join(v)}" for k, v in g.items())
    code = [p for p in (getattr(candidate, "changed_paths", ()) or ()) if not str(p).endswith(_CONFIG_SUFFIXES)]
    if code and mode(ctx) == "split":         # v3 review S4: the split reverts CONFIG keys only; the code rides on every part
        return "mixed_change", (f"the change spans {len(g)} lever groups ({what}) AND changes code ({', '.join(code[:3])}): "
                                "it cannot be split into clean parts (each would carry the code); make it one lever, or put "
                                "[joint: why the parts only work together] in the claim")
    how = ("the loop splits it into one node per group (the combined node waits until every part has screened)"
           if mode(ctx) == "split" else "split it into one node per group")
    return "mixed_change", (f"the config change spans {len(g)} lever groups ({what}): a measurement of all of them at "
                            f"once attributes nothing; {how}, or put [joint: why the parts only work together] in the "
                            "claim")


# ------------------------------------------------------------------------------------------------ the tick side
def _new_id(store: Any) -> str:
    seq = len(store.nodes())
    while store.get_node(f"n{seq:04d}") is not None:
        seq += 1
    return f"n{seq:04d}"


def split(ctx: Any, store: Any, adapter: Any, node: Any, now: float) -> dict[str, Any]:
    """Build the per-group sibling parts of mixed node ``node`` (see the module docstring): the record."""
    from loop.contracts import Node as StoreNode, NodeStatus
    from loop.core import datasets as DS
    from loop.core import factorial as F
    from loop.select import intake as IN
    diff = dict(store.get_state(f"config_diff:{node.id}") or {})
    g = groups(ctx, diff)
    rec: dict[str, Any] = {"groups": g, "parts": {}, "ts": now}
    parent = store.get_node(node.parent_id or "")
    if len(g) < 2 or parent is None:
        rec["parts"]["-"] = "-: fewer than 2 lever groups" if len(g) < 2 else "-: no parent"
        return rec
    from loop.adapters import flatten
    try:                                          # the parent's EFFECTIVE config (root values included)
        pcfg = flatten(adapter.config_at(parent.commit)) if hasattr(adapter, "config_at") else {}
    except Exception:  # noqa: BLE001 - unreadable: the stored chain below
        pcfg = {}
    if not pcfg:
        pcfg = {**flatten(store.get_state(IN.K_BASE) or {}), **IN.parent_config(store, parent.id)}
    claim = ""
    if node.hypothesis_id and (h := store.get_hypothesis(node.hypothesis_id)) is not None:
        claim = str(h.claim or "")
    notes = re.sub(rf"\s*\b{re.escape(TAG)}\S+", "", node.notes or "").strip()
    notes = re.sub(r"\s*\bbrief=bold\b", "", notes).strip()    # v3 review S1: a part is no bold node (it is the combined node's unit)
    cfg_rel = getattr(adapter, "config_rel", "config.yaml")
    for grp, keys in g.items():
        other = [k for o, ks in g.items() if o != grp for k in ks]
        if new := [k for k in other if k not in pcfg]:
            rec["parts"][grp] = f"-: {', '.join(new)} new (no parent value to revert to)"
            continue
        part_diff = {k: v for k, v in diff.items() if k not in other}
        if DS.DATA_KEY in part_diff and (r := DS.rule(ctx, store, claim, IN.dataset_identity(
                ctx, store, part_diff[DS.DATA_KEY]), exclude=node.id))[0]:
            rec["parts"][grp] = f"-: {r[0]}"
            continue
        revert = {k: pcfg[k] for k in other}
        try:
            sha, err = F.build_point(adapter, node.commit, revert,
                                     [F.Knob(k, diff.get(k), v, "lin") for k, v in revert.items()], {},
                                     f"split-{node.id}-{grp}"[:60], cfg_rel)
        except Exception as exc:  # noqa: BLE001 - a part that does not build is left out
            sha, err = "", repr(exc)[:200]
        if not sha or sha == node.commit:
            rec["parts"][grp] = f"-: not built ({err or 'no change'})"
            continue
        pid, path = _new_id(store), ""
        if hasattr(adapter, "diff_commits") and getattr(ctx, "state_dir", ""):
            try:
                p = Path(ctx.state_dir) / "diffs" / f"{pid}.diff"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(adapter.diff_commits(parent.commit, sha))
                path = str(p)
            except Exception:  # noqa: BLE001 - no diff file: the node still runs
                path = ""
        fam = IN.family_of(ctx, keys[0]) or node.family
        store.put_node(StoreNode(
            id=pid, task_id=node.task_id, commit=sha, parent_id=parent.id, strategy=node.strategy,
            operator=node.operator, family=fam, mechanism_tag=f"{node.mechanism_tag}_{grp}"[:120],
            diff_hash=hashlib.sha256(f"{parent.commit}\n{sha}".encode()).hexdigest(), status=NodeStatus.QUEUED,
            island=node.island, diff_path=path, hypothesis_id=node.hypothesis_id, levers=node.levers,
            created_ts=now, updated_ts=now, notes=f"{notes} split_of={node.id} group={grp}".strip()))
        store.set_state(f"config_diff:{pid}", part_diff)
        rec["parts"][grp] = pid
    store.log("mixed_split", node_id=node.id, groups=g, parts=rec["parts"])
    return rec


def parts(rec: Mapping[str, Any]) -> list[str]:
    return [v for v in (rec.get("parts") or {}).values() if isinstance(v, str) and not v.startswith("-")]


def held(ctx: Any, store: Any, adapter: Any, data: Any, statuses: Mapping[str, Any], now: float,
         acts: list[str]) -> frozenset[str]:
    """The QUEUED mixed nodes (``mixed=`` in their notes) held this tick: their parts are built on first sight
    (:func:`split`), and the node waits until every part has screened (>= ``intake.split_screen_seeds`` seeds of its
    own, or a status past SCREENING); < 2 built parts: never held."""
    from loop.tick import status_name
    book, out, dirty = dict(store.get_state(K_SPLIT) or {}), set(), False
    need = int(getattr(getattr(ctx.cfg, "intake", None), "split_screen_seeds", 2) or 0)
    seeds: dict[str, set[int]] | None = None
    for nid, st in statuses.items():
        if status_name(st) != "QUEUED":
            continue
        node = store.get_node(nid)
        if node is None or TAG not in (node.notes or "") or "split_of=" in (node.notes or ""):
            continue
        rec = book.get(nid)
        if rec is None:
            rec = book[nid] = split(ctx, store, adapter, node, now)
            dirty = True
            acts.append(f"mixed change {nid}: split into {rec['parts']}")
            if dead := [g for g, v in (rec.get("parts") or {}).items() if isinstance(v, str) and v.startswith("-: dead_dataset")]:
                # v3 review S4: its dataset part is a dead dataset and was skipped -- the combined node would run confounded
                # by that dataset with nothing to attribute it to: refused with the reason (never run silently)
                store.update_node(nid, status=NodeStatus.REJECTED, reject_reason=(
                    f"mixed_change: the {', '.join(dead)} part is on a dead dataset (dead_dataset) and was skipped; the "
                    "combined change would run confounded by it"))
                store.log("mixed_rejected", node_id=nid, why="dead_dataset part", groups=dead)
                acts.append(f"mixed change {nid}: rejected (its {', '.join(dead)} part is on a dead dataset)")
                continue
        ps = parts(rec)
        if len(ps) < 2:
            continue
        if seeds is None:
            seeds = {}
            for o in getattr(data, "obs", ()):
                seeds.setdefault(o.node, set()).add(o.seed)
        wait = [p for p in ps if status_name(statuses.get(p, "QUEUED")) in ("QUEUED", "SCREENING")
                and len((seeds or {}).get(p, ())) < need]
        stuck = [p for p in ps if str(status_name(statuses.get(p, "QUEUED"))).upper() in ("CRASHED", "VOID", "REJECTED")]
        old = now - float(rec.get("ts") or now) >= RELEASE_H * 3600.0
        if wait and (stuck or old):          # v3 review S4: a dead part / 36 h never holds the combined node for good
            wait = []
            if not rec.get("released"):
                store.log("mixed_release_forced", node_id=nid, why="part crashed / void" if stuck else f"{RELEASE_H:g} h")
                acts.append(f"mixed change {nid}: released ({'part ' + stuck[0] + ' is dead' if stuck else f'{RELEASE_H:g} h passed'})")
        if wait:
            out.add(nid)
        elif not rec.get("released"):
            rec["released"], dirty = now, True
            store.log("mixed_released", node_id=nid, parts=ps)
            acts.append(f"mixed change {nid}: every part screened ({ps}); the combined node may run")
    if dirty:
        store.set_state(K_SPLIT, book)
    return frozenset(out)
