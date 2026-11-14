"""loop.core.altbase — NON-INCUMBENT STARTING POINTS (v3-c9 C2; RSI-top "Escaping Exploration Collapse": ground new
directions in distinct branches of the history DAG, not always its tip).

Every explore / bold session started from the incumbent, so the search kept re-parameterising the incumbent's own
terms. ``ideation.alt_base_share`` of those sessions (a measured arm, not a switch) start from an ALTERNATE base:

- ``cluster``      the best node (by its shrunk own-increment posterior) of a mechanism cluster (``loop.core.alloc``)
                   that is neither saturated-negative nor part of the incumbent's lineage
- ``prev_epoch``   the best node of the PREVIOUS base epoch, in its PORT onto the current base (``Ledger.port``: the
                   existing port machinery makes the port; a previous-epoch node without one is skipped, never run
                   on the old base)
- ``top_outside``  the top posterior node of the current epoch outside the incumbent's ancestry
- ``root``         the current epoch's root base node (the plain base, without the incumbent's stack): exempt from the
                   ancestry / containment rules (far by construction, ``alt_dist=far``), still sound + buildable; its own
                   runs are the paired control. A root child with a positive own increment is auto-ported onto the
                   incumbent (:func:`port_due`, ``port_src=alt_root``)

A parent must be SOUND (store status not rejected / void / crashed / vetoed, derived status not failed / void /
crashed, a commit), have >= ``alt_base_min_seeds`` PAIRED seeds, and an own gain that is not clearly negative
(m >= -``alt_base_z`` sd). The child's credit is its OWN paired increment over the chosen parent (the posterior's
belief; the parent's gain is never inherited) plus its gain over the incumbent for the bar (``bar.paired`` chains the
parents). Each such node is tagged ``arm=alt_base alt_of=<parent> alt_src=<source>`` so the meta report can compare
credit per GPU-h of the arm with the incumbent-based sessions (:func:`section`), as the fork-and-flush A/B does.

Nothing here names a node: sources, thresholds and the share are config / the ledger's."""

from __future__ import annotations

import hashlib
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from loop.core import model as M

ARM = "alt_base"
TAG = "arm=alt_base"
SOURCES = ("cluster", "prev_epoch", "top_outside")
ROOT = "root"                               # the fourth source: the epoch's root base node (not in the rotation)
PORT_TAG = "port_src=alt_root"              # the notes tag of an auto-port of a root child onto the incumbent
K_ALT = "ideation.alt_base"                 # {n: eligible sessions planned, alt: of which alt-based, uses: {node: n}}
K_ALT_BRANCH = "ideation.alt_base.branch"   # {"<round>": {"<branch>": node | ""}}: a fork-and-flush branch's fixed base
BAD_STORE = frozenset({"rejected", "void", "crashed", "vetoed", "audit_failed", "not_confirmed"})
BAD_DERIVED = frozenset({"FAILED", "VOID", "CRASHED", "VETOED"})
EVAL_STORE = frozenset({"evaluating", "eval_passed"})     # eval-only readout nodes: no checkpoint of their own
NO_DIVERSE = "alt_base: no diverse candidate"
BUCKETS = (("near", 0.1, 0.3), ("mid", 0.3, 0.6), ("far", 0.6, 1.01))     # diff-line Jaccard distance buckets
MIN_LINE = 6                                               # added diff lines shorter than this are noise ("}", "else:")


def opt(ctx: Any, name: str) -> Any:
    from loop.core.levers import opt as o
    return o(ctx, name)


@dataclass(frozen=True)
class Pick:
    node: str                # the base the session starts from (a port, for prev_epoch)
    source: str              # one of SOURCES
    m: float                 # its own-increment posterior over ITS parent
    sd: float
    n: int                   # paired seeds
    why: str
    cluster: str = ""
    via: str = ""            # prev_epoch: the previous-epoch node it ports
    rank: float | None = None   # the key picks of a source are ranked by (default: m; prev_epoch: the OLD node's own gain)
    dist: float | None = None   # diff-line Jaccard distance of its cumulative diff from the incumbent's (None: unmeasured)
    contain: float | None = None  # share of its OWN added diff lines already in the incumbent's cumulative diff
    gain: float | None = None   # its paired composite gain over the incumbent (and the se of it)
    gain_se: float | None = None


# ---------------------------------------------------------------------------------------------- the quota
def state(store: Any) -> dict[str, Any]:
    st = dict(store.get_state(K_ALT) or {})
    st.setdefault("n", 0)
    st.setdefault("alt", 0)
    st["uses"] = dict(st.get("uses") or {})
    return st


def due(n: int, share: float) -> bool:
    """Is the ``n``-th (0-based) eligible session an alt-base one? A deterministic hash of ``n`` below ``share``
    (random-like, so the arm is not confounded with time; the realised share converges to ``share``)."""
    if share <= 0:
        return False
    if share >= 1:
        return True
    u = int(hashlib.sha1(f"alt-base:{n}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return u < share


def count(store: Any, plan: Mapping[str, Any], sign: int = 1) -> None:
    """A reserved plan counts as an eligible session (and, when alt-based, as an alt one and a use of its base);
    ``sign`` -1 rolls it back (an unstarted brief dropped)."""
    alt = plan.get("alt") or {}
    if plan.get("branch"):                         # a fork-and-flush branch's base is fixed per round, not a quota draw
        eligible = False
    else:
        eligible = bool(plan.get("alt_eligible"))
    if not eligible and not alt:
        return
    st = state(store)
    if eligible:
        st["n"] = max(0, int(st["n"]) + sign)
        st["alt"] = max(0, int(st["alt"]) + sign * bool(alt))
    if alt:
        node = str(alt.get("node") or "")
        st["uses"][node] = max(0, int(st["uses"].get(node, 0)) + sign)
    store.set_state(K_ALT, st)


# ---------------------------------------------------------------------------------------------- the candidates
def _derived(statuses: Mapping[str, Any] | None, nid: str) -> str:
    s = (statuses or {}).get(nid)
    s = getattr(s, "status", s)
    return str(getattr(s, "name", None) or getattr(s, "value", None) or s or "").upper()


def _eval_family(ctx: Any) -> str:
    return getattr(getattr(getattr(ctx, "cfg", None), "evalonly", None), "family", "") or "eval_readout"


def sound(store: Any, statuses: Mapping[str, Any] | None, nid: str, ctx: Any = None) -> bool:
    """Not rejected / void / crashed / vetoed / failed, a commit to start from, and not an eval-only readout node (it
    has no checkpoint of its own: family ``evalonly.family`` / ``evaluating`` / ``eval_passed``)."""
    sn = store.get_node(nid)
    if sn is None or not sn.commit:
        return False
    if getattr(sn.status, "value", str(sn.status)) in BAD_STORE | EVAL_STORE:
        return False
    if getattr(sn, "family", "") == _eval_family(ctx) or f"fams={_eval_family(ctx)}" in (getattr(sn, "notes", "") or ""):
        return False
    return _derived(statuses, nid) not in BAD_DERIVED


def _repo(ctx: Any) -> str:
    return str(getattr(getattr(ctx, "spec", None), "repo", "") or "")


def buildable(ctx: Any, store: Any, led: Any, nid: str) -> bool:
    """The pick's commit object exists in the repo (``git cat-file -e``) and one of its runs left a checkpoint (the
    session builds on its code and its child is compared with / resumed from it). Unknown (no repo configured) passes."""
    if not bool(opt(ctx, "alt_base_require_ckpt")):
        return True
    sn = store.get_node(nid)
    repo = _repo(ctx)
    if repo and sn is not None and sn.commit:
        try:
            if subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{sn.commit}^{{commit}}"],
                              capture_output=True, timeout=20).returncode != 0:
                return False
        except (OSError, subprocess.SubprocessError):
            return False
    runs = [r for r in getattr(led, "runs", ()) or () if r.node == nid and getattr(r, "output_dir", "")]
    if not runs:
        return not repo                              # no run record to check (a bare test ledger): unknown passes
    for r in runs:
        d = Path(r.output_dir)
        if d.is_dir() and any(next(d.glob(pat), None) is not None for pat in ("latest.pt", "step_*.pt")):
            return True
    return False


# ---------------------------------------------------------------------------------------------- code distinctness
def ancestry(led: Any, nid: str) -> set[str]:
    """``nid`` and everything it was built from: parents, SECOND parents (crossovers) and port originals, transitively."""
    from loop.core.ideation import lineage
    return set(lineage(led, nid, second=True))


def added_lines(store: Any, nid: str) -> frozenset[str]:
    """The node's own added diff lines (stripped; short lines dropped) plus its config-diff leaves."""
    out: set[str] = set()
    sn = store.get_node(nid)
    try:
        text = Path(sn.diff_path).read_text(errors="replace") if sn is not None and getattr(sn, "diff_path", "") else ""
    except OSError:
        text = ""
    for line in text.splitlines():
        if line.startswith("+") and not line.startswith("+++") and len(t := line[1:].strip()) >= MIN_LINE:
            out.add(t)
    try:
        cd = store.get_state(f"config_diff:{nid}")
    except Exception:  # noqa: BLE001
        cd = None
    if cd:
        from loop.adapters import flatten
        out |= {f"cfg {k}={v}" for k, v in flatten(cd).items()}
    return frozenset(out)


class Lines:
    """Cached added-line sets: per node (own) and per ancestry (cumulative) of one candidates() pass."""

    def __init__(self, store: Any, led: Any) -> None:
        self.store, self.led = store, led
        self._own: dict[str, frozenset[str]] = {}

    def own(self, nid: str) -> frozenset[str]:
        if nid not in self._own:
            self._own[nid] = added_lines(self.store, nid)
        return self._own[nid]

    def cumulative(self, nid: str) -> frozenset[str]:
        out: set[str] = set()
        for a in ancestry(self.led, nid):
            out |= self.own(a)
        return frozenset(out)

    def contained(self, nid: str, base: frozenset[str]) -> float | None:
        """Share of the node's OWN added lines that appear in ``base`` (None: it has none / not measurable)."""
        own = self.own(nid)
        if not own and (po := getattr(self.led.nodes.get(nid), "port_of", None)):
            own = self.own(po)                        # a port without its own diff: its original's
        return (len(own & base) / len(own)) if own and base else None

    def distance(self, nid: str, inc: frozenset[str]) -> float | None:
        """Jaccard distance of the node's cumulative added lines from the incumbent's (None: not measurable)."""
        cum = self.cumulative(nid)
        if not cum or not inc:
            return None
        return 1.0 - len(cum & inc) / len(cum | inc)


def gain_vs_incumbent(v: Any, nid: str) -> tuple[float, float] | None:
    """(mean, se) of the paired per-seed composite difference of ``nid`` over the incumbent on the VISIBLE ledger (the
    same base only); None when fewer than 2 shared seeds."""
    led = v.led
    a, b = led.nodes.get(nid), led.nodes.get(v.incumbent)
    if a is None or b is None or a.base != b.base:
        return None
    by = getattr(v.post, "by", None) or led.by_node()
    x = M.seed_means(by.get(nid, ()), M.COMPOSITE, v.weights)
    y = M.seed_means(by.get(v.incumbent, ()), M.COMPOSITE, v.weights)
    both = sorted(x.keys() & y.keys())
    if len(both) < 2:
        return None
    d = [x[k] - y[k] for k in both]
    return statistics.fmean(d), statistics.stdev(d) / math.sqrt(len(d))


def diverse(ctx: Any, store: Any, v: Any, nid: str, anc: set[str], inc_lines: frozenset[str], lines: Lines
            ) -> tuple[float | None, float | None, tuple[float, float] | None] | None:
    """(distance, containment, gain) when ``nid`` is a different code branch than the incumbent, else None: its parent
    is the incumbent or a non-root ancestor of it, > ``alt_base_max_contain`` of its own added lines are already in the
    incumbent's cumulative diff, its cumulative diff is closer than ``alt_base_min_dist`` (Jaccard), or its gain over
    the incumbent is below -``alt_base_gain_z`` se."""
    led = v.led
    node = led.nodes[nid]
    from loop.core.data import ROOT
    roots = {led.base_root, ROOT}
    par = node.parent
    if par is not None and par not in roots and (par == v.incumbent or par in anc):
        return None
    if nid in anc:
        return None
    ct = lines.contained(nid, inc_lines)
    if ct is not None and ct > float(opt(ctx, "alt_base_max_contain")):
        return None
    ds = lines.distance(nid, inc_lines)
    if ds is not None and ds < float(opt(ctx, "alt_base_min_dist")):
        return None
    g = gain_vs_incumbent(v, nid)
    if g is not None and g[1] > 0 and g[0] < -float(opt(ctx, "alt_base_gain_z")) * g[1]:
        return None
    return ds, ct, g


def standing(v: Any, nid: str, min_seeds: int, z: float) -> tuple[float, float, int] | None:
    """(m, sd, paired seeds) when the node's own gain is measured on >= ``min_seeds`` PAIRED seeds and is not clearly
    negative (m >= -z sd); None otherwise."""
    b = v.beliefs.get(nid)
    if b is None:
        return None
    est = ((getattr(v.post, "est", None) or {}).get(nid) or {}).get(M.COMPOSITE)
    paired = int(est.paired) if est is not None else int(b.n)
    sd = math.sqrt(max(0.0, b.v))
    if paired < min_seeds or b.m < -z * sd:
        return None
    return float(b.m), sd, paired


def _skip_node(led: Any, nid: str) -> bool:
    """The base root, a wave's points / x* (a knob design on its parent, not a direction)."""
    from loop.core.acquire import wave_node
    from loop.core.alloc import code_kind
    node = led.nodes[nid]
    return nid == led.base_root or wave_node(node) or code_kind(node.notes)[0] == "wave"


def candidates(ctx: Any, store: Any, v: Any, statuses: Mapping[str, Any] | None = None) -> dict[str, list[Pick]]:
    """{source: ranked picks (best first)} -- every eligible alternate base of this view, per source. A pick is a
    different code branch from the incumbent (:func:`diverse`: ancestry with second parents and ports, diff-line
    containment and Jaccard distance, gain not clearly negative) and is buildable (:func:`buildable`)."""
    from loop.core import alloc as AL
    led = v.led
    ms, zz = int(opt(ctx, "alt_base_min_seeds")), float(opt(ctx, "alt_base_z"))
    anc = ancestry(led, v.incumbent)
    anc |= {n for n, nd in led.nodes.items() if nd.port_of in anc}              # a port of an ancestor is an ancestor
    lines = Lines(store, led)
    inc_lines = lines.cumulative(v.incumbent)
    epoch = led.epoch
    keep = 8                                                                   # picks examined per source, at most
    ok: dict[str, tuple[float, float, int]] = {}
    for nid in led.nodes:
        if nid in v.beliefs and not _skip_node(led, nid) and (got := standing(v, nid, ms, zz)) is not None:
            ok[nid] = got
    out: dict[str, list[Pick]] = {s: [] for s in SOURCES}

    def fit(nid: str) -> tuple[float | None, float | None, tuple[float, float] | None] | None:
        if not sound(store, statuses, nid, ctx) or not buildable(ctx, store, led, nid):
            return None
        return diverse(ctx, store, v, nid, anc, inc_lines, lines)

    def extra(d: tuple[float | None, float | None, tuple[float, float] | None]) -> dict[str, Any]:
        return {"dist": None if d[0] is None else round(d[0], 4), "contain": None if d[1] is None else round(d[1], 4),
                "gain": None if d[2] is None else d[2][0], "gain_se": None if d[2] is None else d[2][1]}

    # (a) the best DIVERSE node of a non-saturated cluster outside the incumbent's lineage
    passed = [j.node for j in v.judged if str(getattr(j.verdict.kind, "value", j.verdict.kind)) == "pass"]
    cls = AL.clusters(ctx, store, led, {n: (b.m, b.n) for n, b in v.beliefs.items()},
                      protect=AL.protected(led, v.incumbent, passed))
    mine = {c.id for c in cls if anc & set(c.members)}
    for c in cls:
        if c.id in mine or c.flag == "saturated-negative":
            continue
        for best in sorted((m for m in c.members if m in ok and led.nodes[m].base == epoch and m not in anc),
                           key=lambda m: (-ok[m][0], m))[:5]:
            if (d := fit(best)) is None:
                continue
            m, sd, n = ok[best]
            out["cluster"].append(Pick(
                best, "cluster", m, sd, n, f"the best node of cluster {c.id} ({c.category}; {c.label}), which is not "
                f"saturated and not in the incumbent's lineage (clusters {', '.join(sorted(mine)) or '-'})", c.id,
                **extra(d)))
            break
    out["cluster"].sort(key=lambda p: (-p.m, p.node))
    out["cluster"] = out["cluster"][:keep]

    # (b) the previous epoch's best, in its port onto the current base
    eps = [sha for _, _, sha in sorted(led.bases)]
    if len(eps) >= 2 and eps[-1] == epoch:
        prev = eps[-2]
        olds = sorted((n for n in led.nodes if led.nodes[n].base == prev and n in ok and sound(store, statuses, n, ctx)),
                      key=lambda n: (-ok[n][0], n))
        for old in olds:
            port = led.port(old)
            if port is None or port == old or port == v.incumbent or port in anc or old in anc or port not in v.beliefs:
                continue                              # not ported (yet) / part of the incumbent: skipped
            if (got := standing(v, port, ms, zz)) is None or (d := fit(port)) is None:
                continue
            out["prev_epoch"].append(Pick(
                port, "prev_epoch", got[0], got[1], got[2],
                f"the previous epoch's best node {old} (own gain {ok[old][0]:+.4f}), ported onto the current base as "
                f"{port}", via=old, rank=ok[old][0], **extra(d)))
            if len(out["prev_epoch"]) >= keep:
                break
        out["prev_epoch"].sort(key=lambda p: (-(p.m if p.rank is None else p.rank), p.node))

    # (c) the top posterior node of the current epoch outside the incumbent's ancestry
    for nid, (m, sd, n) in sorted(ok.items(), key=lambda kv: (-kv[1][0], kv[0]))[:80]:
        node = led.nodes[nid]
        if node.base != epoch or nid in anc or nid == v.incumbent or node.port_of:
            continue
        if (d := fit(nid)) is None:
            continue
        out["top_outside"].append(Pick(nid, "top_outside", m, sd, n,
                                       "the top posterior node of this epoch outside the incumbent's ancestry",
                                       **extra(d)))
        if len(out["top_outside"]) >= keep:
            break
    out["top_outside"].sort(key=lambda p: (-p.m, p.node))
    if (rp := root_pick(ctx, store, v, statuses)) is not None:
        out[ROOT] = [rp]
    return out


def root_pick(ctx: Any, store: Any, v: Any, statuses: Mapping[str, Any] | None = None) -> Pick | None:
    """The current epoch's root base node as an alt base: exempt from the parent / ancestry / containment rules, but
    sound (commit, not void / failed) and buildable (commit object + checkpoint when ``alt_base_require_ckpt``)."""
    if not bool(opt(ctx, "alt_base_root")):
        return None
    led = v.led
    rid = led.base_root
    node = led.nodes.get(rid)
    if node is None or rid == v.incumbent or node.parent is not None or node.base != led.epoch:
        return None
    if not sound(store, statuses, rid, ctx) or not buildable(ctx, store, led, rid):
        return None
    seeds = len({r.seed for r in led.runs if r.node == rid and r.status == "done"})
    return Pick(rid, ROOT, 0.0, 0.0, seeds, "the current epoch's root base node: the plain base without the incumbent's "
                "stack, far from the incumbent by construction; its own runs are the paired control", dist=1.0)


def root_due(turn: int, share: float) -> bool:
    """Does the ``turn``-th alt draw use root while other sources have a pick? (deterministic hash, as :func:`due`)"""
    if share <= 0:
        return False
    if share >= 1:
        return True
    return int(hashlib.sha1(f"alt-root:{turn}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def choose(ctx: Any, store: Any, v: Any, statuses: Mapping[str, Any] | None = None, turn: int = 0,
           exclude: Sequence[str] = (), cands: Mapping[str, Sequence[Pick]] | None = None) -> Pick | None:
    """The base of the ``turn``-th alt session: the sources take turns (those with an eligible pick), the best
    pick of the source whose base was used fewer than ``alt_base_max_uses`` times; None when there is none. Root (when
    eligible) takes ``alt_base_root_share`` of the draws, and all of them when no other source has a pick."""
    cands = candidates(ctx, store, v, statuses) if cands is None else cands
    uses = state(store)["uses"]
    cap = int(opt(ctx, "alt_base_max_uses"))
    live = [s for s in SOURCES if any(p.node not in exclude and int(uses.get(p.node, 0)) < cap for p in cands.get(s, ()))]
    root = next((p for p in cands.get(ROOT, ()) if p.node not in exclude), None)    # root is never "used up"
    if root is not None and (not live or root_due(turn, float(opt(ctx, "alt_base_root_share")))):
        return root
    for i in range(len(live)):
        src = live[(turn + i) % len(live)]
        for p in cands[src]:
            if p.node not in exclude and int(uses.get(p.node, 0)) < cap:
                return p
    return None


def fnf_branches(rnd: Mapping[str, Any], k: int) -> list[str]:
    """The fork-and-flush branches of an open round that start from an alt base: the LAST ``k`` of its branches."""
    br = list(rnd.get("branches") or ())
    return br[len(br) - max(0, min(int(k), len(br))):] if k > 0 else []


# ---------------------------------------------------------------------------------------------- the brief
def incumbent_mechanisms(led: Any, store: Any, incumbent: str, top: int = 14) -> list[str]:
    """The incumbent's stack as context: one line per non-root node of its ancestry (family/mechanism, oldest first)."""
    out = []
    for nid in sorted(ancestry(led, incumbent) - {led.base_root}, key=lambda n: (led.nodes[n].created_ts, n)
                      if n in led.nodes else (0.0, n)):
        sn = store.get_node(nid)
        if sn is not None and nid in led.nodes and led.nodes[nid].parent is not None and not led.nodes[nid].port_of:
            out.append(f"- {nid}: {getattr(sn, 'family', '?')}/{getattr(sn, 'mechanism_tag', '?')}")
    return out[-top:]


def retarget(head: str, node: str) -> str:
    """The generic kind text of an alt-base session: "on / in the incumbent" -> "on / in your starting point (<node>)"
    (the session builds on ``node``, not the incumbent)."""
    sp = f"your starting point ({node})"
    for a, b in (("on the incumbent's", f"on {sp[:-1]}'s)"), ("in the incumbent's", f"in {sp[:-1]}'s)"),
                 ("on the incumbent", f"on {sp}"), ("in the incumbent", f"in {sp}")):
        head = head.replace(a, b)
    return head


def brief_note(led: Any, store: Any, p: Pick, incumbent: str, tag: str = "") -> str:
    """:func:`note`, with the incumbent's mechanisms as context for a root-based session."""
    return note(p, incumbent, tag, incumbent_mechanisms(led, store, incumbent) if p.source == ROOT else ())


def note_root(p: Pick, incumbent: str, mechs: Sequence[str] = ()) -> str:
    return (f"STARTING POINT (measured arm `alt_base`, source `root`): this session builds on the PLAIN BASE **{p.node}** "
            f"(the current epoch's root: the base code without any of the incumbent's stack), NOT on the incumbent "
            f"{incumbent}. Your worktree starts from {p.node}'s code. Your node is judged FIRST as its own paired "
            f"increment over {p.node} (root's own runs are the control); if it works there it is ported onto "
            f"{incumbent} afterwards and tested where it matters (its gain over {incumbent} still decides the bar). "
            f"The incumbent's mechanisms, shown as CONTEXT so you do not just re-invent them:\n"
            + ("\n".join(mechs) if mechs else "- (none recorded)") + "\n"
            f"Do NOT re-add a mechanism the incumbent already has, unless it is a deliberately different variant of it; "
            f"look for what the incumbent's stack does not do.")


def note(p: Pick, incumbent: str, tag: str = "", mechs: Sequence[str] = ()) -> str:
    if p.source == ROOT:
        return note_root(p, incumbent, mechs)
    return (f"STARTING POINT (measured arm `alt_base`): this session builds on **{p.node}**"
            + (f" ({tag})" if tag else "") + f", NOT on the incumbent {incumbent}: {p.why}. Its own gain over its "
            f"parent is {p.m:+.4f} (sd {p.sd:.4f}) over {p.n} paired seeds. Your worktree starts from {p.node}'s code. "
            + (f"Its cumulative code differs from the incumbent's by a diff-line Jaccard distance of {p.dist:.2f}"
               + (f" (only {p.contain:.0%} of its own added lines are in the incumbent)" if p.contain is not None else "")
               + ". " if p.dist is not None else "") +
            f"Your node is credited with its OWN paired increment over {p.node} (never {p.node}'s gain), and its gain "
            f"over the incumbent still decides the bar -- a change that only restates what {p.node} already has earns "
            "nothing. Look for what this lineage does differently from the incumbent and build the mechanism that "
            "complements it.")


# ---------------------------------------------------------------------------------------------- root -> incumbent ports
def tags(node: Any) -> dict[str, str]:
    return dict(x.split("=", 1) for x in (getattr(node, "notes", "") or "").split() if "=" in x)


def open_root_ports(led: Any, statuses: Mapping[str, Any], idea_runs: int) -> int:
    """Auto-ports (``port_src=alt_root``) of the current epoch still being screened."""
    fin: dict[str, int] = {}
    for r in led.runs:
        if r.status in ("done", "crash"):
            fin[r.node] = fin.get(r.node, 0) + 1
    return sum(1 for n in led.nodes.values() if PORT_TAG in (n.notes or "").split() and n.base == led.epoch
               and _derived(statuses, n.id) in ("QUEUED", "SCREENING") and fin.get(n.id, 0) < idea_runs)


def port_due(ctx: Any, store: Any, led: Any, post: Any, statuses: Mapping[str, Any] | None, incumbent: str,
             idea_runs: int = 3) -> list[str]:
    """Root children to auto-port onto the incumbent: alt_src=root nodes of this epoch whose own increment over root is
    positive on >= ``alt_base_min_seeds`` paired seeds (shrunk m > 0 and m > 1 se), sound, not yet ported (a REJECTED
    port is not retried); best first, at most ONE open port of this kind (``port_src=alt_root``) at a time."""
    if not bool(opt(ctx, "alt_base_root")) or not incumbent or incumbent not in led.nodes:
        return []
    room = 1 - open_root_ports(led, statuses, idea_runs)
    if room <= 0:
        return []
    done = {n.port_of for n in led.nodes.values() if n.port_of and PORT_TAG in (n.notes or "").split()}
    ms = int(opt(ctx, "alt_base_min_seeds"))
    kids = [nid for nid, n in led.nodes.items()
            if n.base == led.epoch and nid not in done and nid != incumbent and not n.port_of and n.parent == led.base_root
            and tags(n).get("alt_src") == ROOT and "arm=alt_base" in (n.notes or "").split()]
    if not kids or post is None:
        return []
    beliefs = post.beliefs()
    got = []
    for nid in kids:
        if nid not in beliefs or not sound(store, statuses, nid, ctx):
            continue
        m, var, _ = beliefs[nid]
        est = (getattr(post, "est", {}).get(nid) or {}).get(M.COMPOSITE)
        paired = int(est.paired) if est is not None else int(beliefs[nid][2])
        if paired >= ms and m > 0 and m > math.sqrt(max(0.0, var)):
            got.append((m, nid))
    return [nid for _, nid in sorted(got, key=lambda x: (-x[0], x[1]))[:room]]


# ---------------------------------------------------------------------------------------------- re-root ports
REROOT_TAG = "port_src=reroot"              # an auto-port of an old incumbent's child onto the NEW incumbent
K_REROOT = "reroot.ports"                   # {new incumbent: {"from", "ts", "ported": {old: port}, "skipped": {old: why}}}


@dataclass(frozen=True)
class RerootPick:
    node: str           # the old-incumbent descendant whose change (over the OLD incumbent) is ported
    lineage: str        # its ancestor that is a child of the old incumbent (one port per lineage)
    m: float            # EB-shrunk paired composite gain over the OLD incumbent (core.bar.paired)
    sd: float
    n: int              # paired seeds with the old incumbent
    p: float            # P(gain > 0), Gaussian on (m, sd)


def previous_incumbent(led: Any, inc: str) -> str:
    """The carrier ``inc`` replaced: the latest carrier before ``inc``'s last carrier entry that is not ``inc`` (mapped
    to its counterpart on the current epoch); '' when none (``inc`` was never a carrier switch)."""
    cs = sorted(getattr(led, "carriers", ()) or (), key=lambda x: x[0])
    idx = [i for i, (_, n) in enumerate(cs) if n == inc]
    if not idx:
        return ""
    for _, n in reversed(cs[:idx[-1]]):
        if n != inc:
            old = led.port(n) if hasattr(led, "port") else n
            return old or ""
    return ""


def _chain(led: Any, nid: str) -> list[str]:
    out, seen = [], set()
    while nid and nid not in seen and nid in led.nodes:
        seen.add(nid)
        out.append(nid)
        nid = led.nodes[nid].parent
    return out


def reroot_plan(led: Any, store: Any, statuses: Mapping[str, Any] | None, inc: str, old: str, *, p_min: float,
                min_seeds: int, eb_min_nodes: int, same_mech: Any = None, ctx: Any = None,
                crossable: dict[str, RerootPick] | None = None) -> tuple[list[RerootPick], dict[str, str]]:
    """The old incumbent's descendants worth re-measuring ON the new incumbent ``inc`` (operator 11-10: a re-root left
    a node +.0033 on 5/5 seeds stranded on the old root): every current-epoch descendant of ``old`` outside ``inc``'s
    ancestry / subtree, not a port / crossover / wave point, sound, whose EB-shrunk paired composite gain over ``old``
    (``core.bar.paired`` on ``old``: robust Paule-Mandel prior, chained through its ancestors) is > 0 on >=
    ``min_seeds`` paired seeds with P(gain > 0) >= ``p_min``; the best per LINEAGE (the child of ``old`` it descends
    from: a tuning ladder's points are one mechanism), best gain first. ``same_mech(nid) -> why | ''`` skips a change
    redundant with ``inc``'s own (e.g. both sampling). Returns (picks, {node: why skipped}) -- skips only of nodes that
    otherwise qualify. ``crossable`` (a dict, filled): each skipped node's evidence as a :class:`RerootPick` -- it
    cleared the port bar, so the tick may cross it with ``inc`` instead (:func:`true_same` aside)."""
    from loop.core import bar as BR
    if not inc or not old or inc == old or inc not in led.nodes or old not in led.nodes:
        return [], {}
    pv = BR.paired(led, old, eb_min_nodes=eb_min_nodes)
    if pv is None or pv.prior is None:
        return [], {}                                    # no EB prior: a flat prior would take a best-of-many at face
    inc_line = set(_chain(led, inc))
    best: dict[str, RerootPick] = {}
    skipped: dict[str, str] = {}
    for nid, node in led.nodes.items():
        if nid in inc_line or nid == old or nid not in pv.gains or node.base != led.epoch:
            continue
        ch = _chain(led, nid)
        if old not in ch or inc in ch:                   # not built on the old incumbent / built on the new one
            continue
        if node.port_of or node.second_parent or "pair_of=" in (node.notes or "") or BR._point(led, nid):
            continue
        m, v, k = pv.gains[nid]
        sd = math.sqrt(max(v, 0.0))
        p = (1.0 if m > 0 else 0.0) if sd <= 0 else 0.5 * (1.0 + math.erf(m / (sd * math.sqrt(2.0))))
        if k < min_seeds or not m > 0 or p < p_min or not sound(store, statuses, nid, ctx):
            continue
        lin = ch[ch.index(old) - 1]
        if lin in inc_line:            # shares an ancestor with inc below old: its gain over old double-counts it
            continue
        pick = RerootPick(nid, lin, m, sd, int(k), p)
        if same_mech is not None and (why := same_mech(nid)):
            skipped[nid] = why
            if crossable is not None:
                crossable[nid] = pick
            continue
        if lin not in best or (m, p) > (best[lin].m, best[lin].p):
            best[lin] = pick
    return sorted(best.values(), key=lambda x: (-x.m, x.node)), skipped


REROOT_CROSS_TAG = "reroot_cross_of"      # a crossover (new incumbent x a skipped re-root port candidate): its notes


def stem_same(a: str, b: str) -> bool:
    """Two mechanism tags name the SAME mechanism: equal, or one is the other plus a tuning suffix (``x`` vs
    ``x_muon_lr_mult_hi``: a ladder point of ``x``)."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    return bool(a and b) and (a == b or a.startswith(b + "_") or b.startswith(a + "_"))


def true_same(led: Any, store: Any, inc: str, nid: str) -> str:
    """Why ``nid`` really IS ``inc``'s mechanism ('' = it is not): an identical mechanism-tag stem (:func:`stem_same`,
    either way along ``nid``'s chain below the old root: a ladder point of a node carrying it), or ``nid`` is an
    ancestor / descendant of ``inc``. A family + shared-file overlap alone is NOT the same mechanism (11-10: two
    different discriminative aux heads, both in model.py, were skipped as one): such a candidate is crossed instead."""
    if nid == inc or nid in _chain(led, inc) or inc in _chain(led, nid):
        return f"{nid} is an ancestor / descendant of {inc}"
    si = store.get_node(inc) if store is not None else None
    ti = getattr(si, "mechanism_tag", "") if si is not None else ""
    cur, seen = nid, set()
    while cur and cur not in seen and cur in led.nodes:   # the node, then up its tuning ladder to the ladder's base
        seen.add(cur)
        sx = store.get_node(cur) if store is not None else None
        tx = getattr(sx, "mechanism_tag", "") if sx is not None else ""
        if ti and stem_same(ti, tx):
            return f"identical mechanism stem ({tx} ~ {ti})"
        if "ladder=" not in (getattr(sx, "notes", "") or led.nodes[cur].notes or ""):
            break
        cur = led.nodes[cur].parent
    return ""


# ---------------------------------------------------------------------------------------------- the meta report
def arm_of(node: Any) -> str:
    return ARM if TAG in (getattr(node, "notes", "") or "").split() else ""


def dist_of(node: Any) -> float | None:
    """The diff-line distance of the node's alt base from the incumbent, as recorded when it was made (``alt_dist=``)."""
    t = dict(x.split("=", 1) for x in (getattr(node, "notes", "") or "").split() if "=" in x).get("alt_dist", "")
    if t == "far":
        return 1.0
    try:
        return float(t)
    except ValueError:
        return None


def bucket(node: Any) -> str:
    d = dist_of(node)
    if d is None:
        return "unmeasured"
    return next((n for n, lo, hi in BUCKETS if lo <= d < hi), "near")


def section(ctx: Any, led: Any, store: Any, v: Any) -> dict[str, Any]:
    """The alt-base arm vs the incumbent-based sessions: credit per GPU-h on fresh-seed test gains (``ideation.credit``,
    as the fork-and-flush A/B) and with the shrunk screen posterior of untested nodes, the node counts, the mean own
    increment, the gain over the incumbent and the sources used. The control = the research-session idea nodes of the
    arm's kinds not tagged alt_base."""
    from loop.core import fnf as FN
    from loop.core.acquire import FRESH_SHRINK
    from loop.core.alloc import brief_kind
    from loop.core.ideation import credit
    kinds = tuple(str(k) for k in (opt(ctx, "alt_base_kinds") or ()))

    def key(node: Any) -> str:
        if not FN.idea_node(node) or not (bk := brief_kind(node.notes)):
            return ""
        if arm_of(node):
            return ROOT if tags(node).get("alt_src") == ROOT else "alt_base"      # root is reported separately
        return "incumbent" if bk in kinds and "branch=" not in (node.notes or "") else ""
    out: dict[str, Any] = {}
    plain = credit(v, key=key)
    screen = credit(v, key=key, screen=(FRESH_SHRINK, 2))
    for arm in ("alt_base", ROOT, "incumbent"):
        c, s = plain.get(arm), screen.get(arm)
        ids = [n for n, nd in led.nodes.items() if key(nd) == arm]
        ms = [v.beliefs[n].m for n in ids if n in v.beliefs and v.beliefs[n].n >= 2]
        out[arm] = {"nodes": len(ids), "measured": len(ms), "positive": sum(x > 0 for x in ms),
                    "mean_m": (sum(ms) / len(ms)) if ms else None,
                    "tested": int(c["tested"]) if c else 0, "gpu_h": round(c["gpu_h"], 1) if c else 0.0,
                    "credit_per_gpu_h": c["per_gpu_h"] if c else 0.0,
                    "screen_credit_per_gpu_h": s["per_gpu_h"] if s else 0.0}
    def kb(node: Any) -> str:
        k = key(node)
        return f"alt_base:{bucket(node)}" if k == "alt_base" else ""
    pb, sb = credit(v, key=kb), credit(v, key=kb, screen=(FRESH_SHRINK, 2))
    by_dist: dict[str, Any] = {}
    for b in [n for n, _, _ in BUCKETS] + ["unmeasured"]:
        k = f"alt_base:{b}"
        ids = [n for n, nd in led.nodes.items() if kb(nd) == k]
        if not ids:
            continue
        c, s_ = pb.get(k), sb.get(k)
        ms = [v.beliefs[n].m for n in ids if n in v.beliefs and v.beliefs[n].n >= 2]
        by_dist[b] = {"nodes": len(ids), "measured": len(ms), "mean_m": (sum(ms) / len(ms)) if ms else None,
                      "tested": int(c["tested"]) if c else 0, "gpu_h": round(c["gpu_h"], 1) if c else 0.0,
                      "credit_per_gpu_h": c["per_gpu_h"] if c else 0.0,
                      "screen_credit_per_gpu_h": s_["per_gpu_h"] if s_ else 0.0}
    out["by_distance"] = by_dist
    src: dict[str, int] = {}
    for nd in led.nodes.values():
        if arm_of(nd):
            t = dict(x.split("=", 1) for x in (nd.notes or "").split() if "=" in x).get("alt_src", "?")
            src[t] = src.get(t, 0) + 1
    out["sources"] = src
    return out


def report_text(a: Mapping[str, Any]) -> list[str]:
    if not a or not ((a.get("alt_base") or {}).get("nodes") or (a.get(ROOT) or {}).get("nodes")):
        return ["(no alt-base node yet)"]
    f = lambda x: "-" if x is None else f"{x:+.4f}"     # noqa: E731
    lines = []
    for arm in ("alt_base", ROOT, "incumbent"):
        if arm == ROOT and not (a.get(ROOT) or {}).get("nodes"):
            continue
        x = a.get(arm) or a["incumbent"]
        lines.append(f"{arm}: {x['nodes']} nodes ({x['measured']} measured, {x['positive']} positive, mean own gain "
                     f"{f(x['mean_m'])}), {x['tested']} tested, {x['gpu_h']} GPU-h; credit/GPU-h {x['credit_per_gpu_h']:.2e} "
                     f"(with screen posterior {x['screen_credit_per_gpu_h']:.2e})")
    d = a["alt_base"]["credit_per_gpu_h"] - a["incumbent"]["credit_per_gpu_h"]
    ds = a["alt_base"]["screen_credit_per_gpu_h"] - a["incumbent"]["screen_credit_per_gpu_h"]
    for b, x in (a.get("by_distance") or {}).items():
        lines.append(f"  distance {b}: {x['nodes']} nodes ({x['measured']} measured, mean own gain {f(x['mean_m'])}), "
                     f"{x['tested']} tested, {x['gpu_h']} GPU-h; credit/GPU-h {x['credit_per_gpu_h']:.2e} "
                     f"(with screen posterior {x['screen_credit_per_gpu_h']:.2e})")
    lines.append(f"arm - control credit/GPU-h: {d:+.2e} (tests), {ds:+.2e} (with screen posterior); sources {a.get('sources') or {}}")
    return lines
