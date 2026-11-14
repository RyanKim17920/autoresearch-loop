"""loop.redundancy — duplicate LLM work is skipped, duplicate LLM output is merged (no LLM call here).

Measured on the 10-06 DB (MiniLM cosine >= 0.8 to an EARLIER output of the same source): cadence reviews 65% of
summaries, playbook versions 90% (18 of 20 a near-copy of the previous one), review meta-hypotheses 30%,
postmortems 34%, researcher claims 18%, operator proposals 6% at 0.8 -- but 37 open proposals were only 16
topics (cos >= 0.5), one ("gate the screen on the OWN increment vs the measure parent") proposed 7 times.

* INPUT guard: an LLM job whose evidence (:func:`evidence_key`: closed / confirmed nodes, incumbent, measurement
  and verdict epochs) is unchanged since the last SUCCESSFUL job of its role is skipped (``llm_job_skipped``):
  reviews, the playbook consolidation, the research scout.
* OUTPUT merge, on the loop's own hashing embedder (loop.select.embed; calibrated against MiniLM on the live
  texts): a review proposal restating an earlier one (:data:`PROPOSAL_COS` on its change text) is dropped and
  counted on the earlier one; a meta-hypothesis restating a live one (:data:`META_COS`) is merged into it (its
  probes re-pointed); a playbook whose rules all restate the current ones is no new version."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from loop.contracts import CLOSED_STATUSES, K_EPOCH, HypothesisKind, NodeStatus
from loop.select.embed import HashingEmbedder, cosine

K_INPUTS = "redundancy.inputs"        # {role: {"key", "ts"}}: the evidence the last successful job of a role saw
K_REPEATS = "redundancy.proposal_repeats"   # {proposal id: times a later review restated it}
#: a proposal's CHANGE text (before ' -- '): on the 48 live proposals every pair at >= 0.35 restates the same change
PROPOSAL_COS = 0.35
#: a meta-hypothesis claim: 14 of the 17 live pairs at >= 0.6 are MiniLM >= 0.8 near-duplicates
META_COS = 0.6
#: a playbook rule restating a current one
RULE_COS = 0.6
_EMB = HashingEmbedder()


def evidence_key(store: Any, snapshot: Any = None) -> str:
    """What an LLM review / consolidation / scout run sees change: closes, confirmations, the incumbent, epochs."""
    nodes = store.nodes()
    closed = sum(n.status in CLOSED_STATUSES and n.status is not NodeStatus.REJECTED for n in nodes)   # verdicts
    conf = sum(n.status is NodeStatus.CONFIRMED for n in nodes)
    inc = getattr(snapshot, "incumbent_id", "") or str(store.get_state("world.incumbent") or "")
    epoch = int((store.get_state(K_EPOCH) or {}).get("epoch", 0))
    verdict = len(store.get_state("backfill.seen") or {})
    return f"closed={closed} confirmed={conf} incumbent={inc} epoch={epoch} verdict_epochs={verdict}"


def same_input(store: Any, role: str, key: str) -> bool:
    return (store.get_state(K_INPUTS) or {}).get(role, {}).get("key") == key


def remember(store: Any, role: str, key: str, now: float) -> None:
    """The evidence a SUCCESSFUL job of ``role`` saw (a failed job is retried on the same evidence)."""
    store.set_state(K_INPUTS, {**(store.get_state(K_INPUTS) or {}), role: {"key": key, "ts": now}})


def skipped(store: Any, role: str, key: str, why: str = "no new evidence since the last one") -> str:
    store.log("llm_job_skipped", role=role, evidence=key, reason=why)
    return f"{role} skipped: {why} ({key})"


def _emb(text: str) -> tuple[float, ...]:
    return _EMB.embed(text)


def nearest(text: str, others: Iterable[tuple[str, str]]) -> tuple[str, float]:
    """(id, cosine) of the closest of ``others`` [(id, text)] to ``text`` ('', 0.0 when none)."""
    v, best = _emb(text), ("", 0.0)
    for oid, o in others:
        if (c := cosine(v, _emb(o))) > best[1]:
            best = (oid, c)
    return best


def change_text(proposal: str) -> str:
    return proposal.split(" -- ", 1)[0]


def dedupe_proposals(store: Any, proposals: Sequence[str]) -> tuple[list[str], list[tuple[str, str, float]]]:
    """(kept proposals, [(dropped text, earlier proposal id, cos)]): a proposal restating ANY earlier one (open or
    acked: the operator has seen it) or one kept earlier in the same review is dropped; the earlier one's repeat
    count goes up (``loop status`` shows it)."""
    earlier = [(f"{rv.id}#{i}", change_text(p)) for rv in store.reviews() for i, p in enumerate(rv.proposals)]
    kept, dropped, rep = [], [], dict(store.get_state(K_REPEATS) or {})
    for p in proposals:
        oid, c = nearest(change_text(p), [*earlier, *((f"new#{k}", change_text(q)) for k, q in enumerate(kept))])
        if c >= PROPOSAL_COS:
            dropped.append((p, oid, round(c, 3)))
            if not oid.startswith("new#"):
                rep[oid] = int(rep.get(oid, 0)) + 1
        else:
            kept.append(p)
    if dropped:
        store.set_state(K_REPEATS, rep)
    return kept, dropped


def merge_metas(store: Any, metas: Sequence[Any], live: Sequence[Any]) -> tuple[list[Any], dict[str, str]]:
    """(metas to store, {dropped meta id: the live meta it restates}): a review meta-hypothesis whose claim restates
    a live (not stale) META hypothesis is not stored again; its probes go to the live one."""
    keep, into = [], {}
    pool = [(h.id, h.claim) for h in live]
    for h in metas:
        oid, c = nearest(h.claim, pool) if h.kind is HypothesisKind.META else ("", 0.0)
        if c >= META_COS:
            into[h.id] = oid
        else:
            keep.append(h)
            pool.append((h.id, h.claim))
    return keep, into


def same_rules(old: Sequence[dict], new: Sequence[dict]) -> bool:
    """Every new rule restates a current one of the same component and vice versa (no new playbook version)."""
    if not old or len(old) != len(new):
        return False

    def covered(a: Sequence[dict], b: Sequence[dict]) -> bool:
        return all(nearest(x["rule"], [(str(i), y["rule"]) for i, y in enumerate(b)
                                       if y.get("component") == x.get("component")])[1] >= RULE_COS for x in a)
    return covered(new, old) and covered(old, new)
