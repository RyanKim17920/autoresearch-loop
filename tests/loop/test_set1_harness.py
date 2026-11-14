"""Set-1 harness improvements (2026-10-21): effective capacity, stale-pending handling, program-arm priority and
queue aging, the family-level negative block, program-shadows-explore, the parent-relative novelty metric."""

from __future__ import annotations

import dataclasses

import pytest

from loop import runner as R
from loop.contracts import JobHandle, JobStatus, RunStatus, Stage, VoidType
from loop.spec import PoolCfg, SlurmCfg

from .test_adapters import SlurmRunner, base_repo, make_np, np_ctx  # noqa: F401  (fixtures)
from .test_runner import _node, _pools


# ---------------------------------------------------------------- item 1: effective capacity
def _acct_ctx(np_ctx):
    pools = (PoolCfg("primary", slots=6, account="main", qos="normal"),
             PoolCfg("idle", slots=8, account="preempt", qos="low", preemptible=True))
    return dataclasses.replace(np_ctx, cfg=dataclasses.replace(np_ctx.cfg, slurm=SlurmCfg(pools=pools)))


def _cap_runner(limit_rows, squeue_rows):
    return SlurmRunner({
        "sacctmgr": lambda argv: (0, limit_rows.get(next(a for a in argv if a.startswith("account=")).split("=")[1], ""), ""),
        "squeue": (0, squeue_rows, ""),
    })


def test_pool_capacity_is_the_association_limit_minus_other_jobs_gpus(np_ctx, base_repo):   # noqa: F811
    ctx = _acct_ctx(np_ctx)
    adapter, runner = make_np(ctx, base_repo, runner=_cap_runner(
        {"main": "main|cpu=N,gres/gpu=8\n", "preempt": "preempt|\n"},
        "1 |sw-s4-a |cpu=12,node=1,gres/gpu=1\n2 |sw-s4-b |cpu=12,gres/gpu=1,gres/gpu:a100=1\n"
        "3 |par-proc |cpu=48,node=1\n4 |tpm |cpu=16,gres/gpu=4,gres/gpumem=9\n"
        f"5 |lp{'{tag}'}-n1-screen_s1_a0 |gres/gpu=1\n6 |diag-n7 |gres/gpu=1\n"))
    # re-render the loop's own job names with the adapter's real tag
    runner.replies["squeue"] = (0, runner.replies["squeue"][1].replace("{tag}", adapter._tag), "")
    cap = adapter.pool_capacity()
    assert cap == {"primary": {"limit": 8, "others": 6, "loop": 2, "effective": 2}}     # idle: no limit -> absent
    sq = [a for a in runner.argvs("squeue")][0]
    assert sq[:6] == ["squeue", "-h", "-u", sq[3], "-A", "main"]                     # our user, the pool's account
    assert not runner.argvs("scancel")                                                   # never touches a job


def test_pool_capacity_failure_means_no_cap(np_ctx, base_repo):   # noqa: F811
    adapter, _ = make_np(_acct_ctx(np_ctx), base_repo, runner=SlurmRunner({"sacctmgr": (1, "", "down")}))
    assert adapter.pool_capacity() == {}


class _Cap:
    def __init__(self, caps):
        self.caps, self.calls = caps, 0

    def pool_capacity(self):
        self.calls += 1
        if isinstance(self.caps, Exception):
            raise self.caps
        return self.caps

    def submit(self, *, run, commit):
        return JobHandle(run.id, "1", f"/o/{run.id}", 0.0)

    def poll(self, handle):
        return JobStatus(RunStatus.RUNNING, detail="RUNNING")

    def cancel(self, handle):
        pass


def test_limits_use_the_effective_capacity_and_it_is_read_once_per_tick(ctx, store):
    cap = {"primary": {"limit": 8, "others": 6, "loop": 0, "effective": 2}}
    adapter = _Cap(cap)
    book = R.RunBook(_pools(ctx, primary=6, idle=4), store, adapter)
    assert book.limits(gate=False) == {"primary": 2, "idle": 4}
    book.limits(), book.free(), book.headroom()
    assert adapter.calls == 1                                  # cached for the tick
    assert book.cap_note() == " cap=primary:2/8(other 6)"
    book.poll()                                                # a new tick re-reads it
    book.limits()
    assert adapter.calls == 2
    # others vanish: back to the declared slots, no note
    adapter.caps = {"primary": {"limit": 8, "others": 0, "loop": 2, "effective": 8}}
    book.poll()
    assert book.limits(gate=False)["primary"] == 6 and book.cap_note() == ""


def test_a_failing_probe_or_the_switch_off_keeps_the_declared_slots(ctx, store):
    book = R.RunBook(_pools(ctx, primary=6, idle=4), store, _Cap(RuntimeError("boom")))
    assert book.limits(gate=False) == {"primary": 6, "idle": 4}
    assert len(store.events("capacity_probe_failed")) == 1
    off = R.RunBook(_pools(ctx, primary=6, idle=4, effective_capacity=False), store,
                    _Cap({"primary": {"limit": 8, "others": 6, "loop": 0, "effective": 2}}))
    assert off.limits(gate=False)["primary"] == 6


def test_smokes_and_screens_spill_to_idle_when_primary_is_association_full(ctx, store):
    full = {"primary": {"limit": 8, "others": 8, "loop": 0, "effective": 0}}
    book = R.RunBook(_pools(ctx, primary=6, idle=4), store, _Cap(full))
    node = _node(store, "n1")
    book.poll()
    book.pool_states = {"primary": (0, 0), "idle": (1, 2)}      # idle holds 2 of ours PENDING: a blocked smoke home
    assert book.smoke_blocked("idle") is False                  # ... but primary would hold it at AssocGrpGRES
    assert book.smoke_pool() == "idle"
    assert book.pool(node, Stage.BASELINE, 100_000) == "idle"   # HIGH spills too
    roomy = R.RunBook(_pools(ctx, primary=6, idle=4), store, _Cap({}))
    roomy.pool_states = {"primary": (0, 0), "idle": (1, 2)}
    assert roomy.smoke_blocked("idle") is True                  # no association cap: unchanged behaviour


# ---------------------------------------------------------------- item 2: stale pending
class _Held(_Cap):
    """Every job PENDING for the scripted reason; scancels and repools recorded."""

    def __init__(self, reason="AssocGrpGRES"):
        super().__init__({})
        self.reason, self.cancelled, self.moved, self.jobs = reason, [], [], 0

    def submit(self, *, run, commit):
        self.jobs += 1
        return JobHandle(run.id, str(self.jobs), f"/o/{run.id}", 0.0)

    def poll(self, handle):
        return JobStatus(RunStatus.RUNNING, detail="PENDING")

    def pending_reason(self, handle):
        return self.reason

    def cancel(self, handle):
        self.cancelled.append(handle.job_id)

    def repool(self, handle, run):
        self.jobs += 1
        self.moved.append((run.id, run.pool))
        return JobHandle(run.id, str(self.jobs), handle.output_dir, 0.0)


def _stale_book(ctx, store, *, primary=2, idle=2, adapter=None, **slurm):
    t = [1000.0]
    adapter = adapter or _Held()
    book = R.RunBook(_pools(ctx, primary=primary, idle=idle, **slurm), store, adapter, clock=lambda: t[0])
    return book, adapter, t


def test_a_run_pending_past_the_limit_at_a_cap_moves_to_a_free_pool(ctx, store):
    book, adapter, t = _stale_book(ctx, store, stale_pending_min=90)
    node = _node(store, "n1")
    run = book.launch(node, Stage.BASELINE, 100_000)             # HIGH: primary
    assert run.pool == "primary"
    book.poll()
    t[0] += 60 * 89
    assert book.stale_pending() == []                            # 89 min: not yet
    t[0] += 120
    out = book.stale_pending()
    assert len(out) == 1 and "AssocGrpGRES" in out[0] and "primary -> " in out[0]
    assert adapter.moved == [("n1:baseline:100000", "idle")] and adapter.cancelled == ["1"]   # pending-only cancel, then the move
    moved = store.get_run("n1:baseline:100000")
    assert moved.pool == "idle" and moved.status is RunStatus.RUNNING and moved.attempt == run.attempt
    assert store.events("run_stale_pending")[-1]["action"] == "moved"
    book.poll()                                                  # the clock restarts at the new job
    assert book.stale_pending() == []


def test_with_no_pool_free_the_loops_own_job_is_cancelled_and_the_row_requeued(ctx, store):
    book, adapter, t = _stale_book(ctx, store, primary=1, idle=0 + 1, stale_pending_min=90)
    node = _node(store, "n1")
    a = book.launch(node, Stage.BASELINE, 100_000)               # primary
    b = book.launch(node, Stage.BASELINE, 100_001)               # idle (spill)
    assert (a.pool, b.pool) == ("primary", "idle")
    book.poll()
    t[0] += 60 * 100
    out = book.stale_pending()                                   # both pools full: the first has nowhere to go
    assert "scancelled, back in the queue" in out[0] and adapter.cancelled[0] == "1"   # by job id, the loop's own
    r = store.get_run("n1:baseline:100000")
    assert r.status is RunStatus.VOID and r.void_type is VoidType.INFRA and r.attempt == -1   # no retry burnt (the retry's +1 is the same attempt)
    assert "n1:baseline:100000" not in book.pending                                           # slot released
    # ... and the slot it freed takes the second stale run (a move, not a cancel)
    assert len(out) == 2 and adapter.moved == [("n1:baseline:100001", "primary")]


def test_priority_pending_test_stage_and_off_are_left_alone(ctx, store):
    book, adapter, t = _stale_book(ctx, store, stale_pending_min=90)
    node = _node(store, "n1")
    book.launch(node, Stage.BASELINE, 100_000)
    book.poll()
    t[0] += 60 * 500
    adapter.reason = "Priority"                                  # it will start: plain queueing
    assert book.stale_pending() == [] and not adapter.cancelled and not adapter.moved
    adapter.reason = "AssocGrpGRES"
    off, _, t2 = _stale_book(ctx, store, adapter=adapter, stale_pending_min=0)
    off.pending = {"n1:baseline:100000"}
    assert off.stale_pending() == []
    test_run = book.launch(node, Stage.CONFIRM, 100_100)         # a TEST run waits for primary on purpose
    book.poll()
    t[0] += 60 * 500
    ids = [l.split()[2] for l in book.stale_pending()]
    assert test_run.id not in ids


def test_the_nanopath_adapter_reports_the_pending_reason(np_ctx, base_repo):   # noqa: F811
    from loop.adapters.nanopath_v2 import NanopathV2Adapter  # noqa: F401
    adapter, runner = make_np(np_ctx, base_repo, runner=SlurmRunner(
        {"squeue": (0, "PENDING             |AssocGrpGRES         |\n", "")}))
    h = JobHandle("r", "55", str(adapter.runs / "n1" / "screen"), 0.0)
    assert adapter.pending_reason(h) == "AssocGrpGRES"
    runner.replies = {"squeue": (0, "RUNNING             |None                |\n", "")}
    assert adapter.pending_reason(h) == ""


# ---------------------------------------------------------------- item 3: program-arm priority, queue ageing
from types import SimpleNamespace  # noqa: E402

from loop.contracts import NodeStatus, ROOT_NODE_ID  # noqa: E402
from loop.core import acquire as A  # noqa: E402
from loop.core import data as D  # noqa: E402
from loop.core.status import Status  # noqa: E402

S2 = 0.003 ** 2


def _dn(nid, notes="", ts=0.0, port_of=""):
    return D.Node(nid, ROOT_NODE_ID, None, "explore", "f", "explore", ROOT_NODE_ID, ts, "queued", notes, "B", port_of)


def _dled(nodes, runs=()):
    root = D.Node(ROOT_NODE_ID, None, None, "baseline", "", "baseline", ROOT_NODE_ID, 0.0, "baseline", "", "B")
    return SimpleNamespace(nodes={ROOT_NODE_ID: root, **{n.id: n for n in nodes}}, runs=tuple(runs), obs=(), epoch="B")


def test_rank_places_program_arm_screens_first_up_to_the_share():
    good = [A.Option("screen", f"g{i}", "explore", 0.006, S2, S2 / 4, 2.0, 2) for i in range(6)]
    arms = [A.Option("screen", f"p{i}", "explore", -0.002, S2, S2 / 4, 2.0, 2) for i in range(4)]
    assert [a.node for a in A.rank(good + arms, 10, 0)] == ["g0", "g1", "g2", "g3", "g4"]   # the arms rank nowhere
    got = [a.node for a in A.rank(good + arms, 10, 0, priority={"p0", "p1", "p2", "p3"})]
    assert got == ["p0", "g0", "g1", "g2", "g3"]          # share .3 x 10 slots = 3 -> ONE 2-run arm action, first
    assert [a.node for a in A.rank(good + arms, 20, 0, priority={"p0", "p1", "p2", "p3"})][:3] == ["p0", "p1", "p2"]


def test_rank_priority_share_zero_is_off_and_arms_below_the_floor_still_screen():
    arms = [A.Option("screen", "p0", "explore", -0.02, 0.0, S2 / 4, 2.0, 2)]      # a value <= 0 screen: floors drop it
    other = [A.Option("screen", "g0", "explore", 0.006, S2, S2 / 4, 2.0, 2)]
    assert "p0" not in [a.node for a in A.rank(other + arms, 10, 0)]
    assert A.rank(other + arms, 10, 0, priority={"p0"})[0].node == "p0"
    old, A.PROGRAM_ARM_SHARE = A.PROGRAM_ARM_SHARE, 0.0
    try:
        assert "p0" not in [a.node for a in A.rank(other + arms, 10, 0, priority={"p0"})]
    finally:
        A.PROGRAM_ARM_SHARE = old


def test_program_nodes_are_the_queued_arm_tagged_ones():
    led = _dled([_dn("a", "brief=program op=x arm=A program_arm=1"), _dn("b", "program_arm=1"), _dn("c", "x")])
    st = {ROOT_NODE_ID: Status.INCUMBENT, "a": Status.QUEUED, "b": Status.SCREENING, "c": Status.QUEUED}
    assert A.program_nodes(led, st) == {"a"}


def test_aged_queued_are_the_never_run_old_non_program_nodes():
    now = 1_000_000.0
    h = 3600.0
    led = _dled([_dn("old", ts=now - 97 * h), _dn("young", ts=now - 95 * h), _dn("arm", "program_arm=1", now - 500 * h),
                 _dn("wave", "wave=w001 point=1", now - 500 * h), _dn("ported", ts=now - 500 * h, port_of="x"),
                 _dn("ran", ts=now - 500 * h)])
    st = {n: Status.QUEUED for n in led.nodes}
    st[ROOT_NODE_ID] = Status.INCUMBENT
    st["ran"] = Status.SCREENING
    assert A.aged_queued(led, st, now, 96.0) == ["old"]
    assert A.aged_queued(led, st, now, 0.0) == []                                    # off
    assert A.aged_queued(led, st, now, 96.0, touched={"old": now - h}) == []        # restored recently: a fresh window


def test_age_out_is_recoverable_and_reports_what_it_marked(ctx, store):
    from .test_runner import _node
    n = _node(store, "n1")
    store.update_node("n1", created_ts=1000.0)
    got = A.age_out(store, ["n1", "missing"], now=1000.0 + 100 * 3600)
    assert got == ["n1"]
    aged = store.get_node("n1")
    assert aged.status is NodeStatus.REJECTED and aged.reject_reason.startswith("dropped_backlog: aged_out_queued")
    assert "aged_out=1" in aged.notes and store.events("queued_aged_out")[0]["node_id"] == "n1"
    assert A.age_out(store, ["n1"], now=2e6) == []                                   # not QUEUED any more: untouched
    assert A.restore_aged(store) == ["n1"]
    back = store.get_node("n1")
    assert back.status is NodeStatus.QUEUED and back.reject_reason == "" and "aged_out" not in back.notes
    assert back.commit == n.commit                                                    # nothing was deleted


# ---------------------------------------------------------------- item 4: family-level negative block
from loop.core import alloc as AL  # noqa: E402


def _cluster(cid="C1", gains=None, protected=""):
    gains = gains if gains is not None else {"n10": -0.009, "n11": -0.006, "n12": -0.007, "n13": -0.004, "n14": -0.003}
    cl = AL.Cluster(cid, list(gains), {"position": 1.0}, label="position/head", category="objective")
    cl.n, cl.mean = len(gains), sum(gains.values()) / len(gains) if gains else 0.0
    cl.best, cl.gains, cl.protected = (max(gains.items(), key=lambda kv: kv[1]) if gains else None), dict(gains), protected
    cl.tags = {m: f"objective position_head_{i}" for i, m in enumerate(gains)}
    return cl


def test_negative_family_needs_n_mean_no_positive_and_no_protection(ctx):
    ok = _cluster()
    assert AL.negative_family(ctx, ok)
    assert not AL.negative_family(ctx, _cluster(gains={"a": -.009, "b": -.006, "c": -.007, "d": -.004}))     # n = 4
    assert not AL.negative_family(ctx, _cluster(gains={f"n{i}": -.002 for i in range(6)}))                   # mean -.002
    assert not AL.negative_family(ctx, _cluster(gains={"a": .02, "b": -.03, "c": -.03, "d": -.03, "e": -.03}))  # a +.02 win
    assert not AL.negative_family(ctx, _cluster(protected="n10: TEST pass"))
    assert AL.family_sigma(ctx) > 0


def _remember(ctx, cl=None, fams=None):
    AL.LAST.update(clusters=[cl] if cl else [], df=__import__("collections").Counter({"position": 1}), n=10,
                   window=[], families=fams or {})


def test_family_rule_blocks_a_negative_cluster_unless_the_claim_states_the_difference(ctx, monkeypatch):
    cl = _cluster()
    cl.neg_family = True
    _remember(ctx, cl)
    monkeypatch.setattr(AL, "assign", lambda text, cls=None, tau=0.3: cl)
    rule, fix = AL.family_rule(ctx, "objective", "xview_position_head", "a position pretext head")
    assert rule == "negative_family" and "n10" in fix and "differs_from_cluster" in fix and "5 measured" in fix
    thin = "[differs from cluster: n10: ]"                                             # a bare id never passes
    assert AL.family_rule(ctx, "objective", "t", "x", f"c {thin}")[0] == "negative_family"
    one = "[differs from cluster: n10: predicts cross view cell membership where it regressed raw crop offsets]"
    assert AL.family_rule(ctx, "objective", "t", "x", f"c {one}")[0] == "negative_family"     # one cited < 2
    two = ("[differs from cluster: n10: predicts cross view cell membership where it regressed raw crop offsets; "
           "n11: the head here is read only at eval and takes no gradient into the backbone]")
    assert AL.family_rule(ctx, "objective", "t", "x", f"c {two}") == ("", "")
    off = dataclasses.replace(ctx, cfg=dataclasses.replace(ctx.cfg, alloc=dataclasses.replace(
        ctx.cfg.alloc, family_block=False)))
    assert AL.family_rule(off, "objective", "t", "x") == ("", "")


def test_term_families_find_what_the_lexical_clusters_scatter(ctx):
    def sn(i, tag):
        return SimpleNamespace(id=f"n{i}", family="objective", mechanism_tag=tag)
    nodes = [sn(1, "rel_crop_position_head"), sn(2, "rasa_positional_disentangle"), sn(3, "patch_position_prediction"),
             sn(4, "crop_box_position_scale"), sn(5, "xview_position_cls"), sn(6, "koleo_reg"), sn(7, "ema_decay")]
    nodes += [SimpleNamespace(id=f"m{i}", family="optimization", mechanism_tag=f"schedule_{i}x") for i in range(60)]
    gains = {"n1": -.0055, "n2": -.0276, "n3": .0003, "n4": -.0095, "n5": -.006, "n6": .01, "n7": .0}
    fams = AL.term_families(ctx, nodes, {k: (v, 3) for k, v in gains.items()}, {})
    pos = next(f for s, f in fams.items() if s.startswith("positio"))
    assert pos.n == 5 and pos.neg_family and pos.mean < -0.004 and set(pos.tags) == {"n1", "n2", "n3", "n4", "n5"}
    assert not any(f.neg_family for s, f in fams.items() if not s.startswith("positio"))
    prot = AL.term_families(ctx, nodes, {k: (v, 3) for k, v in gains.items()}, {"n2": "TEST pass"})
    assert not next(f for s, f in prot.items() if s.startswith("positio")).neg_family
    # a stem on more than family_block_max_share of all research nodes names no family
    assert not [s for s in AL.term_families(ctx, nodes, {k: (v, 3) for k, v in gains.items()}, {}) if s.startswith("objec")]


def test_family_rule_applies_to_a_term_family_and_the_researcher_exempts_program_arms(ctx):
    from loop.agents.researcher import LLMResearcher, claim_text
    from loop.core.ideation import REQUEST_PREFIX
    nodes = [SimpleNamespace(id=f"n{i}", family="objective", mechanism_tag=f"head_position_v{i}") for i in range(6)]
    nodes += [SimpleNamespace(id=f"m{i}", family="optimization", mechanism_tag=f"schedule_{i}x") for i in range(60)]
    fams = AL.term_families(ctx, nodes, {f"n{i}": (-.01, 2) for i in range(6)}, {})
    _remember(ctx, None, fams)
    assert AL.family_rule(ctx, "objective", "xview_position_classification", "text")[0] == "negative_family"
    assert AL.family_rule(ctx, "objective", "sampler_flatten", "text") == ("", "")
    me = SimpleNamespace(ctx=ctx)
    plain = SimpleNamespace(id=f"{REQUEST_PREFIX}x1", brief_kind="explore", arm="")
    arm = SimpleNamespace(id=f"{REQUEST_PREFIX}x2", brief_kind="program", arm="program_arm:op:A1")
    port = SimpleNamespace(id=f"{REQUEST_PREFIX}x3", brief_kind="port", arm="")
    got = LLMResearcher._family_block(me, plain, "objective", "xview_position_classification", "c")
    assert got and got[0] == "negative_family"
    assert LLMResearcher._family_block(me, arm, "objective", "xview_position_classification", "c") is None
    assert LLMResearcher._family_block(me, port, "objective", "xview_position_classification", "c") is None
    # the finish field reaches intake as part of the claim
    text = claim_text({"claim": "c", "differs_from_cluster": "n1: x" * 3})
    assert "[differs from cluster: " in text and "[differs from:" not in text
    claim = ("c [differs from cluster: n1: a cross view cell classifier where this was a crop offset regressor; "
             "n2: this head is read out at eval only and takes no gradient into the backbone]")
    assert LLMResearcher._family_block(me, plain, "objective", "xview_position_classification", claim) is None
    AL.LAST.update(clusters=[], families={})


# ---------------------------------------------------------------- item 5: program shadows explore
from loop.core import program as PG  # noqa: E402

PROGRAM = {"id": "op-1", "arms": [
    {"id": "A1", "stage": "A", "lane": "full", "family": "objective", "keys": ["init_align"],
     "note": "frozen-init feature alignment at the last block", "components": [
         "init-align weight: knob {key: dino.init_align_weight, lo: 0.05, hi: 0.5}"]},
    {"id": "site_cap", "stage": "B", "lane": "full", "family": "data",
     "note": "site-level TCGA tissue source site cap on top of the patient cap; a sampler code change"},
    {"id": "stack", "stage": "C", "lane": "full", "stack": True, "after": ["A1", "site_cap"], "note": "stack"}]}


def _shadow_ctx(ctx, programs=(PROGRAM,), **alloc):
    return dataclasses.replace(ctx, cfg=dataclasses.replace(
        ctx.cfg, ideation=dataclasses.replace(ctx.cfg.ideation, operator_programs=tuple(programs)),
        alloc=dataclasses.replace(ctx.cfg.alloc, **alloc)))


def _arm_node(store, nid, aid, status=NodeStatus.QUEUED, pid="op-1"):
    n = _node(store, nid)
    store.update_node(nid, status=status, notes=f"brief=program op={pid} arm={aid} stage=A program_arm=1")
    return n


def test_shadow_rule_rejects_explore_near_an_in_flight_arm(ctx, store):
    c = _shadow_ctx(ctx)
    _arm_node(store, "n1", "A1")
    rule, fix = PG.shadow_rule(c, store, "objective", "dino_feature_anchor", "x", ["dino.init_align_weight"])
    assert rule == "shadowed_by_program" and "op-1 arm A1" in fix and "dino.init_align_weight" in fix   # config key
    rule, fix = PG.shadow_rule(c, store, "data", "site_tempered_sampler", "x", [])                      # tag stems
    assert rule == "shadowed_by_program" and "site_cap" in fix and "sampler" in fix and "site" in fix   # (arm not built yet)
    assert PG.shadow_rule(c, store, "objective", "loca_position_head", "x", ["dino.loca_weight"]) == ("", "")
    assert PG.shadow_rule(c, store, "data", "sampler_only", "x", []) == ("", "")             # ONE shared stem is no overlap


def test_shadow_rule_same_cluster_as_the_arms_node(ctx, store, monkeypatch):
    c = _shadow_ctx(ctx)
    _arm_node(store, "n1", "A1")
    cl = _cluster(gains={"n1": 0.001})
    monkeypatch.setattr(AL, "assign", lambda text, cls=None, tau=0.3: cl)
    rule, fix = PG.shadow_rule(c, store, "objective", "unrelated_words_here", "x", [])
    assert rule == "shadowed_by_program" and "same mechanism cluster as the arm's node n1" in fix


def test_a_decided_superseded_or_old_arm_and_the_stack_arm_no_longer_shadow(ctx, store):
    c = _shadow_ctx(ctx)
    _arm_node(store, "n1", "A1", status=NodeStatus.SCREENED_OUT)                 # decided on its own increment
    assert PG.shadow_rule(c, store, "objective", "dino_feature_anchor", "x", ["dino.init_align_weight"]) == ("", "")
    assert [a["arm"] for a in PG.inflight_arms(c, store)] == ["site_cap"]         # the stack arm never shadows
    sup = _shadow_ctx(ctx, programs=({**PROGRAM, "superseded_by": "op-2"},))
    assert PG.inflight_arms(sup, store) == []
    old = _shadow_ctx(ctx, shadow_max_h=1.0)
    store.set_state("ideation.operator_programs", {"op-1/site_cap": {"ts": 1.0}})
    assert PG.inflight_arms(old, store, now=1.0 + 3 * 3600) == []                 # open past shadow_max_h
    assert [a["arm"] for a in PG.inflight_arms(old, store, now=1.0 + 1800)] == ["site_cap"]
    off = _shadow_ctx(ctx, shadow=False)
    assert PG.shadow_rule(off, store, "data", "site_tempered_sampler", "x", []) == ("", "")


def test_the_researcher_applies_the_shadow_rule_to_explore_and_literature_only(ctx, store):
    from loop.agents.researcher import LLMResearcher
    from loop.core.ideation import REQUEST_PREFIX
    me = SimpleNamespace(ctx=_shadow_ctx(ctx))
    _arm_node(store, "n1", "A1")

    def req(kind, arm="", rid="x"):
        return SimpleNamespace(id=f"{REQUEST_PREFIX}{rid}", brief_kind=kind, arm=arm)
    args = ("data", "site_tempered_sampler", "c")
    for kind in ("explore", "literature"):
        got = LLMResearcher._shadow_block(me, req(kind), store, *args)
        assert got and got[0] == "shadowed_by_program"
    assert LLMResearcher._shadow_block(me, req("program", "program_arm:op-1:site_cap"), store, *args) is None   # the arm itself
    assert LLMResearcher._shadow_block(me, req("repair"), store, *args) is None
    assert LLMResearcher._shadow_block(me, req("port"), store, *args) is None
    assert LLMResearcher._shadow_block(me, req("explore"), store, "data", "something_else", "c") is None


def test_intake_rejects_a_shadowed_explore_and_keeps_an_arms_own_node(ctx, store):
    from loop.core.ideation import Ideation
    ide = SimpleNamespace(ctx=_shadow_ctx(ctx), store=store, clock=lambda: 1.0)
    _arm_node(store, "n1", "A1")
    cand = SimpleNamespace(hypothesis=SimpleNamespace(family="objective", mechanism_tag="frozen_alignment_probe", claim="c"),
                           config_diff={"dino.init_align_weight": 0.2})
    got = Ideation._shadow_rule(ide, cand)
    assert got[0] == "shadowed_by_program"
    assert Ideation._shadow_rule(ide, None) == ("", "")


# ---------------------------------------------------------------- item 6: parent-relative novelty
from loop.core import bold as BOLD  # noqa: E402

from .test_v3_review2 import _code  # noqa: E402

AUX = ["class AuxHead(nn.Module):", "    def __init__(self, d):", "        super().__init__()",
       "        self.p = nn.Linear(d, d)", "", "    def forward(self, z):", "        return self.p(z)", "",
       "def aux_loss(a, b):", "    return (a - b).pow(2).mean()", "", "    total = total + 0.5 * aux_loss(z, t)"]


def test_novelty_is_zero_for_knob_only_and_keeps_knob_only_semantics():
    knob = _code(["    w = 0.5"], ["    w = 0.7", "    _ = 0"])
    assert BOLD.knob_only(knob) and BOLD.novelty(knob) == 0.0 and BOLD.novelty_parts(knob)["knob_only"]
    junk = _code(["    w = 0.5"], ["    w = 0.7", "    new_loss_weight = 0.3"])      # a new loss-ish NAME, no structure
    assert BOLD.knob_only(junk) and BOLD.novelty(junk) == 0.0
    assert BOLD.novelty("") == 0.0
    for d in (knob, junk, _code([], AUX), _code([], ["    h = nn.Linear(4, 4)(z)"])):    # knob_only <=> novelty == 0
        assert BOLD.knob_only(d) == (BOLD.novelty(d) == 0.0)


def test_novelty_counts_new_functions_classes_modules_and_loss_terms():
    p = BOLD.novelty_parts(_code([], AUX))
    assert p["defs"] >= 2 and p["modules"] == 1 and p["classes"] == 0 and p["losses"] >= 1 and not p["knob_only"]
    just_def = BOLD.novelty_parts(_code([], ["def neighbour(z):", "    return z"]))
    assert just_def["defs"] == 1 and just_def["modules"] == 0 and just_def["losses"] == 0
    assert 0.0 < BOLD.novelty(_code([], ["def neighbour(z):", "    return z"])) < BOLD.novelty(_code([], AUX)) < 1.0
    plain_cls = BOLD.novelty_parts(_code([], ["class Holder:", "    pass"]))
    assert plain_cls["classes"] == 1 and plain_cls["modules"] == 0
    assert BOLD.novelty(_code([], ["    h = nn.Linear(4, 4)(z)"])) > 0.0      # the old test's anchor: still structural


def test_novelty_no_longer_saturates_and_is_ordered_by_how_much_is_new():
    small = _code([], AUX[:4] + ["def aux_loss(a, b):", "    return a"])
    big = _code([], AUX + [f"def f{i}(x):\n    return x + {i}" for i in range(12)] + ["class A(nn.Module):", "    pass"])
    s, m, b = BOLD.novelty(small), BOLD.novelty(_code([], AUX)), BOLD.novelty(big)
    assert 0 < s < m < b < 1.0                                                # the old min(1, n / 6) gave 1.0 to all three
    assert BOLD.novelty(_code([], AUX)) != 1.0


def test_novelty_is_relative_to_the_parent_source():
    d = _code([], ["def aux_loss(a, b):", "    return (a - b).pow(2).mean()", "    t = aux_loss(z, y)"])
    alone = BOLD.novelty_parts(d)
    assert alone["defs"] == 1 and alone["losses"] >= 1
    parent_has = lambda path: "def aux_loss(a, b):\n    return a\n"              # noqa: E731 - the parent defined it already
    rel = BOLD.novelty_parts(d, parent_has)
    assert rel["defs"] == 0 and rel["losses"] == 0
    assert BOLD.novelty(d, parent_has) < BOLD.novelty(d)
    assert BOLD.novelty(d, lambda path: None) == BOLD.novelty(d)                # no parent source: the diff's own context


def test_the_bold_gating_still_rejects_a_knob_and_keeps_the_structural(ctx):
    knob = _code(["    w = 0.5"], ["    w = 0.7"])
    assert BOLD.knob_rule(ctx, knob, "x", judge=None)[0] == "bold_knob_only"
    assert BOLD.knob_rule(ctx, _code([], AUX), "x", judge=None) == ("", "")


def test_free_gpus_counts_startable_gpus_of_healthy_nodes_only(np_ctx, base_repo):   # noqa: F811
    pools = (PoolCfg("primary", slots=6, account="main", qos="normal", partition="gpu"),
             PoolCfg("idle", slots=8, account="preempt", qos="low", partition="gpu", preemptible=True))
    ctx = dataclasses.replace(np_ctx, cfg=dataclasses.replace(np_ctx.cfg, slurm=SlurmCfg(pools=pools)))
    g = "gpu:nvidia_h100_80gb_hbm3:8(S:0-1)"
    rows = "".join(f"{n:<24}{g:<60}{u:<80}{st:<24}\n" for n, u, st in [
        ("node-1", "gpu:nvidia_h100_80gb_hbm3:8(IDX:0-7)", "allocated"),
        ("node-2", "gpu:nvidia_h100_80gb_hbm3:5(IDX:0-4)", "mixed"),
        ("node-3", "gpu:nvidia_h100_80gb_hbm3:0(IDX:N/A)", "drained"),
        ("node-4", "gpu:nvidia_h100_80gb_hbm3:0(IDX:N/A)", "reserved"),
        ("node-5", "gpu:nvidia_h100_80gb_hbm3:7(IDX:1-7)", "completing"),
        ("node-6", "gpu:nvidia_h100_80gb_hbm3:0(IDX:N/A)", "idle")])
    adapter, runner = make_np(ctx, base_repo, runner=SlurmRunner({"sinfo": (0, rows, "")}))
    assert adapter.free_gpus() == {"gpu": 3 + 1 + 8}                   # node-2: 3, node-5: 1, node-6: 8
    assert len(runner.argvs("sinfo")) == 1 and not runner.argvs("scancel")
    bad, _ = make_np(ctx, base_repo, runner=SlurmRunner({"sinfo": (1, "", "slurm down")}))
    with pytest.raises(RuntimeError):
        bad.free_gpus()
