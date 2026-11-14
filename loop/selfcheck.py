"""loop.selfcheck -- the end-to-end preflight a research loop must pass before it spends an LLM session (11-08).

11-08 nanobrain: 1,207 research sessions failed in 10 h on defects no session could fix -- an unsatisfiable slate rule
on a thin history (every slate rejected, edits locked), a smoke override train.py asserts against (every smoke died
rc=1), a NaN scan that read the step-0 throughput meters as divergence. Each is checkable WITHOUT an LLM, against the
REAL task config, in seconds. ``python -m loop selfcheck --task T`` runs:

- (a) ``slate``: a minimal synthetic slate (the gate's own limits: count, categories, verbalized probabilities and
  tails, grounding in the CURRENT history table's clusters / node ids, differs_from) passes every slate kind's gate,
  the mechanism memory included -- on a read-only view of the store;
- (b) ``train_config``: the smoke overrides (``adapter.smoke_config`` of the root's config at ``base_sha``) give a
  config the real train.py accepts: the project python merges it exactly as train.py's ``__main__`` does, evaluates
  every startup ``assert`` of ``main()`` that reads only ``args`` (following the ``if``s that do), and builds the model
  on CPU (a smoke-only build failure fails; when the root config itself cannot build here it is a warning). No
  training, no GPU, nothing written outside a temp dir;
- (c) ``nan_scan``: the adapter's smoke NaN scan passes a known-good smoke log (``adapter.KNOWN_GOOD_SMOKE_LOG``);
- (d) ``breaker``: the research breaker (loop.breaker) is not tripped;
- (e) ``locks``: the loop's LOCKED files match its deployed HEAD (core.meta.Locks) and the root commit's
  ``surface.locked`` files match the adapter's canonical lock ref.

Exit status 1 on any FAIL. ``loop run`` runs it at startup (:func:`gate`): a failure is logged
``ALERT_selfcheck`` + stderr, and NO research session is issued (the breaker holds them) until a passing selfcheck
(each restart re-runs it; ``<state_dir>/selfcheck.skip`` bypasses it once, consumed).
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

K_SELFCHECK = "selfcheck.last"
SKIP_FILE = "selfcheck.skip"
PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


@dataclasses.dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str = ""

    def line(self) -> str:
        return f"[{self.status}] {self.name}: {self.detail}"


class ReadOnlyStore:
    """The store with every write a no-op (the selfcheck never changes the loop's state)."""

    _WRITES = frozenset({"set_state", "log", "put_local_run", "put_node", "add_node", "update_node", "put_run",
                         "add_run", "put_hypothesis", "add_handoff", "put_postmortem"})

    def __init__(self, store: Any) -> None:
        self._s = store

    def __getattr__(self, name: str) -> Any:
        if name in self._WRITES:
            return lambda *a, **k: None
        return getattr(self._s, name)


# ---------------------------------------------------------------------------------------------- (a) the slate gate
def history_table(ctx: Any, store: Any, db: str | Path | None) -> tuple[Any, str]:
    """(the history table a session would see now, '' or why there is none) -- built like the tick builds it."""
    if db is None or not Path(db).exists():
        return None, "no ledger db yet"
    try:
        from loop.agents import history as HX
        from loop.core import data as CD
        from loop.core import ideation as IDE
        led = CD.load(db, ctx.task_id)
        v = IDE.view(led, IDE.blocks_of(store, led))
        return HX.build(ctx, store, v, None), ""
    except (ValueError, KeyError, ZeroDivisionError) as exc:      # the live tick has no view either: no table
        return None, f"no history view ({type(exc).__name__}: {str(exc)[:120]})"


def minimal_slate(ctx: Any, kind: str, tab: Any) -> list[dict[str, Any]]:
    """The smallest slate the gate's own limits allow, grounded in ``tab`` (synthetic words: it can repeat nothing)."""
    from loop.core import bold as BOLD
    from loop.core.levers import opt
    lim = BOLD.slate_limits(ctx, kind)
    cats = [BOLD.label(c) for c in BOLD.TAXONOMY if c not in (BOLD.KNOB, BOLD.OTHER)] or [BOLD.label(c) for c in BOLD.TAXONOMY]
    k = max(int(lim["k_min"]), int(lim["min_cats"]))
    tail_p, tail_min = float(opt(ctx, "portfolio_tail_p")), int(opt(ctx, "portfolio_tail_min"))
    n_tail = min(k, max(tail_min, 0))
    tail = min(tail_p, 0.5 / max(1, k)) if n_tail else 0.0
    head = (1.0 - tail * n_tail) / max(1, k - n_tail) if k > n_tail else 0.0
    probs = [tail] * n_tail + [head] * (k - n_tail)
    clusters = sorted(BOLD.table_clusters(tab)) if tab is not None else []
    nodes = sorted(getattr(tab, "by_id", {}) or {}) if tab is not None else []
    ground = clusters or nodes
    out = []
    for i in range(k):
        g = ground[i % len(ground)] if ground else ""
        words = " ".join(f"selfcheckzq{i}w{j}" for j in range(16))
        c = {"title": f"selfcheck probe {i} zq{i}", "category": cats[i % len(cats)],
             "mechanism": f"Synthetic preflight candidate {i}: {words}.", "family": "selfcheck",
             "mechanism_tag": f"selfcheck_probe_{i}", "why": "preflight: proves the gate is satisfiable",
             "plausibility": 0.2, "p_neg": 0.5, "median": 0.0, "p_gap": 0.1, "probability": round(probs[i], 6)}
        if g:
            c["grounded_in"] = g
            c["differs_from"] = [{"id": g, "how": f"selfcheck synthetic probe zq{i} unrelated alpha beta gamma delta"}]
        out.append(c)
    return out


def check_slate(ctx: Any, store: Any, db: str | Path | None) -> Check:
    from loop.core import bold as BOLD
    from loop.core.levers import opt
    kinds = tuple(BOLD.slate_kinds(ctx))
    if not kinds:
        return Check("slate", SKIP, "no slate kinds (ideation.bold_slate_kinds empty)")
    ro = ReadOnlyStore(store)
    tab, why = history_table(ctx, ro, db) if opt(ctx, "history") else (None, "history tool off")
    bad, ok = [], []
    for kind in kinds:
        cands = minimal_slate(ctx, kind, tab)
        gate = BOLD.Slate(ctx, ro, "selfcheck", kind, judge=None, logger=None,
                          history=(lambda t=tab: t) if opt(ctx, "history") else None)
        got = gate(cands)
        if gate.done:
            ok.append(kind)
        else:
            bad.append(f"{kind}: {got.splitlines()[0][:400]}")
    n = len(getattr(tab, "by_id", {}) or {}) if tab is not None else 0
    ctx_line = (f"history: {n} nodes, {len(BOLD.table_clusters(tab))} clusters" if tab is not None else f"history: {why}")
    if bad:
        return Check("slate", FAIL, f"the gate rejects a minimal slate ({ctx_line}): " + " | ".join(bad))
    return Check("slate", PASS, f"a minimal slate passes for {', '.join(ok)} ({ctx_line})")


# ---------------------------------------------------------------------------------------------- (b) train.py accepts
PROBE = r'''
import ast, json, sys, traceback
from pathlib import Path
root, train_rel, default_rel = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
cfgs, overrides = json.loads(sys.argv[4]), json.loads(sys.argv[5])
build = sys.argv[6] if len(sys.argv) > 6 else ""   # the adapter's SELFCHECK_MODEL: code binding `m` from `cfg`
sys.path.insert(0, str(root / "src"))
from omegaconf import OmegaConf
out = {}
src = (root / train_rel).read_text()
tree = ast.parse(src)
main = next((f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == "main"), None)
SAFE = {"len", "isinstance", "int", "float", "str", "bool", "tuple", "list", "min", "max", "abs", "all", "any",
        "sorted", "set", "dict", "sum", "round", "None", "True", "False"}
def only_args(node):
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    return "args" in names and names <= SAFE | {"args"}
def visit(stmts, args, fails):
    for st in stmts:
        if isinstance(st, ast.Assert) and only_args(st.test):
            try:
                ok = eval(compile(ast.Expression(st.test), train_rel, "eval"), {"args": args})
            except Exception as exc:
                ok, why = False, f"{type(exc).__name__}: {exc}"
            else:
                why = ""
            if not ok:
                msg = ast.unparse(st.msg) if st.msg is not None else ""
                fails.append(f"line {st.lineno}: assert {ast.unparse(st.test)} {msg} {why}".strip())
        elif isinstance(st, ast.If) and only_args(st.test):
            try:
                take = bool(eval(compile(ast.Expression(st.test), train_rel, "eval"), {"args": args}))
            except Exception:
                continue
            visit(st.body if take else st.orelse, args, fails)
for name, cfg_path in cfgs.items():
    r = {"asserts": [], "model": ""}
    try:
        cfg = OmegaConf.load(str(root / default_rel))
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(cfg_path))
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(overrides))
    except Exception as exc:
        r["merge"] = f"{type(exc).__name__}: {exc}"
        out[name] = r
        continue
    if main is not None:
        visit(main.body, cfg, r["asserts"])
    if build:
        try:
            ns = {"cfg": cfg}
            exec(build, ns)
            r["params"] = sum(p.numel() for p in ns["m"].parameters())
        except Exception as exc:
            r["model"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    out[name] = r
print("SELFCHECK_JSON " + json.dumps(out))
'''


def _python(ctx: Any, adapter: Any) -> str:
    proj = getattr(getattr(ctx, "cfg", None), "project", None)
    py = (proj.get("python") if isinstance(proj, Mapping) else getattr(proj, "python", "")) or ""
    py = py or getattr(adapter, "python", "") or ""
    if not py and getattr(adapter, "repo", None):
        py = str(Path(adapter.repo) / ".venv" / "bin" / "python")
    return str(py)


def _extract(repo: str, sha: str, dest: Path, timeout: float = 300.0) -> str:
    """``git archive <sha>`` of the repo into ``dest`` (read-only on the repo); '' or the error."""
    tar = dest / "src.tar"
    with open(tar, "wb") as fh:
        cp = subprocess.run(["git", "-C", repo, "archive", "--format=tar", sha], stdout=fh, stderr=subprocess.PIPE,
                            timeout=timeout)
    if cp.returncode != 0:
        return cp.stderr.decode(errors="replace")[-300:] or f"git archive rc={cp.returncode}"
    with tarfile.open(tar) as tf:
        members = [m for m in tf.getmembers() if not (m.issym() or m.islnk() or m.isdev())
                   and not m.name.startswith(("/", "..")) and ".." not in Path(m.name).parts]
        tf.extractall(dest / "tree", members=members)
    tar.unlink()
    return ""


def check_train_config(ctx: Any, adapter: Any, *, timeout: float = 600.0,
                       runner: Callable[..., Any] = subprocess.run) -> Check:
    if not callable(getattr(adapter, "smoke_config", None)):
        return Check("train_config", SKIP, f"{type(adapter).__name__} has no smoke_config")
    repo = str(getattr(adapter, "repo", "") or "")
    sha = str(getattr(adapter, "root_commit", lambda: "")() or "")
    rel = str(getattr(adapter, "config_rel", "") or "")
    train_rel = str(getattr(adapter, "TRAIN_REL", "") or "")
    default_rel = str(getattr(adapter, "DEFAULT_CONFIG_REL", "") or rel)
    if not (repo and sha and rel and train_rel):
        return Check("train_config", SKIP, "the adapter names no repo / base sha / config / TRAIN_REL")
    py = _python(ctx, adapter)
    if not Path(py).exists():
        return Check("train_config", FAIL, f"the project python {py} does not exist")
    with tempfile.TemporaryDirectory(prefix="loop-selfcheck-") as td:
        tdp = Path(td)
        if err := _extract(repo, sha, tdp):
            return Check("train_config", FAIL, f"cannot read base {sha[:10]} from {repo}: {err}")
        root = tdp / "tree"
        try:
            base = yaml.safe_load((root / rel).read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            return Check("train_config", FAIL, f"cannot read the root config {rel} at {sha[:10]}: {exc}")
        smoke = adapter.smoke_config(base)
        (tdp / "base.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
        (tdp / "smoke.yaml").write_text(yaml.safe_dump(smoke, sort_keys=False))
        overrides = [str(o).format(out=tdp / "out", seed=smoke.get("seed", 0))     # the task's dotlist overrides
                     for o in getattr(adapter, "SELFCHECK_OVERRIDES", None) or ()]
        (tdp / "data").mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "VIRTUAL_ENV", "PYTHON"))}
        env.update(CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="4")
        env.update({k: str(v).format(data=tdp / "data") for k, v in
                    dict(getattr(adapter, "SELFCHECK_ENV", None) or {}).items()})   # the task's eval data env etc.
        argv = [py, "-I", "-c", PROBE, str(root), train_rel, default_rel,
                json.dumps({"base": str(tdp / "base.yaml"), "smoke": str(tdp / "smoke.yaml")}), json.dumps(overrides),
                str(getattr(adapter, "SELFCHECK_MODEL", "") or "")]
        try:
            cp = runner(argv, cwd=str(root), env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Check("train_config", FAIL, f"the config probe exceeded {timeout:.0f}s")
        line = next((ln for ln in (cp.stdout or "").splitlines() if ln.startswith("SELFCHECK_JSON ")), "")
        if cp.returncode != 0 or not line:
            return Check("train_config", FAIL, f"the config probe failed rc={cp.returncode}: "
                                               f"{((cp.stderr or '') + (cp.stdout or ''))[-600:]}")
        got = json.loads(line.split(" ", 1)[1])
    return train_verdict(got, sha)


def train_verdict(got: Mapping[str, Any], sha: str = "") -> Check:
    """The probe's {base, smoke} -> the check: a merge error or a failed startup assert on the smoke config fails;
    the model must build from the smoke config whenever it builds from the root's."""
    sm, bs = got.get("smoke") or {}, got.get("base") or {}
    if sm.get("merge"):
        return Check("train_config", FAIL, f"train.py's config merge rejects the smoke config: {sm['merge']}")
    if sm.get("asserts"):
        return Check("train_config", FAIL, "train.py main() asserts fail on the smoke config: "
                                           + "; ".join(sm["asserts"])[:600])
    if bs.get("asserts"):
        return Check("train_config", FAIL, "train.py main() asserts fail on the ROOT config: "
                                           + "; ".join(bs["asserts"])[:600])
    if sm.get("model") and not bs.get("model"):
        return Check("train_config", FAIL, f"the model builds from the root config but not the smoke config: {sm['model']}")
    if sm.get("model"):
        return Check("train_config", WARN, f"asserts pass; the model build cannot be checked here (it fails on the "
                                           f"root config too: {bs.get('model', '')[:200]})")
    return Check("train_config", PASS, f"train.py accepts the smoke config of {sha[:10]} (merge, startup asserts, "
                                       f"model build: {sm.get('params', '?')} params)")


# ---------------------------------------------------------------------------------------------- (c) (d) (e)
def check_nan_scan(adapter: Any) -> Check:
    scan = getattr(adapter, "nan_scan", None)
    path = getattr(adapter, "KNOWN_GOOD_SMOKE_LOG", None)
    if not callable(scan) or not path:
        return Check("nan_scan", SKIP, f"{type(adapter).__name__} declares no NaN scan / known-good smoke log")
    try:
        text = Path(path).read_text()
    except OSError as exc:
        return Check("nan_scan", FAIL, f"known-good smoke log unreadable: {exc}")
    if scan(text):
        return Check("nan_scan", FAIL, f"the smoke NaN scan flags the known-good log {Path(path).name}: every "
                                       "passing smoke would be rejected 'smoke log shows NaN/inf'")
    return Check("nan_scan", PASS, f"the known-good smoke log {Path(path).name} passes the NaN scan")


def check_breaker(ctx: Any, store: Any) -> Check:
    from loop.breaker import Breaker
    why = Breaker(ctx, store).tripped()
    if why:
        return Check("breaker", FAIL, f"the research breaker is tripped: {why[:400]} (clear: touch "
                                      f"{Path(getattr(ctx, 'state_dir', '.') or '.') / 'research_breaker.clear'})")
    return Check("breaker", PASS, "not tripped")


def check_locks(ctx: Any, adapter: Any, locks: Any = None) -> Check:
    from loop.core.meta import Locks
    out = []
    lk = locks if locks is not None else Locks()
    if lk.error:
        out.append(f"loop locks unreadable: {lk.error}")
    elif bad := lk.verify():
        out.append(f"loop LOCKED files differ from the deployed HEAD {lk.head[:10]}: {bad}")
    lv = getattr(adapter, "lock_violations", None)
    root = getattr(adapter, "root_commit", None)
    if callable(lv) and callable(root):
        try:
            bad = lv(root())
        except Exception as exc:  # noqa: BLE001 - fail closed
            bad = [f"<lock check failed: {type(exc).__name__}: {exc}>"]
        if bad:
            out.append(f"the root commit's locked files differ from the canonical lock ref: {bad[:8]}")
    if out:
        return Check("locks", FAIL, "; ".join(out)[:700])
    return Check("locks", PASS, f"loop LOCKED files == deployed HEAD {getattr(lk, 'head', '')[:10]}; the root "
                                "commit's locked files == the lock ref")


# ---------------------------------------------------------------------------------------------- the run
def run(ctx: Any, store: Any, adapter: Any, *, db: str | Path | None = None, train: bool = True,
        locks: Any = None) -> list[Check]:
    """Every check, each isolated (a crash in one is its own FAIL)."""
    steps: list[tuple[str, Callable[[], Check]]] = [
        ("slate", lambda: check_slate(ctx, store, db)),
        ("train_config", (lambda: check_train_config(ctx, adapter)) if train
         else (lambda: Check("train_config", SKIP, "skipped (--no-train-config)"))),
        ("nan_scan", lambda: check_nan_scan(adapter)),
        ("breaker", lambda: check_breaker(ctx, store)),
        ("locks", lambda: check_locks(ctx, adapter, locks)),
    ]
    out = []
    for name, fn in steps:
        try:
            out.append(fn())
        except Exception as exc:  # noqa: BLE001 - a crashing check is a failed check
            out.append(Check(name, FAIL, f"the check crashed: {type(exc).__name__}: {str(exc)[:400]}"))
    return out


def failed(checks: Sequence[Check]) -> list[Check]:
    return [c for c in checks if c.status == FAIL]


def report(checks: Sequence[Check]) -> str:
    bad = failed(checks)
    return "\n".join([c.line() for c in checks] + [f"selfcheck: {'FAIL (' + ', '.join(c.name for c in bad) + ')' if bad else 'PASS'}"])


def gate(ctx: Any, store: Any, adapter: Any, *, db: str | Path | None = None,
         out: Callable[[str], None] = print, runner: Callable[..., list[Check]] | None = None) -> str:
    """``loop run``'s startup gate: '' when the selfcheck passes (or ``<state_dir>/selfcheck.skip`` was dropped:
    consumed), else the hold reason -- logged ``ALERT_selfcheck`` and printed to stderr; the caller holds research."""
    import time
    skip = Path(getattr(ctx, "state_dir", "") or ".") / SKIP_FILE
    if getattr(ctx, "state_dir", "") and skip.exists():
        try:
            skip.unlink()
        except OSError:
            pass
        store.log("selfcheck_skipped", why="operator skip file")
        out(f"selfcheck: SKIPPED once ({skip} consumed)")
        return ""
    checks = (runner or run)(ctx, store, adapter, db=db)
    out(report(checks))
    bad = failed(checks)
    store.set_state(K_SELFCHECK, {"ts": time.time(), "ok": not bad,
                                  "checks": [dataclasses.asdict(c) for c in checks]})
    if not bad:
        store.log("selfcheck", ok=True, checks={c.name: c.status for c in checks})
        return ""
    why = "selfcheck failed: " + "; ".join(f"{c.name}: {c.detail}" for c in bad)
    store.log("ALERT_selfcheck", failed=[c.name for c in bad], detail={c.name: c.detail[:500] for c in bad})
    print(f"ALERT {why[:1500]} -- no research session is issued until a restart passes the selfcheck "
          f"(or `touch {skip}`)", file=sys.stderr, flush=True)
    return why[:600]


def main(argv: Sequence[str]) -> int:
    """``selfcheck --task T [--state-dir D] [--no-train-config]``: print every check; exit 1 on any FAIL."""
    import argparse
    from loop.spec import load_context
    from loop.store import SqliteStore
    ap = argparse.ArgumentParser(prog="python -m loop selfcheck")
    ap.add_argument("--task", required=True)
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--no-train-config", action="store_true", help="skip (b): no project-python probe")
    args = ap.parse_args(list(argv))
    ctx = load_context(args.task, state_dir=args.state_dir, require_measured=False)
    from loop.__main__ import resolve_adapter
    adapter = resolve_adapter(ctx)
    db = Path(ctx.state_dir) / f"{ctx.task_id}.db"
    if db.exists():
        store: Any = ReadOnlyStore(SqliteStore(db, ctx.task_id, weights=ctx.cfg.weights))
    else:
        store = ReadOnlyStore(_EmptyStore())
    checks = run(ctx, store, adapter, db=db if db.exists() else None, train=not args.no_train_config)
    print(report(checks))
    return 1 if failed(checks) else 0


class _EmptyStore:
    """A fresh state dir: no nodes, no state."""

    def nodes(self) -> list[Any]:
        return []

    def get_state(self, key: str, default: Any = None) -> Any:
        return default

    def local_runs(self) -> list[dict[str, Any]]:
        return []

    def get_node(self, nid: str) -> Any:
        return None

    def hypotheses(self) -> list[Any]:
        return []

    def events(self, *a: Any, **k: Any) -> list[Any]:
        return []
