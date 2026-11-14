"""loop.project -- the per-process project registry (multi-project Stage 1, the nanobrain port).

The loop core was written against nanopath's six probe axes as module constants. This module holds them as LIVE
views over one per-process state, so a project's ``loop.project`` / ``loop.domain`` yaml block can replace them
(:func:`install`, called by :func:`loop.spec.load_context`) and every reader -- including names imported with
``from loop.core.data import AXES`` before the install -- sees the installed values.

API (frozen; all packages code against it):

- :data:`AXES` -- live read-only sequence of short axis names (iter / in / len / index / ``list()`` / ``*`` unpack)
- :data:`PROBE_KEYS` (short -> full summary key), :data:`OFFICIAL_WEIGHTS` (short -> panel weight), :data:`HARM_H`
  (short -> tolerated loss), :data:`AXIS_NAMES` (short -> prose name): live read-only mappings
- :data:`GUARD_AXES` -- live sequence
- :func:`target` -> the ideation target axis; :func:`guard` -> the first guard axis or None
- :func:`install` / :func:`reset` (tests reset via an autouse fixture)
- :func:`prompt_path` -- a prompt template, the domain overlay dir first, then ``loop/prompts``
- :func:`rule` -- a domain text rule (``domain.rules[key]``) else the caller's default
- :func:`composite_mode` -- "reliability" (nanopath: official x per-axis reliability) | "official" (pinned)
- :func:`throughput` -- (summary key of a run's optimizer steps, tolerance): runs whose steps deviate from their
  config's median by more than the tolerance are left out of the noise fit (wall-clock budgets; 0 = off)

With nothing installed every value equals the nanopath literal it replaced (parity)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

PROMPTS = Path(__file__).resolve().parent / "prompts"
REPO = Path(__file__).resolve().parent.parent

# -- nanopath defaults: EXACTLY the literals of loop/core/data.py, core/ideation.py, core/test.py, core/acquire.py --
_AXES = ("cls", "seg", "slide", "auc", "surv", "rob")
_PROBE_KEYS = {"cls": "final_probe_classification_mean_f1", "seg": "final_probe_seg_mean_f1",
               "slide": "final_probe_slide_mean_auc", "auc": "final_probe_auc_mean",
               "surv": "final_probe_survival_mean_cindex", "rob": "final_probe_robustness_mean"}
_OFFICIAL_WEIGHTS = {"cls": 0.30, "seg": 0.15, "slide": 0.175, "auc": 0.15, "surv": 0.075, "rob": 0.15}
_AXIS_NAMES = {"cls": "classification", "seg": "segmentation", "slide": "progression", "auc": "mutation",
               "surv": "survival", "rob": "robustness"}
_TARGET = "cls"
_GUARD_AXES = ("rob",)
_HARM_H = {"rob": 0.008}
_AXIS_EXCLUDE = ("surv",)


#: the generic default of ``loop.project.noise_floor_seeds`` (11-10 generalization: ON for every task with a project
#: block; a no-op once the root has this many done seeds). 0 in a task yaml = off.
NOISE_FLOOR_SEEDS = 5


@dataclass(frozen=True)
class AxisCfg:
    name: str                          # short name (AXES entry)
    key: str                           # the full summary / slots.vector key
    label: str = ""                    # prose name (AXIS_NAMES); "" = name
    weight: float = 0.0                # raw panel weight (normalised by install when aggregation is weighted)
    tasks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectCfg:
    name: str = "nanopath"
    base_sha: str = ""
    runs_root: str = ""
    worktrees_root: str = ""
    config_rel: str = ""
    python: str = ""                   # the interpreter / venv jobs and worktrees use ("" = <repo>/.venv/bin/python)
    aggregation: str = "weighted"      # weighted | mean
    axes: tuple[AxisCfg, ...] = ()
    target: str = ""                   # "" = the first axis
    guard: tuple[str, ...] = ()
    harm_h: Mapping[str, float] = field(default_factory=dict)
    axis_exclude: tuple[str, ...] = ()
    budget: Mapping[str, Any] = field(default_factory=dict)
    #: the screening composite's weights (core.model.composite_weights): "reliability" = official x per-axis
    #: reliability (nanopath), "official" = pinned to the official weights (the composite is then the scored metric)
    composite: str = "reliability"
    #: wall-clock budgets: a run's optimizer-step count (``throughput.key`` in its scores) varies with the node / load;
    #: a run whose steps deviate from its config's median by > ``throughput.tol`` (a fraction) is left out of the
    #: noise fit (core.model.fit_noise_target). "" / 0 = off (nanopath: a FLOP budget, steps are fixed)
    throughput_key: str = ""
    throughput_tol: float = 0.0
    #: the root-seed noise floor (core.model.floor_noise): while the current base root has fewer than this many done
    #: seeds, every fitted per-run sigma (axes, panel, composite) is at least the spec's MEASURED one (loop.panel_sd /
    #: axis_sd over sqrt 2) -- a 3-seed fit (nanobrain 11-08: panel .0027 vs measured .0049) is overconfident.
    #: 0 = off. parse_project's default: NOISE_FLOOR_SEEDS (5)
    noise_floor_seeds: int = 0
    #: the world-model cell prior's tau2 estimator (core.model.fit_prior): "robust" (Paule-Mandel, the 11-10 fix --
    #: parse_project's default) | "mad_minus_mean" ((1.4826 MAD)^2 - mean V, the pre-11-10 estimator; nanopath pins it)
    cell_tau2: str = "mad_minus_mean"


@dataclass(frozen=True)
class DomainCfg:
    prompt_dir: str = ""               # overlay: <name>.md here wins over loop/prompts/<name>.md ("" = none)
    rules: Mapping[str, str] = field(default_factory=dict)


@dataclass
class _State:
    axes: tuple[str, ...] = _AXES
    probe_keys: dict[str, str] = field(default_factory=lambda: dict(_PROBE_KEYS))
    weights: dict[str, float] = field(default_factory=lambda: dict(_OFFICIAL_WEIGHTS))
    names: dict[str, str] = field(default_factory=lambda: dict(_AXIS_NAMES))
    target: str = _TARGET
    guard: tuple[str, ...] = _GUARD_AXES
    harm_h: dict[str, float] = field(default_factory=lambda: dict(_HARM_H))
    axis_exclude: tuple[str, ...] = _AXIS_EXCLUDE
    composite: str = "reliability"
    throughput_key: str = ""
    throughput_tol: float = 0.0
    noise_floor_seeds: int = 0
    noise_floor: dict[str, float] = field(default_factory=dict)
    cell_tau2: str = "mad_minus_mean"           # no project installed: the pre-11-10 estimator (parity)
    prompt_dir: Path | None = None
    rules: dict[str, str] = field(default_factory=dict)
    project: ProjectCfg | None = None
    domain: DomainCfg | None = None


_S = _State()


class _LiveSeq(Sequence):
    """A read-only tuple-like view of one ``_State`` field (re-read on every access)."""

    def __init__(self, attr: str) -> None:
        self._attr = attr

    def _v(self) -> tuple:
        return getattr(_S, self._attr)

    def __getitem__(self, i):                       # type: ignore[override]
        return self._v()[i]

    def __len__(self) -> int:
        return len(self._v())

    def __iter__(self) -> Iterator:
        return iter(self._v())

    def __contains__(self, x: object) -> bool:
        return x in self._v()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _LiveSeq):
            other = other._v()
        return self._v() == tuple(other) if isinstance(other, (tuple, list)) else self._v() == other

    def __hash__(self) -> int:
        return hash(self._v())

    def __add__(self, other):
        return self._v() + tuple(other)

    def __radd__(self, other):
        return tuple(other) + self._v()

    def __repr__(self) -> str:
        return repr(self._v())


class _LiveMap(Mapping):
    """A read-only dict-like view of one ``_State`` field (re-read on every access)."""

    def __init__(self, attr: str) -> None:
        self._attr = attr

    def _v(self) -> dict:
        return getattr(_S, self._attr)

    def __getitem__(self, k):
        return self._v()[k]

    def __len__(self) -> int:
        return len(self._v())

    def __iter__(self) -> Iterator:
        return iter(self._v())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _LiveMap):
            other = other._v()
        return self._v() == other

    __hash__ = None                                 # type: ignore[assignment]

    def __repr__(self) -> str:
        return repr(self._v())


AXES: Sequence[str] = _LiveSeq("axes")
GUARD_AXES: Sequence[str] = _LiveSeq("guard")
PROBE_KEYS: Mapping[str, str] = _LiveMap("probe_keys")
OFFICIAL_WEIGHTS: Mapping[str, float] = _LiveMap("weights")
HARM_H: Mapping[str, float] = _LiveMap("harm_h")
AXIS_NAMES: Mapping[str, str] = _LiveMap("names")


def target() -> str:
    return _S.target


def guard() -> str | None:
    return _S.guard[0] if _S.guard else None


def axis_exclude() -> tuple[str, ...]:
    return _S.axis_exclude


def composite_mode() -> str:
    """"reliability" (nanopath default) | "official" (``loop.project.composite``)."""
    return _S.composite


def throughput() -> tuple[str, float]:
    """(steps summary key, tolerance) of the throughput filter (``loop.project.throughput``); ("", 0.0) = off."""
    return _S.throughput_key, _S.throughput_tol


def noise_floor() -> tuple[int, Mapping[str, float]]:
    """(root seeds below which the floor holds, per-run sigma floor per target); (0, {}) = off."""
    return (_S.noise_floor_seeds, _S.noise_floor) if _S.noise_floor_seeds > 0 and _S.noise_floor else (0, {})


def cell_tau2() -> str:
    """The world-model cell prior's tau2 estimator (``loop.project.cell_tau2``)."""
    return _S.cell_tau2


def set_noise_floor(floor: Mapping[str, float]) -> None:
    """The per-run sigma floors (load_context: the spec's measured sds); kept until the next install."""
    _S.noise_floor = {str(k): float(v) for k, v in floor.items()}


def current() -> ProjectCfg | None:
    """The installed project block (None = nanopath defaults, no block)."""
    return _S.project


def domain() -> DomainCfg | None:
    return _S.domain


def _f(x: Any) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"loop.project: non-finite number {x!r}")
    return v


def parse_project(raw: Mapping[str, Any] | ProjectCfg | None) -> ProjectCfg | None:
    """``loop.project`` yaml -> :class:`ProjectCfg` (validated; slots.vector cross-check lives in loop.spec)."""
    if raw is None:
        return None
    if isinstance(raw, ProjectCfg):
        return parse_project({**{k: getattr(raw, k) for k in raw.__dataclass_fields__
                                 if k not in ("throughput_key", "throughput_tol")}, "axes": list(raw.axes),
                              "throughput": {"key": raw.throughput_key, "tol": raw.throughput_tol}})
    if not isinstance(raw, Mapping):
        raise ValueError(f"loop.project must be a mapping, got {type(raw).__name__}")
    known = {"name", "base_sha", "runs_root", "worktrees_root", "config_rel", "python", "aggregation", "axes", "target",
             "guard", "harm_h", "axis_exclude", "budget", "composite", "throughput", "noise_floor_seeds", "cell_tau2"}
    if bad := set(raw) - known:
        raise ValueError(f"loop.project: unknown keys {sorted(bad)}")
    axes: list[AxisCfg] = []
    for i, a in enumerate(raw.get("axes") or ()):
        if isinstance(a, AxisCfg):
            axes.append(a)
            continue
        if not isinstance(a, Mapping) or not a.get("name") or not a.get("key"):
            raise ValueError(f"loop.project.axes[{i}]: needs name and key")
        if bad := set(a) - {"name", "key", "label", "weight", "tasks"}:
            raise ValueError(f"loop.project.axes[{i}]: unknown keys {sorted(bad)}")
        axes.append(AxisCfg(str(a["name"]), str(a["key"]), str(a.get("label") or ""), _f(a.get("weight", 0.0)),
                            tuple(str(t) for t in (a.get("tasks") or ()))))
    if not axes:
        raise ValueError("loop.project.axes: at least one axis is required")
    names = [a.name for a in axes]
    if len(set(names)) != len(names):
        raise ValueError(f"loop.project.axes: duplicate axis names {names}")
    if len({a.key for a in axes}) != len(axes):
        raise ValueError("loop.project.axes: duplicate axis keys")
    agg = str(raw.get("aggregation") or "weighted")
    if agg not in ("weighted", "mean"):
        raise ValueError(f"loop.project.aggregation: {agg!r} (weighted | mean)")
    if agg == "weighted" and (any(a.weight < 0 for a in axes) or sum(a.weight for a in axes) <= 0):
        raise ValueError("loop.project.axes: weighted aggregation needs non-negative weights with a positive sum")
    tgt = str(raw.get("target") or names[0])
    g = raw.get("guard", [])
    guard_t = (str(g),) if isinstance(g, str) else tuple(str(x) for x in (g or ()))
    harm = {str(k): _f(v) for k, v in dict(raw.get("harm_h") or {}).items()}
    ax_ex = tuple(str(x) for x in (raw.get("axis_exclude") or ()))
    for what, vals in (("target", [tgt]), ("guard", guard_t), ("harm_h", list(harm)), ("axis_exclude", ax_ex)):
        if bad := [v for v in vals if v not in names]:
            raise ValueError(f"loop.project.{what}: {bad} not in axes {names}")
    budget = raw.get("budget") or {}
    if not isinstance(budget, Mapping):
        raise ValueError("loop.project.budget must be a mapping")
    comp = str(raw.get("composite") or "reliability")
    if comp not in ("reliability", "official"):
        raise ValueError(f"loop.project.composite: {comp!r} (reliability | official)")
    tp = raw.get("throughput") or {}
    if not isinstance(tp, Mapping) or set(tp) - {"key", "tol"}:
        raise ValueError("loop.project.throughput must be a mapping {key, tol}")
    tp_key, tp_tol = str(tp.get("key") or ""), _f(tp.get("tol", 0.0) or 0.0)
    if tp_tol < 0 or (tp_tol > 0) != bool(tp_key):
        raise ValueError("loop.project.throughput: tol >= 0, and a key exactly when tol > 0")
    return ProjectCfg(name=str(raw.get("name") or "nanopath"), base_sha=str(raw.get("base_sha") or ""),
                      runs_root=str(raw.get("runs_root") or ""), worktrees_root=str(raw.get("worktrees_root") or ""),
                      config_rel=str(raw.get("config_rel") or ""), python=str(raw.get("python") or ""),
                      aggregation=agg, axes=tuple(axes), target=tgt,
                      guard=guard_t, harm_h=harm, axis_exclude=ax_ex, budget=dict(budget), composite=comp,
                      throughput_key=tp_key, throughput_tol=tp_tol,
                      noise_floor_seeds=_floor_seeds(raw.get("noise_floor_seeds", NOISE_FLOOR_SEEDS)),
                      cell_tau2=_cell_tau2(raw.get("cell_tau2", "robust")))


def _cell_tau2(v: Any) -> str:
    if v not in ("robust", "mad_minus_mean"):
        raise ValueError(f"loop.project.cell_tau2: robust | mad_minus_mean, got {v!r}")
    return str(v)


def _floor_seeds(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"loop.project.noise_floor_seeds: an int >= 0, got {v!r}")
    return v


def parse_domain(raw: Mapping[str, Any] | DomainCfg | None) -> DomainCfg | None:
    if raw is None or isinstance(raw, DomainCfg):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError(f"loop.domain must be a mapping, got {type(raw).__name__}")
    if bad := set(raw) - {"prompt_dir", "rules"}:
        raise ValueError(f"loop.domain: unknown keys {sorted(bad)}")
    rules = raw.get("rules") or {}
    if not isinstance(rules, Mapping):
        raise ValueError("loop.domain.rules must be a mapping")
    return DomainCfg(str(raw.get("prompt_dir") or ""), {str(k): str(v) for k, v in rules.items()})


def _weights(p: ProjectCfg) -> dict[str, float]:
    if p.aggregation == "mean":
        return {a.name: 1.0 / len(p.axes) for a in p.axes}
    tot = sum(a.weight for a in p.axes)
    # weights that already sum to 1 are kept byte-exact (nanopath's explicit block == the literals)
    return {a.name: (a.weight if abs(tot - 1.0) <= 1e-12 else a.weight / tot) for a in p.axes}


def install(project: Mapping[str, Any] | ProjectCfg | None = None,
            domain: Mapping[str, Any] | DomainCfg | None = None) -> None:
    """Install a project (raw ``loop.project`` dict or :class:`ProjectCfg`) and / or a domain block (raw
    ``loop.domain`` dict or :class:`DomainCfg`). ``None`` for either leaves that part at the nanopath defaults.
    Always starts from the defaults (an install is not cumulative)."""
    p, d = parse_project(project), parse_domain(domain)
    s = _State()
    if p is not None:
        s.axes = tuple(a.name for a in p.axes)
        s.probe_keys = {a.name: a.key for a in p.axes}
        s.weights = _weights(p)
        s.names = {a.name: (a.label or a.name) for a in p.axes}
        s.target, s.guard, s.harm_h, s.axis_exclude = p.target, tuple(p.guard), dict(p.harm_h), tuple(p.axis_exclude)
        s.composite, s.throughput_key, s.throughput_tol = p.composite, p.throughput_key, p.throughput_tol
        s.noise_floor_seeds, s.cell_tau2 = p.noise_floor_seeds, p.cell_tau2
        s.project = p
    if d is not None:
        if d.prompt_dir:
            pd = Path(d.prompt_dir)
            s.prompt_dir = pd if pd.is_absolute() else REPO / pd
        s.rules = dict(d.rules)
        s.domain = d
    global _S
    _S = s


def reset() -> None:
    """Back to the nanopath defaults (tests)."""
    global _S
    _S = _State()


def prompt_path(name: str) -> Path:
    """``<name>.md`` (or ``name`` as given when it has a suffix): the domain overlay dir first, then loop/prompts."""
    fn = name if Path(name).suffix else f"{name}.md"
    if _S.prompt_dir is not None and (_S.prompt_dir / fn).is_file():
        return _S.prompt_dir / fn
    return PROMPTS / fn


def domain_dir() -> Path | None:
    """The installed domain overlay dir (``loop.domain.prompt_dir``), None when the project has none."""
    return _S.prompt_dir


def rule(key: str, default: str) -> str:
    """The domain text rule ``key`` (``loop.domain.rules``), else ``default`` (today's nanopath literal)."""
    return _S.rules.get(key, default)
