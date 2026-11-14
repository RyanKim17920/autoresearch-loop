"""Package D (domain text): with NO overlay every prompt / brief / domain literal renders byte-identical to the
pre-multiproject text (golden: tests/fixtures/domain_golden/renders.json); P.rule() defaults equal the old literals;
a domain.prompt_dir overlay is picked up.  Regenerate the golden ONLY for an intended nanopath text change:
``DOMAIN_GOLDEN_WRITE=1 uv run pytest tests/loop/test_domain_text.py -k golden``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from loop.agents import briefs as BR
from loop.agents import researcher as RS
from loop.core import bold as BOLD

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "fixtures" / "domain_golden" / "renders.json"
PROMPTS = sorted(p.stem for p in (REPO / "loop" / "prompts").glob("*.md"))


def collect(ctx, store) -> dict[str, str]:
    """Every domain-carrying render with fixed inputs."""
    from loop.core import levers as LV
    out: dict[str, str] = {}
    for name in PROMPTS:
        out[f"prompt/{name}"] = BR.render(name)
    out["prompt/scout+values"] = BR.render("scout", task_id="T", date="2026-01-01", min_ideas=3, max_ideas=5, focus="",
                                           ledger="L", bank="B", lessons="x", record="r", constraints="c",
                                           families="f", cells="")
    for k in ("LEGACY_INIT", "NO_SWEEPS", "CARRIED_LESSON", "INIT_SWAP"):
        out[f"briefs/{k}"] = getattr(BR, k)
    out["briefs/LEVER_HELP"] = json.dumps(BR.LEVER_HELP, sort_keys=True)
    out["briefs/FAMILY_HELP"] = json.dumps(BR.FAMILY_HELP, sort_keys=True)
    out["briefs/guard_line"] = "\n".join(BR.guard_line(ctx))
    out["briefs/hard_constraints"] = BR.hard_constraints(ctx)
    out["briefs/family_lines"] = "\n".join(BR.family_lines(ctx, None))
    snap = SimpleNamespace(nodes={}, incumbent_id="n0", levers=[])
    for kind, kw in (("moonshot", {"ctx": ctx, "probe": "p"}), ("bank", {"ctx": ctx, "moonshot": True, "bank": []}),
                     ("literature", {}), ("probe", {"probe": "q"})):
        try:
            out[f"briefs/brief:{kind}"] = BR.brief(kind, store, snap, **kw)
        except Exception as exc:  # noqa: BLE001 - recorded: must raise identically after the change
            out[f"briefs/brief:{kind}"] = f"RAISES {type(exc).__name__}"
    fake = SimpleNamespace(
        compute=lambda v, s, c: {"multiplier": 2.0, "flop_fraction": 0.5, "keys": {"data.resolution": 224}},
        unrun=lambda v, s, c: [("dsA", 1), ("dsB", 2)], dataset_gaps=lambda v, s, c: [],
        external=lambda v, c: [], opt=lambda c, k: 0.01)
    with mock.patch.multiple(LV, **vars(fake)):
        out["briefs/structural_rows"] = "\n".join(BR.structural_rows(None, store, ctx))
        out["briefs/structural_block"] = BR.structural_block(None, store, ctx)
    out["researcher/analogy_problem"] = RS.analogy_problem({})
    out["researcher/budget_item"] = json.dumps(RS.budget_item(), sort_keys=True) if hasattr(RS, "budget_item") \
        else json.dumps(RS.BUDGET_ITEM, sort_keys=True)
    with mock.patch.object(RS.DB, "bank_lines", lambda c, s: ["dsX: 10 tiles"]):
        out["researcher/_bank_block"] = RS._bank_block(ctx, store)
    out["bold/TAXONOMY"] = BOLD.taxonomy_text()
    out["bold/CATEGORY_RX"] = json.dumps(BOLD.CATEGORY_RX, sort_keys=True)
    out["bold/INIT_RULE"] = BOLD.INIT_RULE
    out["bold/JUDGE_SYSTEM"] = BOLD.JUDGE_SYSTEM
    out["bold/KNOB_JUDGE_SYSTEM"] = BOLD.KNOB_JUDGE_SYSTEM
    out["bold/mix_text"] = "\n".join(BOLD.mix_text({"kinds": {"k": {"n": 1, "knob_only": 0.5, "cats": {}}}}))
    return out


def test_golden_renders(ctx, store):
    got = collect(ctx, store)
    if os.environ.get("DOMAIN_GOLDEN_WRITE"):
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True))
    want = json.loads(GOLDEN.read_text())
    assert sorted(got) == sorted(want)
    bad = [k for k in want if got[k] != want[k]]
    assert not bad, bad


# ------------------------------------------------------------------------------------------ rule() defaults
import re  # noqa: E402

import yaml  # noqa: E402

from loop import project as P  # noqa: E402

OVERLAY_DIR = REPO / "tasks" / "nanobrain" / "prompts"

#: accessor -> (rule key, the literal it returned before multiproject)
ACCESSORS = {
    BR.legacy_init: ("init_legacy", BR.LEGACY_INIT), BR.init_swap: ("init_swap", BR.INIT_SWAP),
    BR.carried_lesson: ("carried_lesson", BR.CARRIED_LESSON), BR.no_sweeps: ("no_sweeps", BR.NO_SWEEPS)}


@pytest.mark.parametrize("fn", list(ACCESSORS))
def test_accessor_default_is_old_literal_and_rule_overrides(fn):
    key, literal = ACCESSORS[fn]
    P.reset()
    assert fn() == literal
    P.install(None, {"rules": {key: "X"}})
    assert fn() == "X"


def test_old_literals_unchanged_text():
    """The defaults are the pre-change literals (spot-pinned beyond the golden: phrases the nanobrain rules replace)."""
    assert "non-pathology, non-GenBio source" in BR.LEGACY_INIT
    assert BR.GENBIO_BAN.startswith("- NO change may use GenBio data or models (GenBio-PathFM embeddings")
    assert BR.FIT_DATA == "charged calibration tiles (jpegs, site_rows)"
    assert BR.DATABANK_HEADER.startswith("DATA BANK (data_curation: set data.dataset_dir to one; TCGA tile datasets")
    assert "pathology-trained / GenBio checkpoint" in BOLD.__dict__["INIT_RULE"] or "no pathology-trained" in BOLD.INIT_RULE
    assert BOLD.taxonomy() == BOLD.TAXONOMY and BOLD.category_rx() == BOLD.CATEGORY_RX


def test_rules_override_every_site(ctx, store):
    P.install(None, {"rules": {
        "forbid_data": "FD", "forbid_fit_data": "FFD", "databank_header": "DBH: ", "taxonomy_pathology": "TP",
        "category_rx_pathology": "zzz", "analogy_field": "AF", "databank_pack_header": "\nDBP\n",
        "knob_scope": "KS", "bold_absolute": "BA", "struct_levers": "SL", "lever_data_curation": "LDC",
        "family_augmentation": "FA"}})
    assert BR.guard_line(ctx)[0] == "FD"
    assert "FD" in BR.hard_constraints(ctx)
    assert BOLD.taxonomy()["pathology"] == "TP" and "- **pathology**: TP" in BOLD.taxonomy_text()
    assert BOLD.category_rx()["pathology"] == "zzz" and BOLD.category_rx()["data"] == BOLD.CATEGORY_RX["data"]
    assert "not AF)" in RS.analogy_problem({})
    assert BR.lever_help()["data_curation"] == "LDC" and BR.family_help()["augmentation"] == "FA"
    with mock.patch.object(RS.DB, "bank_lines", lambda c, s: ["dsX"]):
        assert RS._bank_block(ctx, store).startswith("\nDBP\n")


def test_budget_item_follows_installed_axes():
    P.reset()
    assert RS.budget_item()["properties"]["axis"]["enum"] == list(P.AXES)
    assert RS.BUDGET_ITEM == RS.budget_item() and RS.MECHANISM_PREDICTION == RS.mechanism_prediction()
    assert RS.BOLD_PREDICTION["required"] == ["diagnostic", "direction", "axes"]
    P.install({"name": "t", "axes": [{"name": "a", "key": "ka", "weight": 1}, {"name": "b", "key": "kb", "weight": 1}],
               "target": "a", "guard": [], "harm_h": {}, "axis_exclude": []})
    assert RS.budget_item()["properties"]["axis"]["enum"] == ["a", "b"]
    assert RS.mechanism_prediction()["properties"]["budget"]["items"]["properties"]["axis"]["enum"] == ["a", "b"]
    assert RS.axes_example() == '{"a": 1, "b": 0}'
    P.reset()
    assert RS.axes_example() == '{"cls": 1, "rob": 0}'


# ---------------------------------------------------------------------------------------------- overlays
def test_overlay_picked_up_when_prompt_dir_set():
    P.reset()
    base = BR.render("scout", task_id="T")
    assert "pathology" in base and "brain-MRI" not in base
    P.install(None, {"prompt_dir": str(OVERLAY_DIR)})
    got = BR.render("scout", task_id="T")
    assert "brain-MRI" in got and "FOMO300K" in got and "TCGA" not in got and "{date}" in got
    assert BR.render("brief_literature", family="f") == (REPO / "loop/prompts/brief_literature.md").read_text().replace(
        "{family}", "f")                                   # no overlay file: falls back to loop/prompts
    P.reset()
    assert BR.render("scout", task_id="T") == base


def _fields(text: str) -> set[str]:
    return set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", text))


@pytest.mark.parametrize("path", sorted(OVERLAY_DIR.glob("*.md")), ids=lambda p: p.stem)
def test_overlay_keeps_placeholders_and_is_domain_clean(path):
    orig = (REPO / "loop" / "prompts" / path.name).read_text()
    new = path.read_text()
    assert _fields(new) == _fields(orig), _fields(orig) ^ _fields(new)
    assert not re.search(r"TCGA|GenBio|\bUNI\b|Virchow|stain|\bslide\b", new)


def test_overlays_cover_the_domain_prompts():
    names = {p.stem for p in OVERLAY_DIR.glob("*.md")}
    assert {"scout", "brief_bold", "ideation"} <= names


# ------------------------------------------------------------------- F3: nanobrain renders through the LIVE accessors
NB_YAML = REPO / "tasks" / "nanobrain.yaml"
#: nanopath vocabulary that must not reach a nanobrain session (HED is case-sensitive: "schedule" contains "hed")
LEAK = re.compile(r"(?i:patholog|TCGA|\btile|slide|stain|Virchow)|\bHED\b|\bUNI\b")
#: documented exceptions: none -- internal ids ('pathology' category) are relabelled on display (taxonomy_label_pathology)
LEAK_ALLOW: tuple[str, ...] = ()


def _nb_domain(merge_needs: bool = True) -> dict:
    dom = dict(yaml.safe_load(NB_YAML.read_text())["loop"]["domain"])
    rules = dict(dom.get("rules") or {})
    return {**dom, "rules": rules}


def live_renders(ctx, store) -> dict[str, str]:
    """Every domain-carrying render through the accessors the code calls (no raw module constants)."""
    from loop.core import ideation as ID
    out = collect_live(ctx, store)
    out["bold/taxonomy_text"] = BOLD.taxonomy_text()
    out["bold/category_rx"] = "\n".join(BOLD.category_rx().values())          # values only: the keys are internal ids
    with mock.patch.object(RS.DB, "bank_lines", lambda c, s: ["dsX: 10 images"]):
        out["researcher/_bank_block"] = RS._bank_block(ctx, store)
    out["bold/init_rule"] = P.rule("bold_absolute", BOLD.INIT_RULE)
    out["bold/judge_system"] = P.rule("judge_system", BOLD.JUDGE_SYSTEM)
    out["bold/knob_judge_system"] = P.rule("knob_judge_system", BOLD.KNOB_JUDGE_SYSTEM)
    out["bold/slate_tool"] = json.dumps(BOLD.slate_tool(ctx), sort_keys=True)
    out["bold/slate_section"] = BOLD.slate_section(ctx)
    out["bold/brief_head"] = BOLD.brief_head(ctx)
    out["ideation/kuhn"] = ID.KIND_TEXT["kuhn"].format(plateau="p", review="", distant=P.rule(
        "kuhn_distant", "pathology and self-supervised vision"))
    out["ideation/harm_header"] = P.rule("harm_blocked_header", "Harm-blocked (catastrophic robustness loss only; "
                                                                "the composite prices every axis)")
    return out


def collect_live(ctx, store) -> dict[str, str]:
    out = {k: v for k, v in collect(ctx, store).items() if not k.startswith(("bold/", "briefs/LEVER", "briefs/FAMILY"))
           and k not in {f"briefs/{n}" for n in ("LEGACY_INIT", "NO_SWEEPS", "CARRIED_LESSON", "INIT_SWAP")}}
    out.pop("researcher/_bank_block", None)
    out["briefs/init_legacy"], out["briefs/init_swap"] = BR.legacy_init(), BR.init_swap()
    out["briefs/carried_lesson"], out["briefs/no_sweeps"] = BR.carried_lesson(), BR.no_sweeps()
    out["briefs/lever_help"] = json.dumps(BR.lever_help(), sort_keys=True)
    out["briefs/family_help"] = json.dumps(BR.family_help(), sort_keys=True)
    out["bold/mix_text"] = "\n".join(BOLD.mix_text({"kinds": {"k": {"n": 1, "knob_only": 0.5, "cats": {}}}}))
    return out


def test_nanobrain_renders_have_no_nanopath_vocabulary(ctx, store):
    P.install(yaml.safe_load(NB_YAML.read_text())["loop"]["project"], _nb_domain())
    try:
        got = live_renders(ctx, store)
    finally:
        P.reset()
    assert any(k.startswith("prompt/") for k in got) and "bold/taxonomy_text" in got
    leaks = {}
    for k, v in got.items():
        text = v
        for ok in LEAK_ALLOW:
            text = text.replace(ok, "")
        if hits := sorted({m.group(0) for m in LEAK.finditer(text)}):
            leaks[k] = hits
    assert not leaks, leaks


def test_nanobrain_pretext_regex_does_not_swallow_masking():
    P.install(None, _nb_domain())
    try:
        rx = BOLD.category_rx()
    finally:
        P.reset()
    assert not re.search(rx["pretext"], "mask ratio", re.I) and re.search(rx["schedule"], "mask ratio", re.I)
    assert re.search(BOLD.CATEGORY_RX["pretext"], "mask ratio", re.I)          # the nanopath default is unchanged


def test_taxonomy_label_is_displayed_and_accepted_back():
    P.install(None, {"rules": {"taxonomy_label_pathology": "anatomy"}})
    try:
        assert "- **anatomy**:" in BOLD.taxonomy_text() and "**pathology**" not in BOLD.taxonomy_text()
        enum = BOLD.slate_tool(None)["input_schema"]["properties"]["candidates"]["items"]["properties"]["category"]["enum"]
        assert "anatomy" in enum and "pathology" not in enum
        assert BOLD.canon("Anatomy") == "pathology" and BOLD.canon("data") == "data"
    finally:
        P.reset()
    assert BOLD.label("pathology") == "pathology" and BOLD.slate_tool(None) is BOLD.SLATE_TOOL


def test_nanobrain_weights_policy_is_consistent():
    rules = _nb_domain()["rules"]
    for k in ("bold_absolute", "init_legacy", "init_swap", "forbid_data", "databank_forbid_msg"):
        assert re.search(r"brain-MRI", rules[k]) and re.search(r"natural-image", rules[k]), k


# ------------------------------------------------------------------------------ F3: researcher shell safety
@pytest.mark.parametrize("cmd", [
    "uv run python train.py", "cd x && uv sync", "uv pip install foo", "uvx ruff", "pip install torch",
    "pip3 install -U x", "python -m pip install x", ".venv/bin/pip install x", "poetry install", "FOO=1 uv run pytest"])
def test_researcher_refuses_package_managers(cmd):
    assert any(rx.search(cmd) for rx, _ in RS._BANNED), cmd


@pytest.mark.parametrize("cmd", [".venv/bin/python train.py --smoke", "python -c 'print(1)'", "cat pyproject.toml",
                                 "ls .venv/bin", "echo pipeline"])
def test_researcher_allows_direct_python_and_reads(cmd):
    assert not [w for rx, w in RS._BANNED if rx.search(cmd) and "package" in w], cmd


def test_researcher_bash_env_pins_uv_and_pip(tmp_path):
    t = RS.WorktreeTools.__new__(RS.WorktreeTools)
    t.root, t.bash_timeout_s, t.locked, t.dirty = tmp_path, 20.0, (), False
    t._confined = lambda c: ""
    t._locked_status = lambda: None
    t._restore_locked = lambda b: []
    out = t.t_bash("env | rg '^(UV_|PIP_)' | sort")
    for k, v in RS.SAFE_ENV.items():
        assert f"{k}={v}" in out, out
    assert "REFUSED" in t.t_bash("uv run python -c 1") and "REFUSED" in t.t_bash("pip install x")
