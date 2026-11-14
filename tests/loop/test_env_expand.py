"""Task specs read machine-specific values from the environment: ``${VAR}`` and ``${VAR:-default}``."""

from __future__ import annotations

from pathlib import Path

from kernel.model import expand_env, load_yaml

REPO = Path(__file__).resolve().parents[2]


def test_expand_env_set_default_and_unset(monkeypatch):
    monkeypatch.setenv("AL_TEST_SET", "x")
    monkeypatch.delenv("AL_TEST_UNSET", raising=False)
    assert expand_env("${AL_TEST_SET}/${AL_TEST_UNSET:-d}/${AL_TEST_UNSET}") == "x/d/${AL_TEST_UNSET}"


def test_nanobrain_spec_takes_its_paths_and_pools_from_the_environment(monkeypatch):
    monkeypatch.setenv("NANOBRAIN_REPO", "/srv/nb")
    monkeypatch.setenv("SLURM_PARTITION", "h100")
    raw = load_yaml(REPO / "tasks" / "nanobrain.yaml")
    assert raw["repo"] == "/srv/nb"
    assert {p["partition"] for p in raw["loop"]["slurm"]["pools"]} == {"h100"}
