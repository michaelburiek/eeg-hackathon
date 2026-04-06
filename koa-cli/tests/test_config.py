from __future__ import annotations

from pathlib import Path

import pytest

from koa_cli.config import load_config, validate_slurm_time

# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_applies_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("user: demo\nhost: koa.example.edu\n", encoding="utf-8")

    config = load_config(config_file)

    assert config.user == "demo"
    assert config.host == "koa.example.edu"
    assert config.remote_workdir == Path("~/koa-jobs")
    assert config.remote_data_dir == Path("/mnt/lustre/koa/scratch/demo/koa-jobs")
    # New optional fields should have their defaults.
    assert config.default_partition == "gpu"
    assert config.default_gpu == "rtx2080ti"
    assert config.default_time is None
    assert config.default_results_limit == 20


def test_load_config_custom_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    # Note: PyYAML 1.1 parses bare `2:00:00` as integer 7200 (seconds).
    # We quote it here to preserve the HH:MM:SS string form.
    config_file.write_text(
        "user: alice\nhost: koa.example.edu\n"
        "default_partition: gpu-shared\ndefault_gpu: a100\n"
        "default_time: '2:00:00'\ndefault_results_limit: 50\n",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.default_partition == "gpu-shared"
    assert config.default_gpu == "a100"
    assert config.default_time == "2:00:00"
    assert config.default_results_limit == 50


def test_load_config_env_overrides(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("user: demo\nhost: koa.example.edu\n", encoding="utf-8")
    monkeypatch.setenv("KOA_USER", "env-user")
    monkeypatch.setenv("KOA_DEFAULT_PARTITION", "env-partition")

    config = load_config(config_file)
    assert config.user == "env-user"
    assert config.default_partition == "env-partition"


def test_load_config_project_local_override(tmp_path: Path, monkeypatch) -> None:
    """A .koa.yaml in the project dir should override global config fields."""
    global_cfg = tmp_path / "config.yaml"
    global_cfg.write_text("user: demo\nhost: koa.example.edu\n", encoding="utf-8")

    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    local_cfg = project_dir / ".koa.yaml"
    local_cfg.write_text("default_partition: local-override\n", encoding="utf-8")

    config = load_config(global_cfg, project_dir=project_dir)
    assert config.default_partition == "local-override"


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_missing_required_keys(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("user: demo\n", encoding="utf-8")  # missing host
    with pytest.raises(ValueError, match="Missing required config keys"):
        load_config(config_file)


def test_load_config_invalid_time_format(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "user: demo\nhost: koa.example.edu\ndefault_time: 2hours\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid Slurm time format"):
        load_config(config_file)


# ---------------------------------------------------------------------------
# project_name validation
# ---------------------------------------------------------------------------


def test_project_name_validation(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("user: demo\nhost: koa.example.edu\n", encoding="utf-8")
    config = load_config(config_file)

    assert config.project_workdir("repo-name") == Path("~/koa-jobs/repo-name")
    assert config.project_workdir("my.project_1") == Path("~/koa-jobs/my.project_1")

    for bad in ["../bad", "..", ".", "", "bad/slash", "bad space"]:
        with pytest.raises(ValueError):
            config.project_workdir(bad)


# ---------------------------------------------------------------------------
# validate_slurm_time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["60", "10:00", "1:00:00", "1-00", "1-12:30", "2-00:00:00"],
)
def test_validate_slurm_time_valid(value: str) -> None:
    assert validate_slurm_time(value) == value


@pytest.mark.parametrize(
    "value",
    ["2hours", "1d", "abc", "1:2:3:4", ""],
)
def test_validate_slurm_time_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid Slurm time format"):
        validate_slurm_time(value)
