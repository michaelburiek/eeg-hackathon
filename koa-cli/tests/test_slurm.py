from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from koa_cli.slurm import (
    _normalize_gpu_token,
    get_available_gpus,
    get_job_output_path,
    get_job_snapshot,
    get_latest_result_dir,
    list_result_dirs,
    parse_gpu_count_from_script,
    script_requests_gpu_resources,
    select_best_gpu,
    submit_job,
    validate_job_id,
)

# ---------------------------------------------------------------------------
# parse_gpu_count_from_script
# ---------------------------------------------------------------------------


def test_parse_gpu_count_from_script(tmp_path: Path) -> None:
    script = tmp_path / "job.slurm"
    script.write_text("#SBATCH --gres=gpu:nvidia_h100:2\n", encoding="utf-8")
    assert parse_gpu_count_from_script(script) == 2


def test_parse_gpu_count_returns_none_when_no_gres(tmp_path: Path) -> None:
    script = tmp_path / "job.slurm"
    script.write_text("#!/bin/bash\necho hello\n", encoding="utf-8")
    assert parse_gpu_count_from_script(script) is None


def test_parse_gpu_count_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert parse_gpu_count_from_script(tmp_path / "missing.slurm") is None


def test_parse_gpu_count_with_inline_gres(tmp_path: Path) -> None:
    script = tmp_path / "job.slurm"
    script.write_text("#SBATCH --gres gpu:4\n", encoding="utf-8")
    assert parse_gpu_count_from_script(script) == 4


@pytest.mark.parametrize(
    "line",
    [
        "#SBATCH --gres=gpu:nvidia_h100:2\n",
        "#SBATCH --gpus=2\n",
        "#SBATCH --gpus-per-node 2\n",
        "#SBATCH --gpus-per-task=1\n",
    ],
)
def test_script_requests_gpu_resources_detects_supported_sbatch_flags(
    tmp_path: Path,
    line: str,
) -> None:
    script = tmp_path / "job.slurm"
    script.write_text(line, encoding="utf-8")
    assert script_requests_gpu_resources(script) is True


# ---------------------------------------------------------------------------
# select_best_gpu (mocked)
# ---------------------------------------------------------------------------


def test_select_best_gpu_uses_fallback_when_no_gpu_info(monkeypatch) -> None:
    monkeypatch.setattr(
        "koa_cli.slurm.get_available_gpus",
        lambda config, partition: {},
    )

    class FakeConfig:
        default_gpu = "rtx2080ti"

    assert select_best_gpu(FakeConfig(), "kill-shared") == "rtx2080ti"


def test_select_best_gpu_picks_highest_priority(monkeypatch) -> None:
    monkeypatch.setattr(
        "koa_cli.slurm.get_available_gpus",
        lambda config, partition: {"rtx2080ti": 4, "a100": 2},
    )

    class FakeConfig:
        default_gpu = "rtx2080ti"

    # a100 (score 90) > rtx2080ti (score 50)
    assert select_best_gpu(FakeConfig(), "kill-shared") == "a100"


def test_select_best_gpu_uses_name_map(monkeypatch) -> None:
    monkeypatch.setattr(
        "koa_cli.slurm.get_available_gpus",
        lambda config, partition: {"nvidiah100": 1},
    )

    class FakeConfig:
        default_gpu = "rtx2080ti"

    result = select_best_gpu(FakeConfig(), "kill-shared")
    assert result == "nvidia_h100"


# ---------------------------------------------------------------------------
# validate_job_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("job_id", ["123456", "1", "9999999", "123456_7", "123456.0", "123456_7.0"])
def test_validate_job_id_valid(job_id: str) -> None:
    assert validate_job_id(job_id) == job_id


@pytest.mark.parametrize("job_id", ["abc", "12.a", "", "job-123", "123 456", "123_", "_7"])
def test_validate_job_id_invalid(job_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid job ID"):
        validate_job_id(job_id)


def test_list_result_dirs_uses_mtime_sorting(monkeypatch) -> None:
    seen: dict[str, object] = {}

    # list_result_dirs now calls run_remote_shell (not run_ssh directly),
    # so we patch at that level to avoid needing a full Config stub.
    def fake_run_remote_shell(config, script, **kwargs):
        seen["script"] = script
        seen["kwargs"] = kwargs
        return SimpleNamespace(stdout="/remote/results/222\n/remote/results/111\n")

    monkeypatch.setattr("koa_cli.slurm.run_remote_shell", fake_run_remote_shell)

    class FakeConfig:
        def project_results_dir(self, project_name: str, kind: str) -> Path:
            assert project_name == "demo"
            assert kind == "train"
            return Path("/remote/results")

    rows = list_result_dirs(FakeConfig(), "demo", "train", limit=2)

    assert rows == ["/remote/results/222", "/remote/results/111"]
    assert "-printf '%T@\\t%p\\n'" in seen["script"]
    assert "sort -nr" in seen["script"]


def test_get_latest_result_dir_returns_first_entry(monkeypatch) -> None:
    monkeypatch.setattr(
        "koa_cli.slurm.list_result_dirs",
        lambda config, project_name, kind, limit=1: ["/remote/results/999"] if limit == 1 else [],
    )

    assert get_latest_result_dir(object(), "demo", "train") == "/remote/results/999"


def test_normalize_gpu_token_handles_koa_style_names() -> None:
    assert _normalize_gpu_token("NV-RTX-A4000") == "nvrtxa4000"
    assert _normalize_gpu_token("nvidia_a30_2g.12gb") == "nvidiaa302g12gb"


def test_get_available_gpus_parses_koa_names(monkeypatch) -> None:
    def fake_run_ssh(config, command, **kwargs):
        return SimpleNamespace(
            stdout=(
                "gpu:NV-A30:2|idle|4\n"
                "gpu:NV-RTX-A4000:10|mixed|1\n"
                "gpu:nvidia_a30_2g.12gb:4|alloc|1\n"
            )
        )

    monkeypatch.setattr("koa_cli.slurm.run_ssh", fake_run_ssh)

    rows = get_available_gpus(object(), "gpu")

    assert rows == {"nva30": 4, "nvrtxa4000": 1}


def test_get_available_gpus_accepts_mix_dash_state(monkeypatch) -> None:
    def fake_run_ssh(config, command, **kwargs):
        return SimpleNamespace(stdout="gpu:NV-A30:2|mix-|3\n")

    monkeypatch.setattr("koa_cli.slurm.run_ssh", fake_run_ssh)

    rows = get_available_gpus(object(), "gpu")

    assert rows == {"nva30": 3}


def test_get_job_snapshot_prefers_live_squeue(monkeypatch) -> None:
    def fake_run_ssh(config, command, **kwargs):
        if command[:3] == ["squeue", "-h", "-j"]:
            return SimpleNamespace(
                stdout="123456|koa-smoke|PENDING|0:00|5:00|1|(Priority)|kill-shared\n"
            )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("koa_cli.slurm.run_ssh", fake_run_ssh)

    snapshot = get_job_snapshot(object(), "123456")

    assert snapshot is not None
    assert snapshot.state == "PENDING"
    assert snapshot.reason == "Priority"
    assert snapshot.partition == "kill-shared"


def test_get_job_snapshot_falls_back_to_sacct(monkeypatch) -> None:
    def fake_run_ssh(config, command, **kwargs):
        if command[:3] == ["squeue", "-h", "-j"]:
            return SimpleNamespace(stdout="")
        return SimpleNamespace(
            stdout="123456|koa-smoke|COMPLETED|0:04:52|0:05:00|1|node001|kill-shared\n"
        )

    monkeypatch.setattr("koa_cli.slurm.run_ssh", fake_run_ssh)

    snapshot = get_job_snapshot(object(), "123456")

    assert snapshot is not None
    assert snapshot.state == "COMPLETED"
    assert snapshot.location == "node001"
    assert snapshot.is_terminal is True


def test_get_job_output_path_falls_back_to_project_dir(monkeypatch) -> None:
    monkeypatch.setattr(
        "koa_cli.slurm.run_ssh",
        lambda config, command, **kwargs: SimpleNamespace(stdout=""),
    )

    output_path = get_job_output_path(_FakeConfig(), "demo", "123456")

    assert output_path == "/remote/workdir/demo/slurm-123456.out"


class _FakeConfig:
    user = "demo"
    host = "koa.example.edu"
    login = "demo@koa.example.edu"
    default_partition = "gpu"
    default_gpu = "rtx2080ti"
    default_time = None
    remote_workdir = Path("/remote/workdir")

    def project_workdir(self, project_name: str) -> Path:
        return self.remote_workdir / project_name

    def project_data_dir(self, project_name: str) -> Path:
        return Path("/remote/data") / project_name


def test_submit_job_exports_distinct_koa_paths(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "job.slurm"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr("koa_cli.slurm.copy_to_remote", lambda *args, **kwargs: None)
    monkeypatch.setattr("koa_cli.slurm.ensure_project_directories", lambda *args, **kwargs: [])

    def fake_run_ssh(config, command, **kwargs):
        calls.append(list(command))
        if command and command[0] == "env":
            return SimpleNamespace(stdout="Submitted batch job 123456")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("koa_cli.slurm.run_ssh", fake_run_ssh)

    job_id = submit_job(_FakeConfig(), "demo", script)

    assert job_id == "123456"
    env_command = next(command for command in calls if command and command[0] == "env")
    assert "KOA_PROJECT_DIR=/remote/workdir/demo" in env_command
    assert "KOA_REMOTE_WORKDIR=/remote/workdir" in env_command
    assert "KOA_REMOTE_DATA_DIR=/remote/data/demo" in env_command


def test_submit_job_exports_job_env_and_precreates_wandb_dirs(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "job.slurm"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr("koa_cli.slurm.copy_to_remote", lambda *args, **kwargs: None)
    monkeypatch.setattr("koa_cli.slurm.ensure_project_directories", lambda *args, **kwargs: [])

    def fake_run_ssh(config, command, **kwargs):
        calls.append(list(command))
        if command and command[0] == "env":
            return SimpleNamespace(stdout="Submitted batch job 123456")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("koa_cli.slurm.run_ssh", fake_run_ssh)

    submit_job(
        _FakeConfig(),
        "demo",
        script,
        job_env={
            "WANDB_PROJECT": "demo",
            "WANDB_DIR": "/remote/data/demo/wandb/runs",
            "WANDB_CACHE_DIR": "/remote/data/demo/wandb/.cache",
        },
    )

    mkdir_calls = [command for command in calls if command[:2] == ["mkdir", "-p"]]
    assert ["mkdir", "-p", "/remote/data/demo/wandb/runs"] in mkdir_calls
    assert ["mkdir", "-p", "/remote/data/demo/wandb/.cache"] in mkdir_calls
    env_command = next(command for command in calls if command and command[0] == "env")
    assert "WANDB_PROJECT=demo" in env_command
    assert "WANDB_DIR=/remote/data/demo/wandb/runs" in env_command


@pytest.mark.parametrize(
    "script_line",
    [
        "#SBATCH --gpus=2\n",
        "#SBATCH --gpus-per-node=2\n",
    ],
)
def test_submit_job_skips_auto_gpu_when_script_already_requests_gpus(
    tmp_path: Path,
    monkeypatch,
    script_line: str,
) -> None:
    script = tmp_path / "job.slurm"
    script.write_text(script_line, encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr("koa_cli.slurm.copy_to_remote", lambda *args, **kwargs: None)
    monkeypatch.setattr("koa_cli.slurm.ensure_project_directories", lambda *args, **kwargs: [])
    monkeypatch.setattr("koa_cli.slurm.select_best_gpu", lambda config, partition: "nvidia_h100")

    def fake_run_ssh(config, command, **kwargs):
        calls.append(list(command))
        if command and command[0] == "env":
            return SimpleNamespace(stdout="Submitted batch job 123456")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("koa_cli.slurm.run_ssh", fake_run_ssh)

    submit_job(_FakeConfig(), "demo", script, auto_gpu=True)

    env_command = next(command for command in calls if command and command[0] == "env")
    assert not any(arg.startswith("--gres=") or arg == "--gres" for arg in env_command)
