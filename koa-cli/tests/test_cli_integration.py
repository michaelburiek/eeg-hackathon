from __future__ import annotations

import json
import subprocess
from pathlib import Path

import koa_cli.cli as cli_mod
from koa_cli.cli import main


def _make_config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("user: test\nhost: koa.example.edu\n", encoding="utf-8")
    return cfg


def test_main_jobs_json_outputs_machine_readable_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cfg = _make_config_file(tmp_path)
    monkeypatch.setattr(
        "koa_cli.cli.list_jobs",
        lambda config: "JOBID|NAME|STATE\n123456|train|RUNNING",
    )

    rc = main(["--config", str(cfg), "jobs", "--json"])

    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data == [{"JOBID": "123456", "NAME": "train", "STATE": "RUNNING"}]


def test_main_efficiency_json_outputs_parsed_report(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cfg = _make_config_file(tmp_path)
    monkeypatch.setattr(
        "koa_cli.cli.job_efficiency",
        lambda config, job_id: (
            "Job ID: 123456\n"
            "State: COMPLETED (exit code 0)\n"
            "CPU Efficiency: 92.10% of 01:00:00 core-walltime"
        ),
    )

    rc = main(["--config", str(cfg), "efficiency", "123456", "--json"])

    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["job_id"] == "123456"
    assert data["state"] == "COMPLETED (exit code 0)"
    assert data["cpu_efficiency"] == "92.10% of 01:00:00 core-walltime"


def test_main_run_chains_storage_sync_and_submit(tmp_path: Path, monkeypatch, capsys) -> None:
    cfg = _make_config_file(tmp_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    script = project_dir / "train.slurm"
    script.write_text("#!/bin/bash\n", encoding="utf-8")

    events: list[object] = []

    monkeypatch.setattr(
        "koa_cli.cli.ensure_project_directories",
        lambda config, project_name: events.append(("storage", project_name))
        or ["/remote/code/demo", "/remote/data/demo/train/results"],
    )
    monkeypatch.setattr(
        "koa_cli.cli._storage_link",
        lambda config, project_name: events.append(("link", project_name)),
    )
    monkeypatch.setattr(
        "koa_cli.cli.sync_directory_to_remote",
        lambda config, local_dir, remote_dir, excludes, dry_run=False: events.append(
            ("sync", str(local_dir), str(remote_dir), dry_run, ".git/" in excludes)
        )
        or "train.py\n",
    )
    monkeypatch.setattr(
        "koa_cli.cli.submit_job",
        lambda config, project_name, local_job_script, **kwargs: events.append(
            ("submit", project_name, local_job_script.name, kwargs["sbatch_args"])
        )
        or "123456",
    )
    monkeypatch.setattr(
        "koa_cli.cli.record_job",
        lambda job_id, project, script_path: events.append(
            ("record", job_id, project, script_path)
        ),
    )

    rc = main(
        [
            "--config",
            str(cfg),
            "run",
            str(script),
            "--project",
            "demo",
            "--path",
            str(project_dir),
            "--link-storage",
            "--time",
            "1:00:00",
        ]
    )

    assert rc == 0
    assert events == [
        ("storage", "demo"),
        ("link", "demo"),
        ("sync", str(project_dir.resolve()), "~/koa-jobs/demo", False, True),
        ("submit", "demo", "train.slurm", ["--time", "1:00:00"]),
        ("record", "123456", "demo", str(script)),
    ]
    out = capsys.readouterr().out
    # The submit summary is now rendered as a Rich Panel; check for the
    # job ID and a next-step hint rather than the old plain-text prefix.
    assert "123456" in out
    assert "koa logs 123456" in out


def test_main_submit_watch_enters_live_watch(tmp_path: Path, monkeypatch) -> None:
    cfg = _make_config_file(tmp_path)
    script = tmp_path / "train.slurm"
    script.write_text("#!/bin/bash\n", encoding="utf-8")

    events: list[object] = []

    monkeypatch.setattr(
        "koa_cli.cli.submit_job",
        lambda config, project_name, local_job_script, **kwargs: events.append(
            ("submit", project_name, local_job_script.name)
        )
        or "123456",
    )
    monkeypatch.setattr(
        "koa_cli.cli.record_job",
        lambda job_id, project, script_path: events.append(
            ("record", job_id, project, script_path)
        ),
    )
    monkeypatch.setattr(
        "koa_cli.cli._watch_job",
        lambda config, project_name, job_id, interval: events.append(
            ("watch", project_name, job_id, interval)
        )
        or 0,
    )

    rc = main(
        [
            "--config",
            str(cfg),
            "submit",
            str(script),
            "--project",
            "demo",
            "--watch",
            "--watch-interval",
            "2",
        ]
    )

    assert rc == 0
    assert events == [
        ("submit", "demo", "train.slurm"),
        ("record", "123456", "demo", str(script)),
        ("watch", "demo", "123456", 2.0),
    ]


def test_main_watch_uses_latest_job_when_omitted(tmp_path: Path, monkeypatch) -> None:
    cfg = _make_config_file(tmp_path)
    monkeypatch.setattr("koa_cli.cli.get_latest_job_id", lambda project=None: "123456")
    seen: list[tuple[str, str, float]] = []
    monkeypatch.setattr(
        "koa_cli.cli._watch_job",
        lambda config, project_name, job_id, interval: seen.append(
            (project_name, job_id, interval)
        )
        or 0,
    )

    rc = main(["--config", str(cfg), "watch", "--project", "demo", "--interval", "3"])

    assert rc == 0
    assert seen == [("demo", "123456", 3.0)]


def test_main_logs_no_follow_reports_pending_job_cleanly(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cfg = _make_config_file(tmp_path)
    monkeypatch.setattr(
        "koa_cli.cli.get_job_snapshot",
        lambda config, job_id: type(
            "Snapshot",
            (),
            {"state": "PENDING", "reason": "Priority"},
        )(),
    )
    called = {"streamed": False}
    monkeypatch.setattr(
        "koa_cli.cli.stream_job_logs",
        lambda *args, **kwargs: called.__setitem__("streamed", True),
    )

    rc = main(["--config", str(cfg), "logs", "123456", "--no-follow"])

    assert rc == 0
    assert called["streamed"] is False
    assert "still pending (Priority)" in capsys.readouterr().out


def test_main_submit_passes_wandb_env_to_submit_job(tmp_path: Path, monkeypatch) -> None:
    cfg = _make_config_file(tmp_path)
    script = tmp_path / "train.slurm"
    script.write_text("#!/bin/bash\n", encoding="utf-8")

    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "koa_cli.cli._require_remote_wandb_api_key",
        lambda config, project_name, wandb_env: seen.setdefault("checked", dict(wandb_env)),
    )
    def fake_submit_job(config, project_name, local_job_script, **kwargs):
        seen["job_env"] = kwargs["job_env"]
        return "123456"

    monkeypatch.setattr("koa_cli.cli.submit_job", fake_submit_job)
    monkeypatch.setattr("koa_cli.cli.record_job", lambda *args, **kwargs: None)

    rc = main(
        [
            "--config",
            str(cfg),
            "submit",
            str(script),
            "--project",
            "demo",
            "--partition",
            "kill-shared",
            "--wandb",
            "--wandb-group",
            "smoke",
        ]
    )

    assert rc == 0
    assert seen["checked"]["WANDB_PROJECT"] == "demo"
    assert seen["job_env"]["WANDB_RUN_GROUP"] == "smoke"
    assert seen["job_env"]["WANDB_TAGS"] == "koa,slurm,kill-shared"


def test_main_wandb_sync_requires_api_key(tmp_path: Path, capsys) -> None:
    cfg = _make_config_file(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=secret\n", encoding="utf-8")

    rc = main(["--config", str(cfg), "wandb", "sync", "--env-file", str(env_file)])

    assert rc == 1
    assert "WANDB_API_KEY is missing" in capsys.readouterr().out


def test_sync_env_file_uses_tilde_safe_ssh_commands(tmp_path: Path, monkeypatch) -> None:
    cfg = _make_config_file(tmp_path)
    config = cli_mod.load_config(cfg)
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=secret\n", encoding="utf-8")

    commands: list[list[str]] = []
    copied: list[tuple[Path, Path]] = []

    monkeypatch.setattr(
        "koa_cli.cli.run_ssh",
        lambda config, remote_command, **kwargs: commands.append(list(remote_command))
        or subprocess.CompletedProcess(["ssh"], 0, "", ""),
    )
    monkeypatch.setattr(
        "koa_cli.cli.copy_to_remote",
        lambda config, local_path, remote_path, **kwargs: copied.append((local_path, remote_path)),
    )

    cli_mod._sync_env_file(config, "demo", env_file)

    assert commands == [
        ["mkdir", "-p", "~/koa-jobs/demo"],
        ["chmod", "600", "~/koa-jobs/demo/.env"],
    ]
    assert copied == [(env_file, Path("~/koa-jobs/demo/.env"))]


def test_remote_env_has_key_checks_remote_path_without_shell_quoting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _make_config_file(tmp_path)
    config = cli_mod.load_config(cfg)
    seen: list[list[str]] = []

    def fake_run_ssh(config, remote_command, **kwargs):
        seen.append(list(remote_command))
        return subprocess.CompletedProcess(["ssh"], 0, "", "")

    monkeypatch.setattr("koa_cli.cli.run_ssh", fake_run_ssh)

    assert (
        cli_mod._remote_env_has_key(config, Path("~/koa-jobs/demo/.env"), "WANDB_API_KEY")
        is True
    )
    assert seen == [
        [
            "bash",
            "-lc",
            (
                "test -f ~/koa-jobs/demo/.env && "
                "grep -q '^[[:space:]]*WANDB_API_KEY=' ~/koa-jobs/demo/.env"
            ),
        ]
    ]
