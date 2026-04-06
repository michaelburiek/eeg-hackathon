"""Smoke tests for the CLI argument parser and input-validation logic.

These tests do NOT connect to any cluster — they only verify that the
argument parser, validation helpers, and in-process logic work correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from koa_cli.cli import _build_parser, main

# ---------------------------------------------------------------------------
# Parser: basic flag parsing
# ---------------------------------------------------------------------------


def test_version_flag(capsys) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "0.3.0" in out


def test_sync_dry_run_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["sync", "--dry-run"])
    assert args.dry_run is True
    assert args.command == "sync"


def test_submit_gpus_and_gres_are_separate_args() -> None:
    parser = _build_parser()
    args = parser.parse_args(["submit", "train.slurm", "--gpus", "2"])
    assert args.gpus == 2
    assert args.gres is None


def test_submit_no_auto_gpu_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["submit", "train.slurm", "--no-auto-gpu"])
    assert args.no_auto_gpu is True


def test_submit_watch_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args(["submit", "train.slurm", "--watch", "--watch-interval", "2.5"])
    assert args.watch is True
    assert args.watch_interval == 2.5


def test_submit_wandb_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "submit",
            "train.slurm",
            "--wandb",
            "--wandb-project",
            "brain-decoder",
            "--wandb-group",
            "smoke",
            "--wandb-tags",
            "koa,eeg",
        ]
    )
    assert args.wandb is True
    assert args.wandb_project == "brain-decoder"
    assert args.wandb_group == "smoke"
    assert args.wandb_tags == "koa,eeg"


def test_results_pull_latest_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["results", "pull", "--latest"])
    assert args.latest is True
    assert args.job_id is None


def test_logs_no_follow_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["logs", "123456", "--no-follow"])
    assert args.no_follow is True
    assert args.job_id == "123456"


def test_watch_command_parsed() -> None:
    parser = _build_parser()
    args = parser.parse_args(["watch", "123456", "--interval", "3"])
    assert args.command == "watch"
    assert args.job_id == "123456"
    assert args.interval == 3


def test_wandb_command_parsed() -> None:
    parser = _build_parser()
    args = parser.parse_args(["wandb", "check", "--env-file", ".env"])
    assert args.command == "wandb"
    assert args.wandb_command == "check"


def test_jobs_json_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["jobs", "--json"])
    assert args.command == "jobs"
    assert args.json is True


def test_efficiency_json_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["efficiency", "--json"])
    assert args.command == "efficiency"
    assert args.json is True


def test_history_command_parsed() -> None:
    parser = _build_parser()
    args = parser.parse_args(["history", "--limit", "5"])
    assert args.command == "history"
    assert args.limit == 5


def test_run_command_parsed() -> None:
    parser = _build_parser()
    args = parser.parse_args(["run", "train.slurm", "--path", ".", "--link-storage"])
    assert args.command == "run"
    assert args.path == Path(".")
    assert args.link_storage is True


# ---------------------------------------------------------------------------
# main(): --gpus / --gres conflict
# ---------------------------------------------------------------------------


def _make_config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("user: test\nhost: koa.example.edu\n", encoding="utf-8")
    return cfg


def test_main_submit_gpus_gres_conflict(tmp_path: Path) -> None:
    cfg = _make_config_file(tmp_path)
    script = tmp_path / "train.slurm"
    script.touch()
    rc = main(
        [
            "--config", str(cfg),
            "submit", str(script),
            "--gpus", "2",
            "--gres", "gpu:a100:2",
        ]
    )
    assert rc == 1


# ---------------------------------------------------------------------------
# main(): cancel with invalid job ID
# ---------------------------------------------------------------------------


def test_main_cancel_invalid_job_id(tmp_path: Path) -> None:
    cfg = _make_config_file(tmp_path)
    rc = main(["--config", str(cfg), "cancel", "not-a-number"])
    assert rc == 1


# ---------------------------------------------------------------------------
# main(): submit with invalid time format
# ---------------------------------------------------------------------------


def test_main_submit_invalid_time(tmp_path: Path) -> None:
    cfg = _make_config_file(tmp_path)
    script = tmp_path / "train.slurm"
    script.touch()
    rc = main(["--config", str(cfg), "submit", str(script), "--time", "2hours"])
    assert rc == 1


# ---------------------------------------------------------------------------
# main(): results pull requires job_id or --latest
# ---------------------------------------------------------------------------


def test_main_results_pull_no_id_no_latest(tmp_path: Path) -> None:
    cfg = _make_config_file(tmp_path)
    rc = main(["--config", str(cfg), "results", "pull"])
    assert rc == 1


# ---------------------------------------------------------------------------
# main(): history command (local, no cluster)
# ---------------------------------------------------------------------------


def test_main_history_empty(tmp_path: Path, monkeypatch) -> None:
    import koa_cli.history as history_mod

    monkeypatch.setattr(history_mod, "HISTORY_PATH", tmp_path / "history.json")
    rc = main(["history"])
    assert rc == 0


def test_main_history_shows_records(tmp_path: Path, monkeypatch) -> None:
    import koa_cli.history as history_mod

    monkeypatch.setattr(history_mod, "HISTORY_PATH", tmp_path / "history.json")
    history_mod.record_job("999", "my-project", "train.slurm")
    rc = main(["history"])
    assert rc == 0
