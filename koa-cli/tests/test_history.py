from __future__ import annotations

from pathlib import Path

import pytest

import koa_cli.history as history_mod


@pytest.fixture(autouse=True)
def _tmp_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the history file to a temp location for each test."""
    monkeypatch.setattr(history_mod, "HISTORY_PATH", tmp_path / "history.json")


def test_record_and_list(tmp_path: Path) -> None:
    history_mod.record_job("111", "proj-a", "train.slurm")
    history_mod.record_job("222", "proj-b", "eval.slurm")

    records = history_mod.list_history()
    assert len(records) == 2
    # Newest first
    assert records[0]["job_id"] == "222"
    assert records[1]["job_id"] == "111"


def test_list_filtered_by_project() -> None:
    history_mod.record_job("111", "proj-a", "train.slurm")
    history_mod.record_job("222", "proj-b", "eval.slurm")
    history_mod.record_job("333", "proj-a", "train2.slurm")

    records = history_mod.list_history(project="proj-a")
    assert len(records) == 2
    assert all(r["project"] == "proj-a" for r in records)


def test_get_latest_job_id() -> None:
    history_mod.record_job("100", "proj-x", "a.slurm")
    history_mod.record_job("200", "proj-x", "b.slurm")
    assert history_mod.get_latest_job_id(project="proj-x") == "200"


def test_get_latest_job_id_returns_none_when_empty() -> None:
    assert history_mod.get_latest_job_id() is None


def test_list_respects_limit() -> None:
    for i in range(10):
        history_mod.record_job(str(i), "proj", "script.slurm")
    records = history_mod.list_history(limit=3)
    assert len(records) == 3
    assert records[0]["job_id"] == "9"


def test_history_file_is_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "sub" / "history.json"
    monkeypatch.setattr(history_mod, "HISTORY_PATH", path)
    history_mod.record_job("1", "p", "s.slurm")
    assert path.exists()
