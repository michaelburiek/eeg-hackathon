from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from koa_cli.ssh import _quote_remote_part, run_ssh, sync_directory_to_remote


class _FakeConfig:
    login = "demo@koa.example.edu"
    identity_file = None
    proxy_command = None


def test_sync_directory_to_remote_keeps_excluded_remote_paths(monkeypatch, tmp_path: Path) -> None:
    local_dir = tmp_path / "project"
    local_dir.mkdir()
    (local_dir / "train.py").write_text("print('hi')\n", encoding="utf-8")

    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "koa_cli.ssh.run_ssh",
        lambda config, command: seen.setdefault("mkdir", command),
    )

    def fake_run(command, **kwargs):
        seen["rsync"] = command
        return SimpleNamespace(returncode=0, stdout="train.py\n", stderr="")

    monkeypatch.setattr("koa_cli.ssh.subprocess.run", fake_run)

    output = sync_directory_to_remote(
        _FakeConfig(),
        local_dir,
        Path("~/koa-jobs/demo"),
        excludes=["train/results/"],
    )

    assert output == "train.py\n"
    rsync_command = seen["rsync"]
    assert "--delete" in rsync_command
    assert "--delete-excluded" not in rsync_command


def test_quote_remote_part_preserves_tilde_expansion_for_paths() -> None:
    assert _quote_remote_part("~/koa-jobs/demo") == "~/koa-jobs/demo"
    assert (
        _quote_remote_part("KOA_PROJECT_DIR=~/koa-jobs/demo")
        == "KOA_PROJECT_DIR=~/koa-jobs/demo"
    )


def test_quote_remote_part_quotes_regular_strings() -> None:
    assert _quote_remote_part("plain-value") == "plain-value"


def test_run_ssh_prefixes_remote_commands_with_term(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _FakeConfig:
        login = "demo@koa"
        identity_file = None
        proxy_command = None

    def fake_run(command, **kwargs):
        seen["command"] = command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("koa_cli.ssh.subprocess.run", fake_run)

    run_ssh(_FakeConfig(), ["hostname"], capture_output=True)

    ssh_command = seen["command"]
    assert ssh_command[-1].startswith("env TERM=")
