from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

from .config import Config
from .console import print_cmd, print_verbose


class SSHError(RuntimeError):
    """Raised when an SSH/SCP/rsync command returns a non-zero exit status."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _base_args(config: Config, force_tty: bool = False) -> list[str]:
    term_value = os.environ.get("TERM") or "xterm-256color"
    args = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        f"SetEnv=TERM={term_value}",
    ]
    if force_tty:
        args.append("-tt")
    args.extend(["-o", "LogLevel=ERROR"])
    if config.identity_file:
        args.extend(["-i", str(config.identity_file)])
    if config.proxy_command:
        args.extend(["-o", f"ProxyCommand={config.proxy_command}"])
    return args


def _scp_base_args(config: Config) -> list[str]:
    args = [
        "scp",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "LogLevel=ERROR",
    ]
    if config.identity_file:
        args.extend(["-i", str(config.identity_file)])
    if config.proxy_command:
        args.extend(["-o", f"ProxyCommand={config.proxy_command}"])
    return args


def _rsync_ssh_command(config: Config) -> str:
    term_value = os.environ.get("TERM") or "xterm-256color"
    ssh_parts = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        f"SetEnv=TERM={term_value}",
        "-o",
        "LogLevel=ERROR",
    ]
    if config.identity_file:
        ssh_parts.extend(["-i", str(config.identity_file)])
    if config.proxy_command:
        ssh_parts.extend(["-o", f"ProxyCommand={config.proxy_command}"])
    return " ".join(shlex.quote(part) for part in ssh_parts)


def _quote_remote_part(part: str) -> str:
    if "=" in part:
        name, value = part.split("=", 1)
        if value == "~":
            return f"{name}=~"
        if value.startswith("~/"):
            return f"{name}=~/{shlex.quote(value[2:])}"
    if part == "~":
        return "~"
    if part.startswith("~/"):
        return "~/" + shlex.quote(part[2:])
    return shlex.quote(part)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ssh(
    config: Config,
    remote_command: Iterable[str] | str,
    *,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
    force_tty: bool = False,
) -> subprocess.CompletedProcess:
    term_value = os.environ.get("TERM") or "xterm-256color"
    if isinstance(remote_command, str):
        command_str = f"env TERM={shlex.quote(term_value)} {remote_command}"
    else:
        remote_parts = ["env", f"TERM={term_value}", *remote_command]
        command_str = " ".join(_quote_remote_part(part) for part in remote_parts)

    ssh_command = [*_base_args(config, force_tty=force_tty), config.login, command_str]
    print_cmd(ssh_command)

    result = subprocess.run(
        ssh_command,
        check=False,
        capture_output=capture_output,
        text=text,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise SSHError(
            f"SSH command failed (exit {result.returncode}):\n"
            f"  cmd : {' '.join(ssh_command)}\n"
            + (f"  err : {stderr}" if stderr else "")
        )
    return result


def run_remote_shell(
    config: Config,
    script: str,
    *,
    check: bool = True,
    capture_output: bool = False,
    force_tty: bool = False,
) -> subprocess.CompletedProcess:
    return run_ssh(
        config,
        ["bash", "-lc", script],
        check=check,
        capture_output=capture_output,
        force_tty=force_tty,
    )


def copy_to_remote(
    config: Config,
    local_path: Path,
    remote_path: Path,
    *,
    recursive: bool = False,
) -> None:
    args = _scp_base_args(config)
    if recursive:
        args.append("-r")
    scp_command = [
        *args,
        str(local_path),
        f"{config.login}:{remote_path}",
    ]
    print_cmd(scp_command)
    result = subprocess.run(scp_command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise SSHError(
            f"SCP upload failed (exit {result.returncode}):\n"
            f"  cmd : {' '.join(scp_command)}\n"
            + (f"  err : {stderr}" if stderr else "")
        )


def copy_from_remote(
    config: Config,
    remote_path: Path,
    local_path: Path,
    *,
    recursive: bool = False,
) -> None:
    args = _scp_base_args(config)
    if recursive:
        args.append("-r")
    scp_command = [
        *args,
        f"{config.login}:{remote_path}",
        str(local_path),
    ]
    print_cmd(scp_command)
    result = subprocess.run(scp_command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise SSHError(
            f"SCP download failed (exit {result.returncode}):\n"
            f"  cmd : {' '.join(scp_command)}\n"
            + (f"  err : {stderr}" if stderr else "")
        )


def sync_directory_to_remote(
    config: Config,
    local_dir: Path,
    remote_dir: Path,
    *,
    excludes: Sequence[str] | None = None,
    dry_run: bool = False,
) -> str:
    """Sync *local_dir* to *remote_dir* via rsync.

    Parameters
    ----------
    dry_run:
        When *True*, passes ``--dry-run`` to rsync so nothing is actually
        transferred.  The rsync stdout (file list) is returned either way.
    """
    local_dir = local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir}")

    excludes = list(excludes or [])

    run_ssh(config, ["mkdir", "-p", str(remote_dir)])
    ssh_command = _rsync_ssh_command(config)

    rsync_command: list[str] = [
        "rsync",
        "--archive",        # preserve permissions, timestamps, symlinks, etc.
        "--verbose",
        "--human-readable",
        "--delete",         # remove remote files absent locally
    ]
    # Intentionally do not pass --delete-excluded. Excluded remote paths such
    # as result symlinks should survive syncs instead of being removed.
    if dry_run:
        rsync_command.append("--dry-run")

    for pattern in excludes:
        rsync_command.extend(["--exclude", pattern])

    rsync_command.extend(
        [
            "-e", ssh_command,
            f"{local_dir}/",
            f"{config.login}:{remote_dir}",
        ]
    )

    print_cmd(rsync_command)
    print_verbose(f"rsync source : {local_dir}/")
    print_verbose(f"rsync dest   : {config.login}:{remote_dir}")

    result = subprocess.run(
        rsync_command,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise SSHError(
            f"rsync failed (exit {result.returncode}):\n"
            f"  cmd : {' '.join(rsync_command)}\n"
            + (f"  err : {stderr}" if stderr else "")
        )
    return result.stdout
