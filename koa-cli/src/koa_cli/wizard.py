from __future__ import annotations

import subprocess
from pathlib import Path

import yaml
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax

from .console import console, print_error, print_info, print_success, print_warning


def _test_ssh_connection(user: str, host: str, identity_file: str | None = None) -> bool:
    cmd = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "LogLevel=ERROR",  # suppress banners and warnings during the setup wizard
    ]
    if identity_file:
        cmd.extend(["-i", identity_file])
    cmd.extend([f"{user}@{host}", "exit"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _show_diff(existing: dict, proposed: dict) -> None:
    """Print a human-readable summary of what will change."""
    console.print("\n[bold]Config changes:[/bold]")
    all_keys = sorted(set(existing) | set(proposed))
    changed = False
    for key in all_keys:
        old = existing.get(key)
        new = proposed.get(key)
        if old == new:
            continue
        changed = True
        if old is None:
            console.print(f"  [green]+ {key}: {new}[/green]")
        elif new is None:
            console.print(f"  [red]- {key}: {old}[/red]")
        else:
            console.print(f"  [yellow]~ {key}: {old!r}  →  {new!r}[/yellow]")
    if not changed:
        console.print("  [dim](no changes)[/dim]")


def _print_ssh_setup_guidance(user: str, host: str, identity_file: str | None = None) -> None:
    suggested_key = identity_file or "~/.ssh/id_ed25519_koa"
    console.print("\n[bold]Recommended SSH setup for KOA[/bold]")
    console.print(
        "koa-cli runs SSH/SCP/rsync in non-interactive batch mode, so the clean path is:\n"
        "1. use a dedicated KOA SSH key\n"
        "   [dim](recommended; do not delete existing keys used elsewhere)[/dim]\n"
        "2. add an SSH config alias with connection sharing\n"
        "3. open one manual `ssh koa` session to satisfy Duo\n"
        "4. reuse that authenticated connection for `koa check`, `koa sync`, and `koa submit`"
    )
    console.print("\n[bold]Suggested commands[/bold]")
    console.print(
        Syntax(
            (
                f"ssh-keygen -t ed25519 -C \"{user}@hawaii.edu\" -f {suggested_key}\n"
                "mkdir -p ~/.ssh/agent\n"
                "chmod 700 ~/.ssh ~/.ssh/agent"
            ),
            "bash",
            theme="monokai",
        )
    )
    console.print("[bold]Suggested ~/.ssh/config entry[/bold]")
    console.print(
        Syntax(
            (
                "Host koa\n"
                f"  HostName {host}\n"
                f"  User {user}\n"
                f"  IdentityFile {suggested_key}\n"
                "  IdentitiesOnly yes\n"
                "  AddKeysToAgent yes\n"
                "  UseKeychain yes\n"
                "  ServerAliveInterval 60\n"
                "  ServerAliveCountMax 3\n"
                "  ControlMaster auto\n"
                "  ControlPersist 8h\n"
                "  ControlPath ~/.ssh/agent/%r@%h:%p"
            ),
            "sshconfig",
            theme="monokai",
        )
    )
    console.print(
        "After adding that config, run [bold]ssh koa[/bold] once and complete Duo.\n"
        "For the cleanest koa-cli config, set [bold]host: koa[/bold] in "
        "[bold]~/.config/koa-cli/config.yaml[/bold]."
    )


def run_setup() -> int:
    console.print("[bold cyan]koa-cli setup wizard[/bold cyan]")
    console.print("[dim]Press Enter to keep the default value shown in brackets.[/dim]\n")

    user = Prompt.ask("KOA username").strip()
    if not user:
        print_error("Username is required.")
        return 1

    host = Prompt.ask("KOA host", default="koa.its.hawaii.edu")
    remote_workdir = Prompt.ask("Remote code root", default="~/koa-jobs")
    remote_data_dir = Prompt.ask(
        "Remote scratch root",
        default=f"/mnt/lustre/koa/scratch/{user}/koa-jobs",
    )
    identity_file = Prompt.ask(
        "SSH identity file [leave blank to use default]",
        default="",
        show_default=False,
    ).strip() or None
    proxy_command = Prompt.ask(
        "SSH proxy command [leave blank if none]",
        default="",
        show_default=False,
    ).strip() or None

    console.print("\n[bold]Cluster defaults[/bold] [dim](can be overridden per-command)[/dim]")
    default_partition = Prompt.ask(
        "Default Slurm partition",
        default="gpu",
    )
    default_gpu = Prompt.ask("Fallback GPU type", default="rtx2080ti")
    default_time = Prompt.ask(
        "Default time limit [leave blank for none]",
        default="",
        show_default=False,
    ).strip() or None

    # Build proposed config dict.
    proposed: dict = {
        "user": user,
        "host": host,
        "remote_workdir": remote_workdir,
        "remote_data_dir": remote_data_dir,
    }
    if identity_file:
        proposed["identity_file"] = identity_file
    if proxy_command:
        proposed["proxy_command"] = proxy_command
    if default_partition != "gpu":
        proposed["default_partition"] = default_partition
    if default_gpu != "rtx2080ti":
        proposed["default_gpu"] = default_gpu
    if default_time:
        proposed["default_time"] = default_time

    # Test SSH connectivity.
    print_info(f"Testing SSH connection to {user}@{host} …")
    ssh_verified = _test_ssh_connection(user, host, identity_file)
    if ssh_verified:
        print_success("SSH connection succeeded.")
    else:
        print_warning("SSH connection could not be verified (host may require VPN or a jump host).")
        if not Confirm.ask("Write the config anyway?", default=True):
            return 1

    config_dir = Path.home() / ".config" / "koa-cli"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"

    # Show diff if an existing config is present.
    existing: dict = {}
    if config_path.exists():
        try:
            existing = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            existing = {}
        _show_diff(existing, proposed)
        console.print()
        if not Confirm.ask(f"Overwrite {config_path}?", default=False):
            print_info("Aborted — existing config was not modified.")
            return 1
    else:
        console.print("\n[bold]Proposed config:[/bold]")
        console.print(Syntax(yaml.safe_dump(proposed, sort_keys=False), "yaml", theme="monokai"))

    config_path.write_text(yaml.safe_dump(proposed, sort_keys=False), encoding="utf-8")
    print_success(f"Wrote {config_path}")
    if not ssh_verified:
        _print_ssh_setup_guidance(user, host, identity_file)
    console.print(
        "\n[dim]Tip: run [bold]koa check[/bold] to verify connectivity, "
        "then [bold]koa storage setup[/bold] to create remote directories.[/dim]"
    )
    return 0
