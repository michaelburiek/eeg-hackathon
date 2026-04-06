"""koa — command-line interface for the KOA HPC cluster."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from time import sleep

from . import __version__
from .config import Config, load_config, validate_slurm_time
from .console import (
    console,
    print_error,
    print_info,
    print_rule,
    print_success,
    print_warning,
    render_state_badge,
    set_verbose,
    spinner,
    state_style,
    supports_live_rendering,
)
from .history import get_latest_job_id, list_history, record_job
from .slurm import (
    JobSnapshot,
    cancel_job,
    ensure_project_directories,
    get_job_output_path,
    get_job_snapshot,
    get_latest_result_dir,
    job_efficiency,
    list_jobs,
    list_result_dirs,
    queue_status,
    run_health_checks,
    stream_job_logs,
    submit_job,
    validate_job_id,
)
from .ssh import (
    SSHError,
    copy_from_remote,
    copy_to_remote,
    run_remote_shell,
    run_ssh,
    sync_directory_to_remote,
)
from .wizard import run_setup

# ---------------------------------------------------------------------------
# rsync excludes that are always applied during koa sync
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDES = [
    ".git/",
    ".gitignore",
    ".venv/",
    ".venv-vllm/",
    "__pycache__/",
    "*.pyc",
    "*.log",
    ".DS_Store",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    "node_modules/",
    "dist/",
    "build/",
    "*.egg-info/",
    "train/results/",
    "eval/results/",
    "artifacts/",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_name(path: Path) -> str:
    return path.expanduser().resolve().name


def _load(args: argparse.Namespace) -> Config:
    return load_config(args.config)


def _resolve_project_name(args: argparse.Namespace, local_path: Path | None = None) -> str:
    if getattr(args, "project", None):
        return args.project
    if local_path is not None:
        return _project_name(local_path)
    return _project_name(Path.cwd())


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        metavar="FILE",
        help="Path to a koa-cli config file (default: ~/.config/koa-cli/config.yaml).",
    )


def _add_project_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project",
        default=None,
        metavar="NAME",
        help="Override the project name (default: current directory name).",
    )


def _add_json_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a Rich table/report.",
    )


def _add_submit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job_script", type=Path, help="Local path to the .slurm script.")
    parser.add_argument(
        "--remote-name",
        default=None,
        metavar="NAME",
        help="Remote file name inside the project directory (default: same as local name).",
    )
    parser.add_argument("--partition", default=None, help="Slurm partition to use.")
    parser.add_argument(
        "--time",
        default=None,
        metavar="HH:MM:SS",
        help="Wall-clock time limit (e.g. 2:00:00 or 1-12:00:00).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        metavar="N",
        help="Number of GPUs to request (sets --gres=gpu:<type>:N, conflicts with --gres).",
    )
    parser.add_argument(
        "--gres",
        default=None,
        metavar="SPEC",
        help="Raw --gres spec (e.g. gpu:a100:2).  Conflicts with --gpus.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        metavar="N",
        help="Number of CPUs per task.",
    )
    parser.add_argument("--memory", default=None, metavar="SIZE", help="Memory (e.g. 64G).")
    parser.add_argument("--account", default=None, help="Slurm account/allocation.")
    parser.add_argument("--qos", default=None, help="Slurm QOS.")
    parser.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Raw sbatch argument (repeatable, e.g. --sbatch-arg='--nodes=2').",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        metavar="N",
        help="Number of nodes to request (sets --nodes=N in sbatch).",
    )
    parser.add_argument(
        "--no-auto-gpu",
        action="store_true",
        help="Disable automatic GPU-type selection.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Enter a live status view after submission.",
    )
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Refresh interval for live status updates (default: 5).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable the built-in Weights & Biases job integration for this submit.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        metavar="NAME",
        help="Set WANDB_PROJECT for the remote job (default: project name).",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        metavar="NAME",
        help="Set WANDB_ENTITY for the remote job.",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        metavar="NAME",
        help="Set WANDB_NAME for the remote job.",
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        metavar="NAME",
        help="Set WANDB_RUN_GROUP for the remote job.",
    )
    parser.add_argument(
        "--wandb-tags",
        default=None,
        metavar="CSV",
        help="Comma-separated WANDB_TAGS value (e.g. koa,slurm,smoke).",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
        help="Set WANDB_MODE for the remote job.",
    )


def _print_json(data: object) -> None:
    sys.stdout.write(json.dumps(data, indent=2, sort_keys=True))
    sys.stdout.write("\n")


def _parse_pipe_rows(raw_output: str) -> list[dict[str, str]]:
    lines = [line for line in raw_output.splitlines() if line.strip()]
    if not lines:
        return []

    header = lines[0].split("|")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        cells = line.split("|")
        if len(cells) < len(header):
            cells.extend([""] * (len(header) - len(cells)))
        rows.append(dict(zip(header, cells[: len(header)], strict=False)))
    return rows


def _normalize_json_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _parse_seff_output(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {"raw": output}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[_normalize_json_key(key)] = value.strip()
    return parsed


def _print_rsync_output(output: str) -> None:
    for line in output.splitlines():
        if line.strip() and not line.startswith("sending") and not line.startswith("sent"):
            console.print(f"  [dim]{line}[/dim]")


def _env_file_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _resolve_env_file(args: argparse.Namespace) -> Path:
    return args.env_file.expanduser().resolve() if args.env_file else Path.cwd() / ".env"


def _remote_env_path(config: Config, project_name: str) -> Path:
    return config.project_workdir(project_name) / ".env"


def _remote_file_exists(config: Config, path: Path) -> bool:
    return run_ssh(
        config,
        ["test", "-f", str(path)],
        capture_output=True,
        check=False,
    ).returncode == 0


def _remote_env_has_key(config: Config, remote_env: Path, key: str) -> bool:
    script = (
        f"test -f {str(remote_env)} && "
        f"grep -q '^[[:space:]]*{re.escape(key)}=' {str(remote_env)}"
    )
    result = run_ssh(
        config,
        ["bash", "-lc", script],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _sync_env_file(config: Config, project_name: str, env_file: Path) -> None:
    remote_env = _remote_env_path(config, project_name)
    run_ssh(config, ["mkdir", "-p", str(remote_env.parent)])
    copy_to_remote(config, env_file, remote_env)
    run_ssh(config, ["chmod", "600", str(remote_env)])


def _wandb_requested(args: argparse.Namespace) -> bool:
    return any(
        [
            getattr(args, "wandb", False),
            getattr(args, "wandb_project", None),
            getattr(args, "wandb_entity", None),
            getattr(args, "wandb_name", None),
            getattr(args, "wandb_group", None),
            getattr(args, "wandb_tags", None),
            getattr(args, "wandb_mode", None),
        ]
    )


def _build_wandb_env(args: argparse.Namespace, config: Config, project_name: str) -> dict[str, str]:
    if not _wandb_requested(args):
        return {}

    data_dir = config.project_data_dir(project_name)
    project_dir = config.project_workdir(project_name)
    partition = getattr(args, "partition", None) or config.default_partition

    env: dict[str, str] = {
        "WANDB_PROJECT": args.wandb_project or project_name,
        "WANDB_DIR": str(data_dir / "wandb" / "runs"),
        "WANDB_CACHE_DIR": str(data_dir / "wandb" / ".cache"),
        "WANDB_CONFIG_DIR": str(project_dir / ".wandb"),
        "WANDB_JOB_TYPE": Path(args.job_script).stem,
    }
    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_name:
        env["WANDB_NAME"] = args.wandb_name
    if args.wandb_group:
        env["WANDB_RUN_GROUP"] = args.wandb_group
    tags = [tag.strip() for tag in (args.wandb_tags or "").split(",") if tag.strip()]
    for default_tag in ("koa", "slurm", partition):
        if default_tag not in tags:
            tags.append(default_tag)
    env["WANDB_TAGS"] = ",".join(tags)
    if args.wandb_mode:
        env["WANDB_MODE"] = args.wandb_mode
    elif getattr(args, "wandb", False):
        env["WANDB_MODE"] = "online"
    return env


def _require_remote_wandb_api_key(
    config: Config,
    project_name: str,
    wandb_env: dict[str, str],
) -> None:
    mode = wandb_env.get("WANDB_MODE", "online").lower()
    if mode in {"disabled", "offline"}:
        return
    remote_env = _remote_env_path(config, project_name)
    if _remote_env_has_key(config, remote_env, "WANDB_API_KEY"):
        return
    raise FileNotFoundError(
        "Weights & Biases was requested, but the remote project .env does not contain "
        "WANDB_API_KEY.\n"
        "Run `koa wandb sync` or upload a .env with `koa auth sync` first."
    )


def _build_sbatch_args(args: argparse.Namespace) -> list[str]:
    if args.gpus is not None and args.gres:
        raise ValueError("--gpus and --gres are mutually exclusive — use one or the other.")

    if args.time:
        validate_slurm_time(args.time)

    sbatch_args: list[str] = []
    if args.partition:
        sbatch_args.extend(["--partition", args.partition])
    if args.time:
        sbatch_args.extend(["--time", args.time])
    if args.gpus is not None:
        sbatch_args.append(f"--gres=gpu:{args.gpus}")
    if args.gres:
        sbatch_args.append(f"--gres={args.gres}")
    if args.cpus:
        sbatch_args.extend(["--cpus-per-task", str(args.cpus)])
    if getattr(args, "nodes", None) is not None:
        sbatch_args.extend(["--nodes", str(args.nodes)])
    if args.memory:
        sbatch_args.extend(["--mem", args.memory])
    if args.account:
        sbatch_args.extend(["--account", args.account])
    if args.qos:
        sbatch_args.extend(["--qos", args.qos])
    sbatch_args.extend(args.sbatch_arg)
    return sbatch_args


def _print_submit_summary(job_id: str, project_name: str, partition_used: str) -> None:
    from rich.panel import Panel
    from rich.text import Text

    body = Text.assemble(
        (" ✓ Submitted", "bold green"),
        f"  job {job_id}  ·  project '{project_name}'\n\n",
        ("   Watch  ", "dim"),
        (f"koa watch {job_id}\n", "bold"),
        ("   Track  ", "dim"),
        ("koa jobs\n", "bold"),
        ("    Logs  ", "dim"),
        (f"koa logs {job_id}\n", "bold"),
        ("  Cancel  ", "dim"),
        (f"koa cancel {job_id}\n", "bold"),
        ("     Eff  ", "dim"),
        (f"koa efficiency {job_id}", "bold"),
        ("  [dim](after job completes)[/dim]", ""),
    )
    console.print(Panel(body, border_style="green", padding=(0, 1)))

    if "gpu" in partition_used.lower() and "sandbox" not in partition_used.lower():
        print_warning(
            "KOA gpu partition: idle GPU monitoring is active.\n"
            "  Jobs that leave GPUs idle for 3+ consecutive hours are auto-cancelled.\n"
            "  Ensure your script starts GPU work promptly after launch."
        )


def _watch_event_label(snapshot: JobSnapshot | None) -> str:
    if snapshot is None:
        return "Waiting for scheduler metadata"
    if snapshot.reason and snapshot.reason.lower() != "none":
        return f"{snapshot.state} · {snapshot.reason}"
    has_location = snapshot.location and snapshot.location not in {"(None)", "(none)"}
    if has_location and not snapshot.location.startswith("("):
        return f"{snapshot.state} · {snapshot.location}"
    return snapshot.state


def _render_watch_view(
    job_id: str,
    project_name: str,
    snapshot: JobSnapshot | None,
    output_path: str,
    events: list[tuple[str, str]],
    frame: str,
) -> object:
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    details = Table.grid(expand=True)
    details.add_column(style="dim", width=12)
    details.add_column()

    if snapshot is None:
        details.add_row("State", Text("DISCOVERING", style="bold cyan"))
        details.add_row("Project", project_name)
        details.add_row("Output", output_path)
        details.add_row("Next", "Waiting for Slurm to report the job status.")
        border_style = "cyan"
    else:
        where_value = snapshot.location
        if where_value in {"", "(None)", "(none)"}:
            where_value = "(waiting)"
        details.add_row("State", render_state_badge(snapshot.state))
        details.add_row("Job", f"{snapshot.job_id}  ·  {snapshot.name}")
        details.add_row("Project", project_name)
        details.add_row("Partition", snapshot.partition or "unknown")
        details.add_row("Elapsed", f"{snapshot.elapsed} / {snapshot.time_limit}")
        details.add_row("Nodes", snapshot.nodes)
        details.add_row("Where", where_value or "(unknown)")
        details.add_row("Output", output_path)
        if snapshot.state.upper().startswith("PENDING"):
            details.add_row(
                "Next",
                "Waiting for scheduler resources. Logs will appear after start.",
            )
        elif snapshot.state.upper() == "RUNNING":
            details.add_row("Next", f"Use `koa logs {job_id}` to stream output.")
        elif snapshot.is_terminal:
            details.add_row(
                "Next",
                f"Use `koa results pull --latest` and `koa efficiency {job_id}`.",
            )
        else:
            details.add_row("Next", f"Use `koa logs {job_id}` when you want stdout.")
        border_style = state_style(snapshot.state).split()[-1]

    timeline = Table(title="Recent Updates", header_style="bold cyan", border_style="bright_black")
    timeline.add_column("Time", style="dim", width=8)
    timeline.add_column("Status")
    for stamp, message in events[-5:]:
        timeline.add_row(stamp, message)

    title = Text.assemble(
        (f"{frame} ", border_style),
        ("Watching ", "bold"),
        (f"job {job_id}", "bold"),
    )
    return Group(
        Panel(details, title=title, border_style=border_style, padding=(0, 1)),
        timeline,
    )


def _watch_job(config: Config, project_name: str, job_id: str, *, interval: float) -> int:
    from rich.live import Live

    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    output_path = get_job_output_path(config, project_name, job_id)
    last_label: str | None = None
    events: list[tuple[str, str]] = []
    missing_snapshots = 0
    frame_index = 0

    if not supports_live_rendering():
        print_info(f"Watching job {job_id} (refresh every {max(interval, 1.0):g}s) …")
        console.print(f"[dim]  Output: {output_path}[/dim]")

        while True:
            snapshot = get_job_snapshot(config, job_id)
            if snapshot is None:
                missing_snapshots += 1
            else:
                missing_snapshots = 0
                label = _watch_event_label(snapshot)
                if label != last_label:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    detail = ""
                    if "·" in label:
                        _, detail = label.split("·", 1)
                        detail = detail.strip()
                    line = [f"[dim][{timestamp}][/dim] ", render_state_badge(snapshot.state)]
                    if detail:
                        line.extend(["  ", f"[dim]{detail}[/dim]"])
                    console.print(*line)
                    last_label = label

            if snapshot and snapshot.is_terminal:
                return 0
            if snapshot is None and missing_snapshots >= 3:
                print_warning(
                    f"Job {job_id} is no longer visible in Slurm. "
                    "It may have finished before accounting data became available."
                )
                return 1

            sleep(max(interval, 1.0))

    with Live(console=console, refresh_per_second=8, transient=False) as live:
        while True:
            snapshot = get_job_snapshot(config, job_id)
            if snapshot is None:
                missing_snapshots += 1
            else:
                missing_snapshots = 0
                label = _watch_event_label(snapshot)
                if label != last_label:
                    events.append((datetime.now().strftime("%H:%M:%S"), label))
                    last_label = label

            live.update(
                _render_watch_view(
                    job_id,
                    project_name,
                    snapshot,
                    output_path,
                    events,
                    frames[frame_index % len(frames)],
                )
            )

            if snapshot and snapshot.is_terminal:
                return 0
            if snapshot is None and missing_snapshots >= 3:
                print_warning(
                    f"Job {job_id} is no longer visible in Slurm. "
                    "It may have finished before accounting data became available."
                )
                return 1

            frame_index += 1
            sleep(max(interval, 1.0))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="koa",
        description=(
            "koa-cli — submit and manage model-training jobs on the KOA HPC cluster.\n\n"
            "Quick start:\n"
            "  koa setup                   configure your credentials\n"
            "  koa check                   verify SSH + Slurm connectivity\n"
            "  koa storage setup           create remote project directories\n"
            "  koa sync                    push local code to the cluster\n"
            "  koa submit train.slurm      submit a training job\n"
            "  koa jobs                    watch your running jobs\n"
            "  koa logs <job_id>           stream job output in real-time\n"
            "  koa results pull --latest   download the latest training run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"koa-cli {__version__}",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Print every SSH/rsync/SCP command before running it.",
    )
    # --config is also accepted here at the top level so it doesn't have to
    # come after the subcommand name.
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to a koa-cli config file (default: ~/.config/koa-cli/config.yaml).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # ---- setup ----
    subparsers.add_parser(
        "setup",
        help="Run the interactive config wizard.",
        description="Interactively create or update ~/.config/koa-cli/config.yaml.",
    )

    # ---- check ----
    check_parser = subparsers.add_parser(
        "check",
        help="Run SSH and Slurm health checks.",
        description="Verify that SSH connectivity and Slurm are working.",
    )
    _add_common_arguments(check_parser)

    # ---- jobs ----
    jobs_parser = subparsers.add_parser(
        "jobs",
        help="Show your running/pending jobs.",
        description="List all of your jobs currently visible in the Slurm queue.",
    )
    _add_common_arguments(jobs_parser)
    _add_json_argument(jobs_parser)

    # ---- queue ----
    queue_parser = subparsers.add_parser(
        "queue",
        help="Show the full cluster queue.",
        description="Show all jobs in the queue.  Your jobs are highlighted.",
    )
    _add_common_arguments(queue_parser)
    _add_json_argument(queue_parser)
    queue_parser.add_argument("--partition", default=None, help="Filter by partition name.")

    # ---- cancel ----
    cancel_parser = subparsers.add_parser(
        "cancel",
        help="Cancel a job.",
        description="Cancel a running or pending Slurm job, array element, or job step.",
    )
    _add_common_arguments(cancel_parser)
    cancel_parser.add_argument(
        "job_id",
        help="Slurm job ID, array element, or step ID (e.g. 123456, 123456_7, 123456.0).",
    )

    # ---- logs ----
    logs_parser = subparsers.add_parser(
        "logs",
        help="Stream the output of a submitted job.",
        description=(
            "Tail the Slurm stdout file for a job in real-time.\n"
            "The job output path is discovered automatically via scontrol.\n"
            "Use --no-follow to just print the last 200 lines and exit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_arguments(logs_parser)
    _add_project_argument(logs_parser)
    logs_parser.add_argument(
        "job_id",
        nargs="?",
        default=None,
        help="Job ID to stream.  Omit to use the most recently submitted job.",
    )
    logs_parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Print the last 200 lines and exit instead of following.",
    )

    # ---- watch ----
    watch_parser = subparsers.add_parser(
        "watch",
        help="Live-watch a submitted job change state.",
        description=(
            "Refresh a single job in-place so you can follow it from pending\n"
            "to running to completion without re-running `koa jobs` manually."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_arguments(watch_parser)
    _add_project_argument(watch_parser)
    watch_parser.add_argument(
        "job_id",
        nargs="?",
        default=None,
        help="Job ID to watch. Omit to use the most recently submitted job.",
    )
    watch_parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Refresh interval in seconds (default: 5).",
    )

    # ---- sync ----
    sync_parser = subparsers.add_parser(
        "sync",
        help="Push local project code to the cluster.",
        description=(
            "Rsync the current directory (or --path) to the remote project directory.\n"
            "Files present on the cluster but absent locally are deleted.\n"
            "Use --dry-run to preview what would change without transferring anything."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_arguments(sync_parser)
    _add_project_argument(sync_parser)
    sync_parser.add_argument(
        "--path",
        type=Path,
        default=None,
        metavar="DIR",
        help="Local directory to sync (default: current directory).",
    )
    sync_parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Additional rsync exclude pattern (repeatable).",
    )
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without transferring any files.",
    )

    # ---- submit ----
    submit_parser = subparsers.add_parser(
        "submit",
        help="Upload and submit a Slurm job script.",
        description=(
            "Upload a local .slurm (or shell) script to the remote project directory\n"
            "and submit it via sbatch.  GPU type is auto-selected by default.\n\n"
            "Example:\n"
            "  koa submit train.slurm --gpus 2 --time 4:00:00"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_arguments(submit_parser)
    _add_project_argument(submit_parser)
    _add_submit_arguments(submit_parser)

    # ---- run ----
    run_parser = subparsers.add_parser(
        "run",
        help="Prepare storage, sync code, and submit a job in one command.",
        description=(
            "One-shot happy path for KOA ML jobs:\n"
            "  1. create project directories on the cluster\n"
            "  2. optionally refresh train/eval scratch symlinks\n"
            "  3. rsync the project to KOA\n"
            "  4. submit the Slurm job script\n\n"
            "Example:\n"
            "  koa run scripts/train.slurm --path . --gpus 2 --time 4:00:00"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_arguments(run_parser)
    _add_project_argument(run_parser)
    run_parser.add_argument(
        "--path",
        type=Path,
        default=None,
        metavar="DIR",
        help="Local directory to sync before submission (default: current directory).",
    )
    run_parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Additional rsync exclude pattern (repeatable).",
    )
    run_parser.add_argument(
        "--link-storage",
        action="store_true",
        help="Refresh train/eval → scratch result symlinks before syncing.",
    )
    _add_submit_arguments(run_parser)

    # ---- storage ----
    storage_parser = subparsers.add_parser(
        "storage",
        help="Manage remote project directories.",
        description="Create and inspect the remote code and scratch directories for a project.",
    )
    _add_common_arguments(storage_parser)
    _add_project_argument(storage_parser)
    storage_subparsers = storage_parser.add_subparsers(
        dest="storage_command",
        required=True,
        metavar="<subcommand>",
    )
    storage_setup = storage_subparsers.add_parser(
        "setup",
        help="Create remote code and data directories.",
    )
    storage_setup.add_argument(
        "--link",
        action="store_true",
        help="Also refresh train/eval result symlinks.",
    )
    storage_subparsers.add_parser("show", help="Print remote project paths.")
    storage_subparsers.add_parser(
        "link",
        help="Refresh train/eval → scratch result symlinks.",
    )

    # ---- auth ----
    auth_parser = subparsers.add_parser(
        "auth",
        help="Manage the remote .env secrets file.",
        description=(
            "Upload or verify the .env file that carries API keys and secrets\n"
            "needed by your training scripts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_arguments(auth_parser)
    _add_project_argument(auth_parser)
    auth_subparsers = auth_parser.add_subparsers(
        dest="auth_command",
        required=True,
        metavar="<subcommand>",
    )
    auth_subparsers.add_parser("check", help="Check whether a remote .env file exists.")
    auth_sync = auth_subparsers.add_parser("sync", help="Upload a local .env file.")
    auth_sync.add_argument(
        "--env-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to the local .env file (default: ./.env).",
    )

    # ---- wandb ----
    wandb_parser = subparsers.add_parser(
        "wandb",
        help="Check and sync Weights & Biases credentials for KOA jobs.",
        description=(
            "W&B is the recommended experiment dashboard layer for koa-cli.\n"
            "Use these commands to verify that WANDB_API_KEY is present locally\n"
            "and in the remote project .env before submitting jobs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_arguments(wandb_parser)
    _add_project_argument(wandb_parser)
    wandb_subparsers = wandb_parser.add_subparsers(
        dest="wandb_command",
        required=True,
        metavar="<subcommand>",
    )
    wandb_check = wandb_subparsers.add_parser(
        "check",
        help="Check local and remote W&B credential readiness.",
    )
    wandb_check.add_argument(
        "--env-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to the local .env file to inspect (default: ./.env).",
    )
    wandb_sync = wandb_subparsers.add_parser(
        "sync",
        help="Upload a local .env and require WANDB_API_KEY to be present.",
    )
    wandb_sync.add_argument(
        "--env-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to the local .env file (default: ./.env).",
    )

    # ---- results ----
    results_parser = subparsers.add_parser(
        "results",
        help="Browse and download training artifacts.",
        description=(
            "List or download result directories from the remote scratch filesystem.\n\n"
            "Examples:\n"
            "  koa results list              show recent training runs\n"
            "  koa results pull --latest     download the most recent run\n"
            "  koa results pull 123456       download results for job 123456"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_arguments(results_parser)
    _add_project_argument(results_parser)
    results_subparsers = results_parser.add_subparsers(
        dest="results_command",
        required=True,
        metavar="<subcommand>",
    )
    results_list = results_subparsers.add_parser("list", help="List recent result directories.")
    results_list.add_argument(
        "--kind",
        choices=["train", "eval", "all"],
        default="train",
        help="Which results to list (default: train).",
    )
    results_list.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of entries to show (default: from config, usually 20).",
    )
    results_list.add_argument(
        "--full-path",
        action="store_true",
        help="Print the full remote path instead of just the directory name.",
    )
    results_pull = results_subparsers.add_parser("pull", help="Download a result directory.")
    pull_id_group = results_pull.add_mutually_exclusive_group()
    pull_id_group.add_argument(
        "job_id",
        nargs="?",
        default=None,
        help="Job ID of the result directory to download.",
    )
    pull_id_group.add_argument(
        "--latest",
        action="store_true",
        help="Download the most recently modified result directory.",
    )
    results_pull.add_argument("--kind", choices=["train", "eval"], default="train")
    results_pull.add_argument(
        "--dest",
        type=Path,
        default=None,
        metavar="DIR",
        help="Local destination path (default: artifacts/<project>/<kind>/<job_id>).",
    )

    # ---- efficiency ----
    eff_parser = subparsers.add_parser(
        "efficiency",
        help="Show CPU/memory efficiency report for a completed job.",
        description=(
            "Runs `seff <job_id>` on the cluster to show a quick efficiency summary\n"
            "for a completed job.  This helps you answer:\n"
            "  - Did my job use the CPU cores it requested?\n"
            "  - Did it use the memory it was allocated?\n\n"
            "Tip: aim for CPU efficiency above 90%.  Low efficiency means you\n"
            "requested more resources than needed, which wastes your allocation\n"
            "and increases queue wait time for everyone.\n\n"
            "For a deeper time-series breakdown, SSH in and run: jobstats <job_id>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_arguments(eff_parser)
    _add_project_argument(eff_parser)
    _add_json_argument(eff_parser)
    eff_parser.add_argument(
        "job_id",
        nargs="?",
        default=None,
        help="Job ID to inspect.  Omit to use the most recently submitted job.",
    )

    # ---- history ----
    history_parser = subparsers.add_parser(
        "history",
        help="Show recently submitted jobs.",
        description=(
            "Display the local submission log of jobs you have submitted via `koa submit`.\n"
            "This is a local record only — it does not query the cluster."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_project_argument(history_parser)
    _add_json_argument(history_parser)
    history_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of entries to display (default: 20).",
    )

    return parser


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _storage_link(config: Config, project_name: str) -> None:
    project_dir = config.project_workdir(project_name)
    data_dir = config.project_data_dir(project_name)
    script = "; ".join(
        [
            f"mkdir -p {shlex.quote(str(project_dir / 'train'))}",
            f"mkdir -p {shlex.quote(str(project_dir / 'eval'))}",
            f"rm -rf {shlex.quote(str(project_dir / 'train' / 'results'))}",
            f"rm -rf {shlex.quote(str(project_dir / 'eval' / 'results'))}",
            (
                f"ln -sfn {shlex.quote(str(data_dir / 'train' / 'results'))} "
                f"{shlex.quote(str(project_dir / 'train' / 'results'))}"
            ),
            (
                f"ln -sfn {shlex.quote(str(data_dir / 'eval' / 'results'))} "
                f"{shlex.quote(str(project_dir / 'eval' / 'results'))}"
            ),
        ]
    )
    run_remote_shell(config, script)


def _cmd_check(config: Config) -> int:
    with spinner("Connecting to KOA …"):
        output = run_health_checks(config)
    console.print(output, end="")
    return 0


def _cmd_jobs(config: Config, *, as_json: bool = False) -> int:
    from .console import render_pipe_table

    if as_json:
        raw_output = list_jobs(config)
    else:
        with spinner("Fetching jobs …"):
            raw_output = list_jobs(config)
    if as_json:
        _print_json(_parse_pipe_rows(raw_output))
        return 0

    render_pipe_table("Your Jobs", raw_output)
    return 0


def _cmd_queue(config: Config, partition: str | None, *, as_json: bool = False) -> int:
    from .console import render_pipe_table

    label = f"partition '{partition}'" if partition else "all partitions"
    if as_json:
        raw_output = queue_status(config, partition=partition)
    else:
        with spinner(f"Fetching queue for {label} …"):
            raw_output = queue_status(config, partition=partition)
    if as_json:
        _print_json(_parse_pipe_rows(raw_output))
        return 0

    render_pipe_table("Cluster Queue", raw_output, highlight_user=config.user)
    return 0


def _cmd_cancel(config: Config, job_id: str) -> int:
    validate_job_id(job_id)  # raises ValueError on bad input
    cancel_job(config, job_id)
    print_success(f"Cancelled job {job_id}")
    return 0


def _cmd_logs(args: argparse.Namespace, config: Config) -> int:
    project_name = _resolve_project_name(args)
    job_id = args.job_id

    if not job_id:
        job_id = get_latest_job_id(project=project_name)
        if not job_id:
            print_error(
                "No job ID provided and no recent jobs found in local history.\n"
                "  Usage: koa logs <job_id>"
            )
            return 1
        print_info(f"Using most recent job: {job_id}")

    validate_job_id(job_id)
    snapshot = get_job_snapshot(config, job_id)
    if snapshot and snapshot.state.upper().startswith("PENDING"):
        reason = f" ({snapshot.reason})" if snapshot.reason else ""
        if args.no_follow:
            print_info(f"Job {job_id} is still pending{reason}. No log output yet.")
            return 0
        print_info(f"Job {job_id} is still pending{reason}. Waiting for logs to appear …")
    print_info(f"Streaming logs for job {job_id} (Ctrl-C to stop) …")
    stream_job_logs(config, project_name, job_id, follow=not args.no_follow)
    return 0


def _cmd_watch(args: argparse.Namespace, config: Config) -> int:
    project_name = _resolve_project_name(args)
    job_id = args.job_id
    if not job_id:
        job_id = get_latest_job_id(project=project_name)
        if not job_id:
            print_error(
                "No job ID provided and no recent jobs found in local history.\n"
                "  Usage: koa watch <job_id>"
            )
            return 1
        print_info(f"Using most recent job: {job_id}")

    validate_job_id(job_id)
    return _watch_job(config, project_name, job_id, interval=args.interval)


def _cmd_sync(args: argparse.Namespace, config: Config) -> int:
    local_path = args.path.expanduser().resolve() if args.path else Path.cwd()
    project_name = _resolve_project_name(args, local_path)
    remote_dir = config.project_workdir(project_name)
    excludes = [*DEFAULT_EXCLUDES, *(args.exclude or [])]

    label = "[DRY RUN] " if args.dry_run else ""
    print_info(f"{label}Syncing {local_path} → {config.login}:{remote_dir}")

    with spinner("Syncing files …"):
        output = sync_directory_to_remote(
            config, local_path, remote_dir, excludes=excludes, dry_run=args.dry_run
        )

    _print_rsync_output(output)

    if args.dry_run:
        print_info("Dry run complete — no files were transferred.")
    else:
        print_success(f"Sync complete for project '{project_name}'")
    return 0


def _cmd_submit(args: argparse.Namespace, config: Config) -> int:
    project_name = _resolve_project_name(args)
    try:
        sbatch_args = _build_sbatch_args(args)
    except ValueError as exc:
        print_error(str(exc))
        return 1
    wandb_env = _build_wandb_env(args, config, project_name)
    if wandb_env:
        _require_remote_wandb_api_key(config, project_name, wandb_env)

    with spinner(f"Submitting {args.job_script.name} …"):
        job_id = submit_job(
            config,
            project_name,
            args.job_script,
            sbatch_args=sbatch_args,
            remote_name=args.remote_name,
            auto_gpu=not (args.no_auto_gpu or args.gpus is not None or args.gres),
            job_env=wandb_env,
        )

    # Save to local history.
    record_job(job_id, project_name, str(args.job_script))

    partition_used = args.partition or config.default_partition
    _print_submit_summary(job_id, project_name, partition_used)
    if args.watch:
        return _watch_job(config, project_name, job_id, interval=args.watch_interval)
    return 0


def _cmd_run(args: argparse.Namespace, config: Config) -> int:
    local_path = args.path.expanduser().resolve() if args.path else Path.cwd()
    project_name = _resolve_project_name(args, local_path)
    remote_dir = config.project_workdir(project_name)
    excludes = [*DEFAULT_EXCLUDES, *(args.exclude or [])]

    try:
        sbatch_args = _build_sbatch_args(args)
    except ValueError as exc:
        print_error(str(exc))
        return 1
    wandb_env = _build_wandb_env(args, config, project_name)
    if wandb_env:
        _require_remote_wandb_api_key(config, project_name, wandb_env)

    # ── 1. Storage ───────────────────────────────────────────────────────────
    print_rule("storage", style="dim cyan")
    with spinner(f"Setting up remote storage for '{project_name}' …"):
        created = ensure_project_directories(config, project_name)
    for path in created:
        print_success(f"Created {path}")

    if args.link_storage:
        with spinner("Refreshing train/eval symlinks …"):
            _storage_link(config, project_name)
        print_success("Refreshed train/eval → scratch result symlinks")

    print_warning(
        "KOA scratch storage policy: files in koa_scratch are automatically\n"
        "  deleted after 90 days of inactivity.  Download important results\n"
        "  with `koa results pull` before they expire."
    )

    # ── 2. Sync ──────────────────────────────────────────────────────────────
    print_rule("sync", style="dim cyan")
    print_info(f"Syncing {local_path} → {config.login}:{remote_dir}")
    with spinner("Syncing files …"):
        output = sync_directory_to_remote(config, local_path, remote_dir, excludes=excludes)
    _print_rsync_output(output)
    print_success(f"Sync complete for project '{project_name}'")

    # ── 3. Submit ────────────────────────────────────────────────────────────
    print_rule("submit", style="dim cyan")
    with spinner(f"Submitting {args.job_script.name} …"):
        job_id = submit_job(
            config,
            project_name,
            args.job_script,
            sbatch_args=sbatch_args,
            remote_name=args.remote_name,
            auto_gpu=not (args.no_auto_gpu or args.gpus is not None or args.gres),
            ensure_dirs=False,  # already called ensure_project_directories above
            job_env=wandb_env,
        )
    record_job(job_id, project_name, str(args.job_script))

    partition_used = args.partition or config.default_partition
    _print_submit_summary(job_id, project_name, partition_used)
    if args.watch:
        return _watch_job(config, project_name, job_id, interval=args.watch_interval)
    return 0


def _cmd_storage(args: argparse.Namespace, config: Config) -> int:
    project_name = _resolve_project_name(args)
    project_dir = config.project_workdir(project_name)
    data_dir = config.project_data_dir(project_name)

    if args.storage_command == "show":
        console.print(f"[bold]Project:[/bold] {project_name}")
        console.print(f"  Code (workdir) : {project_dir}")
        console.print(f"  Data (scratch) : {data_dir}")
        console.print(f"  Train results  : {data_dir / 'train' / 'results'}")
        console.print(f"  Eval results   : {data_dir / 'eval' / 'results'}")
        return 0

    created = ensure_project_directories(config, project_name)
    if args.storage_command == "setup":
        for path in created:
            print_success(f"Created {path}")
        if args.link:
            _storage_link(config, project_name)
            print_success("Refreshed train/eval → scratch result symlinks")
        print_warning(
            "KOA scratch storage policy: files in koa_scratch are automatically\n"
            "  deleted after 90 days of inactivity.  Download important results\n"
            "  with `koa results pull` before they expire."
        )
        return 0

    if args.storage_command == "link":
        _storage_link(config, project_name)
        print_success("Refreshed train/eval → scratch result symlinks")
        return 0

    return 1


def _cmd_auth(args: argparse.Namespace, config: Config) -> int:
    project_name = _resolve_project_name(args)
    remote_env = _remote_env_path(config, project_name)

    if args.auth_command == "check":
        if _remote_file_exists(config, remote_env):
            print_success(f"Remote .env exists at {remote_env}")
            return 0
        print_warning(f"No remote .env found at {remote_env}  (run: koa auth sync)")
        return 1

    env_file = _resolve_env_file(args)
    if not env_file.exists():
        raise FileNotFoundError(f"Local .env not found: {env_file}")
    _sync_env_file(config, project_name, env_file)
    print_success(f"Uploaded {env_file} → {remote_env} (mode 600)")
    return 0


def _cmd_wandb(args: argparse.Namespace, config: Config) -> int:
    project_name = _resolve_project_name(args)
    env_file = _resolve_env_file(args)
    remote_env = _remote_env_path(config, project_name)

    if args.wandb_command == "check":
        local_file_exists = env_file.exists()
        local_values = _env_file_values(env_file) if local_file_exists else {}
        shell_has_key = bool(os.getenv("WANDB_API_KEY"))
        local_file_has_key = "WANDB_API_KEY" in local_values

        remote_env_exists = _remote_file_exists(config, remote_env)
        remote_has_key = _remote_env_has_key(config, remote_env, "WANDB_API_KEY")

        if shell_has_key:
            print_success("Local shell has WANDB_API_KEY set.")
        else:
            print_warning("Local shell does not have WANDB_API_KEY set.")

        if local_file_exists and local_file_has_key:
            print_success(f"Local .env contains WANDB_API_KEY: {env_file}")
        elif local_file_exists:
            print_warning(f"Local .env exists but is missing WANDB_API_KEY: {env_file}")
        else:
            print_warning(f"Local .env not found: {env_file}")

        if remote_env_exists and remote_has_key:
            print_success(f"Remote .env is ready for W&B: {remote_env}")
            return 0
        if remote_env_exists:
            print_warning(f"Remote .env exists but is missing WANDB_API_KEY: {remote_env}")
        else:
            print_warning(f"Remote .env not found: {remote_env}")

        print_info("Run `koa wandb sync` after adding WANDB_API_KEY to your local .env.")
        return 1

    if not env_file.exists():
        raise FileNotFoundError(f"Local .env not found: {env_file}")
    local_values = _env_file_values(env_file)
    if "WANDB_API_KEY" not in local_values:
        raise ValueError(
            f"WANDB_API_KEY is missing from {env_file}.\n"
            "Add it locally, then re-run `koa wandb sync`."
        )

    _sync_env_file(config, project_name, env_file)
    print_success(f"Uploaded W&B-ready .env → {remote_env}")
    print_info(f"W&B jobs can now use WANDB_API_KEY from {remote_env}")
    return 0


def _cmd_results(args: argparse.Namespace, config: Config) -> int:
    project_name = _resolve_project_name(args)

    if args.results_command == "list":
        limit = args.limit if args.limit is not None else config.default_results_limit
        kinds = ["train", "eval"] if args.kind == "all" else [args.kind]
        for kind in kinds:
            remote_dir = config.project_results_dir(project_name, kind)
            print_info(f"{kind} results → {remote_dir}")
            rows = list_result_dirs(config, project_name, kind, limit=limit)
            if not rows:
                console.print("  [dim](empty)[/dim]")
                continue
            for row in rows:
                console.print(f"  {row if args.full_path else Path(row).name}")
        return 0

    # ---- pull ----
    if args.latest:
        # Try local history first, then query the remote filesystem.
        remote_path = get_latest_result_dir(config, project_name, args.kind)
        if not remote_path:
            results_dir = config.project_results_dir(project_name, args.kind)
            raise FileNotFoundError(f"No result directories found in {results_dir}")
        job_dir_name = Path(remote_path).name
        print_info(f"Using latest result: {job_dir_name}")
        remote_dir = config.project_results_dir(project_name, args.kind) / job_dir_name
    else:
        if not args.job_id:
            print_error("Provide a job ID or pass --latest.\n  Usage: koa results pull <job_id>")
            return 1
        validate_job_id(args.job_id)
        remote_dir = config.project_results_dir(project_name, args.kind) / args.job_id
        job_dir_name = args.job_id

    exists = run_remote_shell(
        config,
        f"test -d {shlex.quote(str(remote_dir))}",
        capture_output=True,
        check=False,
    )
    if exists.returncode != 0:
        raise FileNotFoundError(f"Remote results directory not found: {remote_dir}")

    destination = (
        args.dest.expanduser()
        if args.dest
        else Path("artifacts") / project_name / args.kind / job_dir_name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    print_info(f"Downloading {remote_dir} → {destination.resolve()} …")
    copy_from_remote(config, remote_dir, destination, recursive=True)
    print_success(f"Downloaded results to {destination.resolve()}")
    return 0


def _cmd_efficiency(args: argparse.Namespace, config: Config) -> int:
    project_name = _resolve_project_name(args)
    job_id = args.job_id
    if not job_id:
        job_id = get_latest_job_id(project=project_name)
        if not job_id:
            print_error(
                "No job ID provided and no recent jobs in local history.\n"
                "  Usage: koa efficiency <job_id>"
            )
            return 1
        print_info(f"Using most recent job: {job_id}")
    validate_job_id(job_id)
    output = job_efficiency(config, job_id)
    if args.json:
        payload = _parse_seff_output(output)
        payload["job_id"] = job_id
        _print_json(payload)
        return 0
    console.print(output)
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    project = getattr(args, "project", None)
    records = list_history(project=project, limit=args.limit)
    if args.json:
        _print_json(records)
        return 0
    if not records:
        label = f"project '{project}'" if project else "any project"
        print_info(f"No job history found for {label}.")
        return 0

    from rich.table import Table

    table = Table(title="Job History (newest first)", header_style="bold cyan", border_style="blue")
    table.add_column("Job ID", style="bold")
    table.add_column("Project")
    table.add_column("Script")
    table.add_column("Submitted (UTC)")
    for record in records:
        table.add_row(
            record["job_id"],
            record["project"],
            record["script"],
            record["submitted_at"],
        )
    console.print(table)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Enable verbose mode globally before any real work.
    if getattr(args, "verbose", False):
        set_verbose(True)

    if args.command == "setup":
        return run_setup()

    if args.command == "history":
        return _cmd_history(args)

    try:
        config = _load(args)

        if args.command == "check":
            return _cmd_check(config)
        if args.command == "jobs":
            return _cmd_jobs(config, as_json=args.json)
        if args.command == "queue":
            return _cmd_queue(config, args.partition, as_json=args.json)
        if args.command == "cancel":
            return _cmd_cancel(config, args.job_id)
        if args.command == "logs":
            return _cmd_logs(args, config)
        if args.command == "watch":
            return _cmd_watch(args, config)
        if args.command == "sync":
            return _cmd_sync(args, config)
        if args.command == "submit":
            return _cmd_submit(args, config)
        if args.command == "run":
            return _cmd_run(args, config)
        if args.command == "storage":
            return _cmd_storage(args, config)
        if args.command == "auth":
            return _cmd_auth(args, config)
        if args.command == "wandb":
            return _cmd_wandb(args, config)
        if args.command == "results":
            return _cmd_results(args, config)
        if args.command == "efficiency":
            return _cmd_efficiency(args, config)

    except (FileNotFoundError, ValueError) as exc:
        print_error(str(exc))
        return 1
    except SSHError as exc:
        print_error(str(exc))
        if not getattr(args, "verbose", False):
            print_info("Tip: re-run with --verbose to see the full SSH command.")
        return 1
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return 130

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
