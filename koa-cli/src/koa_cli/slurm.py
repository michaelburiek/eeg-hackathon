from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import sleep

from .config import Config
from .ssh import SSHError, copy_to_remote, run_remote_shell, run_ssh

SBATCH_JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")
GRES_PATTERN = re.compile(r"--gres[=\s]+gpu:(?:[^:]+:)?(\d+)")
JOB_ID_PATTERN = re.compile(r"^\d+(?:_\d+)?(?:\.\d+)?$")
SBATCH_GPU_REQUEST_PATTERN = re.compile(r"--gpus(?:-per-(?:node|task))?(?:[=\s]|$)")
GPU_TOKEN_PATTERN = re.compile(r"gpu:([A-Za-z0-9_.-]+)")
SQUEUE_JOB_FORMAT = "%i|%j|%T|%M|%l|%D|%R|%P"
SACCT_JOB_FORMAT = "JobIDRaw,JobName,State,Elapsed,Timelimit,NNodes,NodeList,Partition"
TERMINAL_JOB_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}

# GPU priority scores based on actual KOA hardware (as of 2024).
# Higher score = preferred by auto-GPU selection.
# Sources: KOA HPC-101 GPU node table and sinfo GRES output.
GPU_PRIORITY = {
    # --- H200 family (Sierra Forest / Granite Rapids nodes) ---
    "h200": 110,
    "nvidiah200nvl": 110,
    "nv-h200": 110,
    # --- H100 family (Emerald Rapids / Genoa AMD nodes) ---
    "h100": 100,
    "nvidiah100": 100,
    "nv-h100": 100,
    "nvidiah100pcie": 100,
    "nvidiah100nvl": 100,
    # --- A100 family ---
    "a100": 90,
    "nvidiaa100": 90,
    "nv-a100": 90,
    # --- L40 (Sapphire Rapids nodes, 48 GB VRAM) ---
    "l40": 88,
    "nv-l40": 88,
    "nvl40": 88,
    # --- A30 MIG / full (Ice Lake and Sapphire Rapids nodes) ---
    "a3012g": 83,   # A30 MIG 12g slice
    "nv-a3012g": 83,
    "a306g": 78,    # A30 MIG 6g slice
    "nv-a306g": 78,
    "nvidiaa302g12gb": 79,
    "nvidiaa301g6gb": 77,
    "a30": 80,
    "nvidiaa30": 80,
    "nv-a30": 80,
    # --- V100 family (Cascade Lake nodes) ---
    "v100smx": 72,
    "v100-smx": 72,
    "nv-v100smx": 72,
    "nvv100sxm2": 72,
    "v100": 70,
    "nvidiav100": 70,
    "nv-v100": 70,
    # --- Professional / workstation GPUs (Ice Lake / Cascade Lake nodes) ---
    "rtxa4000": 65,   # RTX A4000 — 16 GB VRAM
    "nv-rtxa4000": 65,
    "nvrtxa4000": 65,
    "rtx5000": 60,    # Quadro RTX 5000 — 16 GB VRAM
    "nv-rtx5000": 60,
    "nvrtx5000": 60,
    # --- RTX 2080Ti (Skylake nodes, 11 GB VRAM) ---
    "rtx2080ti": 50,
    "rtx_2080_ti": 50,
    "geforce_rtx_2080_ti": 50,
    "nvidiageforcertx2080ti": 50,
    "nv-rtx2080ti": 50,
    # --- RTX 2070 (single Skylake node, 8 GB VRAM) ---
    "rtx2070": 45,
    "nv-rtx2070": 45,
}

# Normalised GRES names to pass to sbatch --gres=gpu:<name>:<count>.
# Keys are the stripped/lowercase sinfo GRES token; values are the sbatch form.
GPU_NAME_MAP = {
    "nvidiah200nvl": "nvidia_h200_nvl",
    "nvidiah100": "nvidia_h100",
    "nvidiah100pcie": "nvidia_h100_pcie",
    "nvidiah100nvl": "nvidia_h100_nvl",
    "nvidiaa100": "nvidia_a100",
    "nvidiaa30": "nvidia_a30",
    "nvidiaa302g12gb": "nvidia_a30_2g.12gb",
    "nvidiaa301g6gb": "nvidia_a30_1g.6gb",
    "nvidiav100": "nvidia_v100",
    "nvidiageforcertx2080ti": "geforce_rtx_2080_ti",
    # NV- prefixed forms seen in KOA sinfo output
    "nv-h200": "NV-H200",
    "nv-h100": "NV-H100",
    "nv-l40": "NV-L40",
    "nv-a100": "NV-A100",
    "nv-a30": "NV-A30",
    "nv-a3012g": "NV-A30_12g",
    "nv-a306g": "NV-A30_6g",
    "nv-v100": "NV-V100",
    "nv-v100smx": "NV-V100SMX",
    "nvv100sxm2": "NV-V100-SXM2",
    "nv-rtxa4000": "NV-RTXA4000",
    "nvrtxa4000": "NV-RTX-A4000",
    "nv-rtx5000": "NV-RTX5000",
    "nv-rtx2080ti": "NV-RTX2080Ti",
    "nv-rtx2070": "NV-RTX2070",
}


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    name: str
    state: str
    elapsed: str
    time_limit: str
    nodes: str
    location: str
    partition: str
    reason: str = ""
    source: str = "squeue"

    @property
    def is_terminal(self) -> bool:
        return self.state.upper() in TERMINAL_JOB_STATES


def validate_job_id(job_id: str) -> str:
    """Return *job_id* if it looks like a valid Slurm job identifier.

    Accepts plain job IDs (``12345``), array elements (``12345_7``), and
    job-step identifiers (``12345.0`` or ``12345_7.0``).
    """
    if not JOB_ID_PATTERN.match(job_id.strip()):
        raise ValueError(
            f"Invalid job ID: {job_id!r}. "
            "Expected a Slurm job ID like 123456, 123456_7, or 123456.0."
        )
    return job_id.strip()


def _has_partition_flag(args: Iterable[str]) -> bool:
    return any(
        arg in {"--partition", "-p"}
        or arg.startswith("--partition=")
        or (arg.startswith("-p") and arg != "-p")
        for arg in args
    )


def _has_gres_flag(args: Iterable[str]) -> bool:
    return any(
        arg in {"--gres", "--gpus", "--gpus-per-node", "--gpus-per-task"}
        or arg.startswith("--gres=")
        or arg.startswith("--gpus=")
        or arg.startswith("--gpus-per-node=")
        or arg.startswith("--gpus-per-task=")
        for arg in args
    )


def parse_gpu_count_from_script(script_path: Path) -> int | None:
    try:
        for line in script_path.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#SBATCH") and "--gres" in line:
                match = GRES_PATTERN.search(line)
                if match:
                    return int(match.group(1))
    except OSError:
        return None
    return None


def _normalize_gpu_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


NORMALIZED_GPU_PRIORITY = {
    _normalize_gpu_token(name): score for name, score in GPU_PRIORITY.items()
}

NORMALIZED_GPU_NAME_MAP = {
    _normalize_gpu_token(name): gpu_name for name, gpu_name in GPU_NAME_MAP.items()
}


def script_requests_gpu_resources(script_path: Path) -> bool:
    """Return True if a Slurm script declares GPU resources via #SBATCH flags."""
    try:
        for line in script_path.read_text(encoding="utf-8").splitlines():
            line = line.lstrip()
            if not line.startswith("#SBATCH"):
                continue
            if "--gres" in line and GRES_PATTERN.search(line):
                return True
            if SBATCH_GPU_REQUEST_PATTERN.search(line):
                return True
    except OSError:
        return False
    return False


def get_available_gpus(config: Config, partition: str) -> dict[str, int]:
    try:
        result = run_ssh(
            config,
            ["sinfo", "-h", "-p", partition, "-o", "%G|%t|%D"],
            capture_output=True,
        )
    except SSHError:
        return {}

    counts: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split("|", 2)]
        if len(parts) < 3:
            continue
        gres = parts[0].lower()
        state = parts[1].lower()
        try:
            node_count = int(parts[2])
        except ValueError:
            node_count = 1
        state_is_usable = any(state.startswith(prefix) for prefix in ("idle", "mix", "mixed"))
        if not state_is_usable or "gpu:" not in gres:
            continue
        for gpu_token in GPU_TOKEN_PATTERN.findall(gres):
            gpu_name = _normalize_gpu_token(gpu_token)
            counts[gpu_name] = counts.get(gpu_name, 0) + node_count
    return counts


def select_best_gpu(config: Config, partition: str) -> str:
    """Query partition GPU availability and return the highest-priority GPU type name."""
    available = get_available_gpus(config, partition)
    best_name = None
    best_score = -1
    for gpu_name, count in available.items():
        if count <= 0:
            continue
        score = NORMALIZED_GPU_PRIORITY.get(gpu_name, 0)
        if score > best_score:
            best_name = gpu_name
            best_score = score
    if not best_name:
        return config.default_gpu
    return NORMALIZED_GPU_NAME_MAP.get(best_name, best_name)


def _parse_job_snapshot_row(row: str, *, source: str) -> JobSnapshot | None:
    parts = [part.strip() for part in row.split("|")]
    if source == "squeue":
        if len(parts) < 8 or not parts[0]:
            return None
        location = parts[6]
        reason = ""
        if location.startswith("(") and location.endswith(")"):
            reason = location[1:-1]
        return JobSnapshot(
            job_id=parts[0],
            name=parts[1],
            state=parts[2],
            elapsed=parts[3],
            time_limit=parts[4],
            nodes=parts[5],
            location=location,
            partition=parts[7],
            reason=reason,
            source=source,
        )

    if len(parts) < 8 or not parts[0]:
        return None
    return JobSnapshot(
        job_id=parts[0],
        name=parts[1],
        state=parts[2],
        elapsed=parts[3],
        time_limit=parts[4],
        nodes=parts[5],
        location=parts[6] or "(completed)",
        partition=parts[7],
        reason="",
        source=source,
    )


def get_job_snapshot(config: Config, job_id: str) -> JobSnapshot | None:
    validate_job_id(job_id)

    live_result = run_ssh(
        config,
        ["squeue", "-h", "-j", job_id, "-o", SQUEUE_JOB_FORMAT],
        capture_output=True,
        check=False,
    )
    for row in live_result.stdout.splitlines():
        snapshot = _parse_job_snapshot_row(row, source="squeue")
        if snapshot:
            return snapshot

    history_result = run_ssh(
        config,
        ["sacct", "-n", "-P", "-X", "-j", job_id, f"--format={SACCT_JOB_FORMAT}"],
        capture_output=True,
        check=False,
    )
    for row in history_result.stdout.splitlines():
        snapshot = _parse_job_snapshot_row(row, source="sacct")
        if snapshot and snapshot.job_id == job_id:
            return snapshot

    return None


def ensure_project_directories(config: Config, project_name: str) -> list[str]:
    """Create remote project directories. Returns a list of paths that were created."""
    project_dir = config.project_workdir(project_name)
    data_dir = config.project_data_dir(project_name)
    paths = [
        str(project_dir),
        str(data_dir / "train" / "results"),
        str(data_dir / "eval" / "results"),
    ]
    for p in paths:
        run_ssh(config, ["mkdir", "-p", p])
    return paths


def submit_job(
    config: Config,
    project_name: str,
    local_job_script: Path,
    *,
    sbatch_args: Iterable[str] | None = None,
    remote_name: str | None = None,
    auto_gpu: bool = True,
    ensure_dirs: bool = True,
    job_env: Mapping[str, str] | None = None,
) -> str:
    """Upload *local_job_script* and submit it via sbatch. Returns the job ID string.

    Parameters
    ----------
    ensure_dirs:
        When *True* (the default), ``ensure_project_directories`` is called
        before submission.  Pass *False* if the caller has already created the
        directories (e.g. ``koa run`` does this explicitly to avoid a second
        round-trip).
    """
    if not local_job_script.exists():
        raise FileNotFoundError(f"Job script not found: {local_job_script}")

    if ensure_dirs:
        ensure_project_directories(config, project_name)
    project_dir = config.project_workdir(project_name)
    data_dir = config.project_data_dir(project_name)

    remote_script = project_dir / (remote_name or local_job_script.name)
    run_ssh(config, ["mkdir", "-p", str(remote_script.parent)])
    copy_to_remote(config, local_job_script, remote_script)

    for env_key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"):
        env_value = (job_env or {}).get(env_key)
        if env_value:
            run_ssh(config, ["mkdir", "-p", env_value])

    args = [
        "env",
        f"KOA_PROJECT_DIR={project_dir}",
        f"KOA_REMOTE_WORKDIR={config.remote_workdir}",
        f"KOA_REMOTE_DATA_DIR={data_dir}",
        *[f"{key}={value}" for key, value in (job_env or {}).items()],
        "sbatch",
    ]
    sbatch_args_list = list(sbatch_args or [])

    if not _has_partition_flag(sbatch_args_list):
        args.extend(["--partition", config.default_partition])

    # Apply default time limit from config if none given in the script or CLI.
    has_time_flag = any(
        a in {"--time", "-t"} or a.startswith("--time=") for a in sbatch_args_list
    )
    if not has_time_flag and config.default_time:
        args.extend(["--time", config.default_time])

    script_requests_gpu = script_requests_gpu_resources(local_job_script)
    if auto_gpu and not _has_gres_flag(sbatch_args_list) and not script_requests_gpu:
        partition = config.default_partition
        for index, arg in enumerate(sbatch_args_list):
            if arg in {"--partition", "-p"} and index + 1 < len(sbatch_args_list):
                partition = sbatch_args_list[index + 1]
            elif arg.startswith("--partition="):
                partition = arg.split("=", 1)[1]
        args.extend(["--gres", f"gpu:{select_best_gpu(config, partition)}:1"])

    args.extend(sbatch_args_list)
    args.append(str(remote_script))

    result = run_ssh(config, args, capture_output=True)
    match = SBATCH_JOB_ID_PATTERN.search(result.stdout)
    if not match:
        raise SSHError(f"Unable to parse sbatch output: {result.stdout.strip()!r}")
    return match.group(1)


def cancel_job(config: Config, job_id: str) -> None:
    """Cancel a Slurm job after validating the job ID."""
    validate_job_id(job_id)
    run_ssh(config, ["scancel", job_id])


def stream_job_logs(config: Config, project_name: str, job_id: str, *, follow: bool = True) -> None:
    """Stream the Slurm output file for *job_id* to stdout.

    Uses ``scontrol show job`` to discover the real output path first; falls
    back to ``<project_workdir>/slurm-<job_id>.out``.
    """
    validate_job_id(job_id)

    output_file = get_job_output_path(config, project_name, job_id)
    file_exists = run_remote_shell(
        config,
        f"test -f {shlex.quote(output_file)}",
        capture_output=True,
        check=False,
    )

    if file_exists.returncode != 0:
        snapshot = get_job_snapshot(config, job_id)
        if snapshot and snapshot.state.upper().startswith("PENDING"):
            reason = f" ({snapshot.reason})" if snapshot.reason else ""
            if not follow:
                raise SSHError(
                    f"Job {job_id} is still pending{reason}. "
                    "No Slurm output file exists yet."
                )

            while True:
                snapshot = get_job_snapshot(config, job_id)
                file_exists = run_remote_shell(
                    config,
                    f"test -f {shlex.quote(output_file)}",
                    capture_output=True,
                    check=False,
                )
                if file_exists.returncode == 0:
                    break
                if snapshot and snapshot.is_terminal:
                    raise SSHError(
                        f"Job {job_id} reached state {snapshot.state} before "
                        "a log file appeared."
                    )
                sleep(5)
        elif file_exists.returncode != 0:
            raise SSHError(f"Remote log file not found: {output_file}")

    flag = "-f" if follow else "-n 200"
    run_ssh(
        config,
        f"tail {flag} {shlex.quote(output_file)}",
        force_tty=True,
        capture_output=False,
    )


def get_job_output_path(config: Config, project_name: str, job_id: str) -> str:
    """Return the remote stdout path for *job_id*, falling back to the project directory."""
    output_file: str | None = None
    result = run_ssh(
        config,
        ["scontrol", "show", "job", job_id],
        capture_output=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        for part in line.split():
            if part.startswith("StdOut="):
                candidate = part.split("=", 1)[1].strip()
                if candidate and candidate != "/dev/null":
                    output_file = candidate
                break

    if not output_file:
        project_dir = config.project_workdir(project_name)
        output_file = str(project_dir / f"slurm-{job_id}.out")
    return output_file


def list_result_dirs(
    config: Config,
    project_name: str,
    kind: str,
    *,
    limit: int = 20,
) -> list[str]:
    """Return result sub-directories sorted by descending modification time."""
    remote_dir = config.project_results_dir(project_name, kind)
    quoted = shlex.quote(str(remote_dir))
    script = (
        f"if [ -d {quoted} ]; then "
        f"find {quoted} -mindepth 1 -maxdepth 1 -type d "
        r"-printf '%T@\t%p\n' "
        f"| sort -nr | head -n {max(1, limit)} | cut -f2-; "
        "fi"
    )
    result = run_remote_shell(config, script, capture_output=True, check=False)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def get_latest_result_dir(config: Config, project_name: str, kind: str) -> str | None:
    """Return the path of the most recently modified results sub-directory, or None."""
    rows = list_result_dirs(config, project_name, kind, limit=1)
    return rows[0] if rows else None


def job_efficiency(config: Config, job_id: str) -> str:
    """Run ``seff <job_id>`` and return its output.

    ``seff`` is a Slurm efficiency tool available on KOA that gives a quick
    CPU/memory efficiency report for a completed job.  It helps you answer
    "Did my job make good use of the resources I requested?"
    """
    validate_job_id(job_id)
    result = run_ssh(config, ["seff", job_id], capture_output=True, check=False)
    output = result.stdout.strip()
    if result.returncode != 0 or not output:
        stderr = (result.stderr or "").strip()
        raise SSHError(
            f"seff failed for job {job_id}. "
            "The job may still be running, or seff may not be available.\n"
            + (f"stderr: {stderr}" if stderr else "")
        )
    return output


def list_jobs(config: Config) -> str:
    result = run_ssh(
        config,
        ["squeue", "-h", "-u", config.user, "-o", "%i|%j|%T|%M|%l|%D|%R"],
        capture_output=True,
    )
    return "JOBID|NAME|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)\n" + result.stdout.strip()


def queue_status(config: Config, partition: str | None = None) -> str:
    command = ["squeue", "-h", "-o", "%i|%u|%j|%T|%M|%l|%D|%C|%m|%R"]
    if partition:
        command.extend(["-p", partition])
    result = run_ssh(config, command, capture_output=True)
    header = "JOBID|USER|NAME|STATE|TIME|TIME_LIMIT|NODES|CPUS|MIN_MEMORY|NODELIST(REASON)\n"
    return header + result.stdout.strip()


def run_health_checks(config: Config) -> str:
    health_script = (
        "set -euo pipefail; "
        "echo '== hostname =='; hostname; "
        "echo '== sinfo =='; sinfo -o '%P %a %l %D %G %m'"
    )
    result = run_ssh(
        config,
        ["bash", "-lc", health_script],
        capture_output=True,
    )
    return result.stdout
