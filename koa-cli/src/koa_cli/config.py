from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("~/.config/koa-cli/config.yaml").expanduser()

# File name searched in the current and parent directories for project-local overrides.
PROJECT_LOCAL_CONFIG_NAME = ".koa.yaml"

PROJECT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

# Valid Slurm time formats:  minutes | HH:MM:SS | D-HH:MM:SS | D-HH:MM | D-HH | MM:SS
SLURM_TIME_PATTERN = re.compile(
    r"^\d+$"                      # plain minutes
    r"|^\d+:\d{2}$"               # MM:SS or HH:MM
    r"|^\d+:\d{2}:\d{2}$"         # HH:MM:SS
    r"|^\d+-\d{2}$"               # D-HH
    r"|^\d+-\d{2}:\d{2}$"         # D-HH:MM
    r"|^\d+-\d{2}:\d{2}:\d{2}$"  # D-HH:MM:SS
)


def validate_slurm_time(value: str) -> str:
    """Return *value* if it is a valid Slurm time string, else raise ValueError."""
    if not SLURM_TIME_PATTERN.match(value.strip()):
        raise ValueError(
            f"Invalid Slurm time format: {value!r}\n"
            "Expected one of: minutes, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS"
        )
    return value.strip()


@dataclass(frozen=True)
class Config:
    """Configuration for connecting to the KOA cluster."""

    user: str
    host: str
    remote_workdir: Path
    remote_data_dir: Path
    identity_file: Path | None = None
    proxy_command: str | None = None

    # ---- optional cluster defaults (all overridable per-command) ----
    default_partition: str = "gpu"
    default_gpu: str = "rtx2080ti"
    default_time: str | None = None
    default_results_limit: int = 20

    @property
    def login(self) -> str:
        return f"{self.user}@{self.host}"

    def project_name(self, value: str) -> str:
        name = value.strip()
        if not name or name in {".", ".."} or not PROJECT_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid project name: {value!r}")
        return name

    def project_workdir(self, project_name: str) -> Path:
        return self.remote_workdir / self.project_name(project_name)

    def project_data_dir(self, project_name: str) -> Path:
        return self.remote_data_dir / self.project_name(project_name)

    def project_results_dir(self, project_name: str, kind: str) -> Path:
        if kind not in {"train", "eval"}:
            raise ValueError(f"Unsupported results kind: {kind}")
        return self.project_data_dir(project_name) / kind / "results"


def _find_project_local_config(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default: cwd) looking for a .koa.yaml file."""
    current = (start or Path.cwd()).resolve()
    home = Path.home()
    while True:
        candidate = current / PROJECT_LOCAL_CONFIG_NAME
        if candidate.exists():
            return candidate
        if current == home or current == current.parent:
            return None
        current = current.parent


def load_config(
    config_path: os.PathLike[str] | str | None = None,
    *,
    project_dir: Path | None = None,
) -> Config:
    """Load configuration from disk and apply environment and project-local overrides.

    Resolution order (later values win):
      1. Global config file (~/.config/koa-cli/config.yaml or *config_path*)
      2. Project-local .koa.yaml (searched from *project_dir* / cwd upward)
      3. Environment variables (KOA_*)
    """
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {path}. "
            "Run `koa setup` or create it from config.example.yaml."
        )

    with path.open("r", encoding="utf-8") as handle:
        data: dict = yaml.safe_load(handle) or {}

    # Merge project-local overrides (non-destructive: only keys present in the file).
    local_cfg_path = _find_project_local_config(project_dir)
    if local_cfg_path:
        with local_cfg_path.open("r", encoding="utf-8") as handle:
            local_data: dict = yaml.safe_load(handle) or {}
        data.update({k: v for k, v in local_data.items() if v is not None})

    # Environment variable overrides.
    for key, env_var in {
        "user": "KOA_USER",
        "host": "KOA_HOST",
        "identity_file": "KOA_IDENTITY_FILE",
        "remote_workdir": "KOA_REMOTE_WORKDIR",
        "remote_data_dir": "KOA_REMOTE_DATA_DIR",
        "proxy_command": "KOA_PROXY_COMMAND",
        "default_partition": "KOA_DEFAULT_PARTITION",
        "default_gpu": "KOA_DEFAULT_GPU",
        "default_time": "KOA_DEFAULT_TIME",
        "default_results_limit": "KOA_DEFAULT_RESULTS_LIMIT",
    }.items():
        value = os.getenv(env_var)
        if value:
            data[key] = value

    missing = [key for key in ("user", "host") if not data.get(key)]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    identity_path: Path | None = None
    if data.get("identity_file"):
        identity_path = Path(data["identity_file"]).expanduser()
        if not identity_path.exists():
            raise FileNotFoundError(f"Configured identity_file not found: {identity_path}")

    remote_workdir = Path(str(data.get("remote_workdir", "~/koa-jobs")))
    remote_data_default = f"/mnt/lustre/koa/scratch/{data['user']}/koa-jobs"
    remote_data_dir = Path(str(data.get("remote_data_dir", remote_data_default)))

    # Validate default_time if set.
    # NOTE: PyYAML (1.1 schema) parses bare HH:MM:SS values as integers (seconds).
    # We always coerce to str so the user can write either form in their YAML.
    raw_default_time = data.get("default_time")
    default_time: str | None = None
    if raw_default_time is not None and str(raw_default_time).strip():
        default_time = validate_slurm_time(str(raw_default_time).strip())

    return Config(
        user=str(data["user"]).strip(),
        host=str(data["host"]).strip(),
        remote_workdir=remote_workdir,
        remote_data_dir=remote_data_dir,
        identity_file=identity_path,
        proxy_command=(
            str(data["proxy_command"]).strip() if data.get("proxy_command") else None
        ),
        default_partition=str(data.get("default_partition", "gpu")).strip(),
        default_gpu=str(data.get("default_gpu", "rtx2080ti")).strip(),
        default_time=default_time,
        default_results_limit=int(data.get("default_results_limit", 20)),
    )
