"""Public package exports for koa-cli."""

from importlib.metadata import PackageNotFoundError, version

from .config import DEFAULT_CONFIG_PATH, Config, load_config
from .history import get_latest_job_id, list_history, record_job
from .ssh import SSHError

try:
    __version__ = version("koa-cli")
except PackageNotFoundError:  # package not installed (e.g. running from source)
    __version__ = "0.3.0"

__all__ = [
    "Config",
    "DEFAULT_CONFIG_PATH",
    "SSHError",
    "get_latest_job_id",
    "list_history",
    "load_config",
    "record_job",
]
