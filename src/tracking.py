"""
Optional experiment tracking helpers.

W&B logging is treated as best-effort so training can continue even when the
package is not installed or credentials are missing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

log = logging.getLogger(__name__)

_WARNED_MESSAGES: set[str] = set()
_WANDB_ENV_KEYS = ("WANDDB_API_KEY", "WANDB_API_KEY", "WAND_DB_API_KEY")


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(key)
    log.warning(message)


def _get_repo_wandb_api_key() -> tuple[str | None, str | None]:
    for env_key in _WANDB_ENV_KEYS:
        value = os.getenv(env_key)
        if value:
            return env_key, value
    return None, None


def _wandb_requested(cfg: Mapping[str, Any]) -> bool:
    wandb_cfg = cfg.get("wandb", {})
    enabled = wandb_cfg.get("enabled")
    if enabled is False:
        return False
    if enabled is True:
        return True
    _, api_key = _get_repo_wandb_api_key()
    return bool(api_key)


def _ensure_wandb_api_key() -> bool:
    env_key, api_key = _get_repo_wandb_api_key()
    if not api_key:
        _warn_once(
            "wandb_missing_key",
            "W&B logging was requested, but no WANDDB_API_KEY was found. Skipping W&B.",
        )
        return False

    if env_key == "WANDDB_API_KEY":
        os.environ["WANDB_API_KEY"] = api_key
        return True

    if env_key == "WANDB_API_KEY":
        _warn_once(
            "wandb_standard_key_alias",
            "Using WANDB_API_KEY for W&B auth; prefer WANDDB_API_KEY in this repo.",
        )
        return True

    if env_key == "WAND_DB_API_KEY":
        os.environ["WANDB_API_KEY"] = api_key
        _warn_once(
            "wandb_legacy_key",
            "Using legacy env var WAND_DB_API_KEY for W&B auth; prefer WANDDB_API_KEY.",
        )
        return True

    return True


def _import_wandb():
    try:
        import wandb  # type: ignore
    except ImportError:
        _warn_once(
            "wandb_missing_package",
            "wandb is not installed, so W&B logging is disabled. Install `wandb` to enable it.",
        )
        return None
    return wandb


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def init_wandb_run(
    cfg: Mapping[str, Any],
    run_name: str,
    job_type: str,
    *,
    group: str | None = None,
    extra_config: Mapping[str, Any] | None = None,
    tags: list[str] | None = None,
):
    if not _wandb_requested(cfg):
        return None
    if not _ensure_wandb_api_key():
        return None

    wandb = _import_wandb()
    if wandb is None:
        return None

    wandb_cfg = cfg.get("wandb", {})
    run_dir = Path(wandb_cfg.get("dir", "experiments/wandb"))
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = _json_safe(dict(cfg))
    if extra_config:
        config_payload.update(_json_safe(dict(extra_config)))

    entity = os.getenv("WANDB_ENTITY") or wandb_cfg.get("entity")
    project = (
        os.getenv("WANDB_PROJECT")
        or wandb_cfg.get("project")
        or cfg.get("project", {}).get("name")
        or "eeg-hackathon"
    )
    run_group = group
    if run_group is None and wandb_cfg.get("group_by_experiment", True):
        run_group = cfg.get("experiment", {}).get("name")

    try:
        return wandb.init(
            entity=entity,
            project=project,
            name=run_name,
            group=run_group,
            job_type=job_type,
            dir=str(run_dir),
            config=config_payload,
            tags=tags or list(wandb_cfg.get("tags", [])),
            reinit=True,
        )
    except Exception as exc:  # pragma: no cover - network/auth dependent
        _warn_once(
            "wandb_init_failed",
            f"Failed to initialize W&B logging ({exc}). Continuing without W&B.",
        )
        return None


def log_wandb_metrics(run, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
    if run is None:
        return

    payload = {}
    for key, value in metrics.items():
        safe_value = _json_safe(value)
        if isinstance(safe_value, (int, float)):
            payload[key] = safe_value

    if not payload:
        return

    try:
        run.log(payload, step=step)
    except Exception as exc:  # pragma: no cover - network/auth dependent
        _warn_once(
            "wandb_log_failed",
            f"Failed to log metrics to W&B ({exc}). Continuing without further W&B updates.",
        )


def set_wandb_summary(run, metrics: Mapping[str, Any]) -> None:
    if run is None:
        return

    try:
        for key, value in metrics.items():
            safe_value = _json_safe(value)
            if isinstance(safe_value, (int, float, str)):
                run.summary[key] = safe_value
    except Exception as exc:  # pragma: no cover - network/auth dependent
        _warn_once(
            "wandb_summary_failed",
            f"Failed to update W&B summary ({exc}).",
        )


def finish_wandb_run(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:  # pragma: no cover - network/auth dependent
        _warn_once("wandb_finish_failed", f"Failed to finish W&B run cleanly ({exc}).")
