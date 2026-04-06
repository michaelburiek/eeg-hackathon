"""
Shared YAML config loading utilities.

Supports:
- recursive ``defaults`` resolution relative to the current YAML file
- deep merging for nested dictionaries
- dot-notation updates for sweep/ablation overrides
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable

import yaml


def deep_merge(base: dict | None, override: dict | None) -> dict:
    """Recursively merge two dictionaries, with ``override`` taking precedence."""
    base = deepcopy(base or {})
    override = override or {}

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _resolve_default_path(default_entry: str, current_path: Path) -> Path:
    """Resolve a ``defaults`` entry relative to the current YAML file."""
    candidate = Path(default_entry)
    if not candidate.is_absolute():
        candidate = current_path.parent / candidate

    if candidate.exists():
        return candidate.resolve()

    if candidate.suffix == "":
        yaml_candidate = candidate.with_suffix(".yaml")
        if yaml_candidate.exists():
            return yaml_candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve config default '{default_entry}' from {current_path}"
    )


def load_config(path: str | Path, seen: set[Path] | None = None) -> dict:
    """
    Load a YAML config, recursively resolving ``defaults`` entries.

    Later files override earlier ones, and nested dictionaries are merged deeply.
    """
    cfg_path = Path(path).resolve()
    seen = seen or set()
    if cfg_path in seen:
        raise ValueError(f"Cyclic config defaults detected at {cfg_path}")

    with open(cfg_path) as f:
        raw_cfg = yaml.safe_load(f) or {}

    defaults = raw_cfg.pop("defaults", []) or []
    merged: dict = {}
    next_seen = seen | {cfg_path}

    for entry in defaults:
        if isinstance(entry, str):
            default_path = _resolve_default_path(entry, cfg_path)
        elif isinstance(entry, dict) and "config" in entry:
            default_path = _resolve_default_path(str(entry["config"]), cfg_path)
        else:
            raise ValueError(f"Unsupported defaults entry in {cfg_path}: {entry!r}")
        merged = deep_merge(merged, load_config(default_path, seen=next_seen))

    return deep_merge(merged, raw_cfg)


def load_and_merge_configs(*paths: str | Path) -> dict:
    """Load and deep-merge multiple configs in order."""
    merged: dict = {}
    for path in paths:
        merged = deep_merge(merged, load_config(path))
    return merged


def set_nested(cfg: dict, dotkey: str, value) -> None:
    """Set a nested config key using dot notation, e.g. ``training.lr``."""
    keys = dotkey.split(".")
    current = cfg
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value
