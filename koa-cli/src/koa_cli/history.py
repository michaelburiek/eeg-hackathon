"""Local job history — records every submitted job so you can reference it later."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

HISTORY_PATH = Path("~/.config/koa-cli/history.json").expanduser()
_MAX_HISTORY = 500  # cap file size


class JobRecord(TypedDict):
    job_id: str
    project: str
    script: str
    submitted_at: str  # ISO-8601 UTC


def _load() -> list[JobRecord]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save(records: list[JobRecord]) -> None:
    """Write *records* to disk atomically using a temp-file + rename.

    This prevents a corrupt history file if the process is killed mid-write.
    """
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(records[-_MAX_HISTORY:], indent=2)
    fd, tmp_path = tempfile.mkstemp(dir=HISTORY_PATH.parent, prefix=".koa-history-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp_path, HISTORY_PATH)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def record_job(job_id: str, project: str, script: str) -> None:
    """Append a new job submission to the local history file."""
    records = _load()
    records.append(
        JobRecord(
            job_id=job_id,
            project=project,
            script=script,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )
    )
    _save(records)


def list_history(project: str | None = None, limit: int = 20) -> list[JobRecord]:
    """Return recent job records, optionally filtered by project name."""
    records = _load()
    if project:
        records = [r for r in records if r["project"] == project]
    return records[-limit:][::-1]  # newest first


def get_latest_job_id(project: str | None = None) -> str | None:
    """Return the job ID of the most recently submitted job (optionally per-project)."""
    records = list_history(project=project, limit=1)
    return records[0]["job_id"] if records else None
